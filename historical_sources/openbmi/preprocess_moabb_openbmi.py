"""Prepare the frozen OpenBMI S1-to-S2 confirmation dataset.

The OpenBMI MI files contain two labelled phases per recording session:
``EEG_MI_train`` and ``EEG_MI_test``.  Following recent fixed-direction
cross-session studies, both phases are used, giving 200 balanced trials in
session 1 and 200 in session 2 for every subject.  Session 1 is exported as
``OxxT`` and session 2 as ``OxxE``.

Research-integrity guardrail: session 2 cannot be downloaded or processed by
this script until the no-demean and raw-demean epoch maps have both been
frozen from session-1-only selection runs.  The freeze script creates the
cryptographic unlock file checked here.

Examples
--------
Download and prepare session 1 only::

    python -u data/openbmi/preprocess_moabb_openbmi.py --role selection

After both selection runs and ``freeze_openbmi_selection.py`` complete,
prepare session 2::

    python -u data/openbmi/preprocess_moabb_openbmi.py --role evaluation
"""

from __future__ import annotations

import argparse
import errno
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import requests
from moabb.datasets import Lee2019_MI
from scipy.io import loadmat
from scipy.signal import resample_poly
from tqdm.auto import tqdm


OPENBMI_CHANNELS = (
    "FC5",
    "FC3",
    "FC1",
    "FC2",
    "FC4",
    "FC6",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP5",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "CP6",
)
PHASE_KEYS = ("EEG_MI_train", "EEG_MI_test")
TARGET_SAMPLING_RATE = 250.0
EPOCH_SECONDS = (0.0, 4.0)
EXPECTED_TRIALS_PER_PHASE = 100
EXPECTED_TRIALS_PER_CLASS_PER_PHASE = 50
DEFAULT_UNLOCK = Path("results/openbmi_evaluation_unlock.json")
LEE2019_URL = (
    "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/"
    "pub/10.5524/100001_101000/100542"
)
DOWNLOAD_ATTEMPTS = 12
DOWNLOAD_CHUNK_BYTES = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_evaluation_unlock(path: Path, repository_root: Path) -> dict[str, Any]:
    """Refuse session-2 access unless all frozen artifacts still match."""
    if not path.is_file():
        raise RuntimeError(
            "Session 2 is locked. Complete both session-1-only selection runs "
            "and run scripts/freeze_openbmi_selection.py first. Missing: "
            f"{path}"
        )
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if payload.get("status") != "ready_for_session2":
        raise RuntimeError(f"Invalid OpenBMI evaluation unlock status in {path}")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RuntimeError(f"OpenBMI evaluation unlock has no artifacts: {path}")
    for artifact in artifacts:
        artifact_path = Path(artifact["path"])
        if not artifact_path.is_absolute():
            artifact_path = repository_root / artifact_path
        if not artifact_path.is_file():
            raise RuntimeError(
                f"Frozen OpenBMI artifact disappeared after locking: {artifact_path}"
            )
        observed = sha256_file(artifact_path)
        if observed != artifact["sha256"]:
            raise RuntimeError(
                "Frozen OpenBMI artifact changed after session-1 selection: "
                f"{artifact_path}. Expected {artifact['sha256']}, got {observed}."
            )
    return payload


def _matlab_string(value: Any) -> str:
    squeezed = np.squeeze(value)
    if squeezed.ndim == 0:
        return str(squeezed.item()).strip()
    if squeezed.dtype.kind in {"U", "S"}:
        return "".join(str(item) for item in squeezed.flat).strip()
    return str(squeezed.tolist()).strip()


def channel_names(eeg_struct: np.void) -> list[str]:
    names = [_matlab_string(item) for item in np.ravel(eeg_struct["chan"])]
    if len(names) != len(set(names)):
        raise RuntimeError("OpenBMI MAT contains duplicate EEG channel names")
    return names


def extract_phase(
    eeg_struct: np.void,
    selected_channels: tuple[str, ...] = OPENBMI_CHANNELS,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Extract one 100-trial phase without filtering or label-dependent steps."""
    signal = np.asarray(eeg_struct["x"])
    onsets = np.asarray(eeg_struct["t"]).squeeze().astype(np.int64)
    labels = np.asarray(eeg_struct["y_dec"]).squeeze().astype(np.int64)
    sampling_rate = float(np.asarray(eeg_struct["fs"]).squeeze())
    names = channel_names(eeg_struct)
    missing = sorted(set(selected_channels) - set(names))
    if missing:
        raise RuntimeError(f"OpenBMI file is missing frozen channels: {missing}")
    indices = np.asarray([names.index(name) for name in selected_channels])

    if signal.ndim != 2 or signal.shape[1] != len(names):
        raise RuntimeError(
            f"Expected OpenBMI continuous data (samples, channels), got {signal.shape}"
        )
    if len(onsets) != EXPECTED_TRIALS_PER_PHASE or len(labels) != len(onsets):
        raise RuntimeError(
            f"Expected {EXPECTED_TRIALS_PER_PHASE} trials, got "
            f"onsets={len(onsets)}, labels={len(labels)}"
        )
    if set(np.unique(labels).tolist()) != {1, 2}:
        raise RuntimeError(f"Expected OpenBMI labels 1/2, got {np.unique(labels)}")
    class_counts = np.bincount(labels, minlength=3)[1:3]
    if not np.all(class_counts == EXPECTED_TRIALS_PER_CLASS_PER_PHASE):
        raise RuntimeError(
            "OpenBMI phase is not 50/50 balanced: "
            f"class counts={class_counts.tolist()}"
        )

    source_samples = int(round((EPOCH_SECONDS[1] - EPOCH_SECONDS[0]) * sampling_rate))
    windows = []
    for onset in onsets:
        start = int(onset + round(EPOCH_SECONDS[0] * sampling_rate))
        stop = start + source_samples
        if start < 0 or stop > signal.shape[0]:
            raise RuntimeError(
                f"OpenBMI epoch [{start}, {stop}) exceeds signal length {signal.shape[0]}"
            )
        # The MAT signal is already expressed in microvolts.  Cast before
        # stacking so a large continuous float64 recording is not duplicated.
        windows.append(
            np.asarray(signal[start:stop, indices].T, dtype=np.float32)
        )
    data = np.stack(windows, axis=0)

    ratio = Fraction(TARGET_SAMPLING_RATE / sampling_rate).limit_denominator(10000)
    data = resample_poly(data, ratio.numerator, ratio.denominator, axis=-1)
    data = np.asarray(data, dtype=np.float32)
    expected_samples = int(
        round((EPOCH_SECONDS[1] - EPOCH_SECONDS[0]) * TARGET_SAMPLING_RATE)
    )
    if data.shape != (
        EXPECTED_TRIALS_PER_PHASE,
        len(selected_channels),
        expected_samples,
    ):
        raise RuntimeError(f"Unexpected OpenBMI epoch shape after resampling: {data.shape}")
    if not np.isfinite(data).all():
        raise RuntimeError("OpenBMI epochs contain NaN or infinity")

    metadata = {
        "shape": list(data.shape),
        "source_sampling_rate": sampling_rate,
        "target_sampling_rate": TARGET_SAMPLING_RATE,
        "class_counts": class_counts.tolist(),
        "label_mapping": {"1": "right_hand", "2": "left_hand"},
    }
    return data, labels, metadata


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        np.save(file, array)
    os.replace(temporary, path)


def atomic_save_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def canonical_session_path(cache_path: Path, session: int, subject: int) -> Path:
    """Use MOABB's cache layout without its Windows drive-colon sanitizing bug."""
    return (
        cache_path
        / "MNE-lee2019-mi-data"
        / "gigadb-datasets"
        / "live"
        / "pub"
        / "10.5524"
        / "100001_101000"
        / "100542"
        / f"session{session}"
        / f"s{subject}"
        / f"sess{session:02d}_subj{subject:02d}_EEG_MI.mat"
    )


def download_with_resume(url: str, destination: Path) -> Path:
    """Download one OpenBMI MAT with visible progress and byte-range resume."""
    if destination.is_file():
        print(f"Using cached OpenBMI file: {destination}", flush=True)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {
            "User-Agent": "MOABB-OpenBMI-confirmatory-downloader/1.0",
            **({"Range": f"bytes={offset}-"} if offset else {}),
        }
        try:
            with requests.get(
                url,
                headers=headers,
                stream=True,
                allow_redirects=True,
                timeout=(30, 300),
            ) as response:
                response.raise_for_status()
                resumed = offset > 0 and response.status_code == 206
                if offset and not resumed:
                    print(
                        "OpenBMI server did not accept Range; restarting this MAT.",
                        flush=True,
                    )
                    offset = 0

                content_range = response.headers.get("Content-Range", "")
                if "/" in content_range:
                    total = int(content_range.rsplit("/", 1)[1])
                else:
                    remaining = int(response.headers.get("Content-Length", 0))
                    total = offset + remaining if remaining else None
                mode = "ab" if resumed else "wb"
                with partial.open(mode) as file, tqdm(
                    total=total,
                    initial=offset,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"S{session_from_url(url)} subj{subject_from_url(url):02d}",
                    dynamic_ncols=True,
                ) as progress:
                    for chunk in response.iter_content(DOWNLOAD_CHUNK_BYTES):
                        if chunk:
                            file.write(chunk)
                            progress.update(len(chunk))

            if total is not None and partial.stat().st_size != total:
                raise IOError(
                    f"Incomplete OpenBMI download: {partial.stat().st_size} of {total} bytes"
                )
            os.replace(partial, destination)
            print(f"Saved {destination}", flush=True)
            return destination
        except OSError as error:
            if error.errno == errno.ENOSPC:
                raise RuntimeError(
                    f"Disk full; partial OpenBMI file retained at {partial}"
                ) from error
            failure: Exception = error
        except requests.RequestException as error:
            failure = error

        if attempt == DOWNLOAD_ATTEMPTS:
            raise RuntimeError(
                f"OpenBMI download failed after {DOWNLOAD_ATTEMPTS} attempts; "
                f"partial retained at {partial}"
            ) from failure
        delay = min(60, 5 * attempt)
        retained = partial.stat().st_size if partial.exists() else 0
        print(
            f"Download interrupted ({failure}); retained {retained / 2**20:.1f} MiB. "
            f"Retrying in {delay}s [{attempt}/{DOWNLOAD_ATTEMPTS}]...",
            flush=True,
        )
        time.sleep(delay)
    raise AssertionError("unreachable")


def session_from_url(url: str) -> int:
    marker = "/session"
    return int(url.split(marker, 1)[1].split("/", 1)[0])


def subject_from_url(url: str) -> int:
    return int(url.rsplit("subj", 1)[1].split("_", 1)[0])


def prepare_subject(
    subject: int,
    role: str,
    cache_path: Path,
    output_path: Path,
    download_only: bool,
) -> None:
    session = 1 if role == "selection" else 2
    # Construct the dataset to validate the subject/session against the local
    # MOABB version, while using a resumable downloader for the 609-MB files.
    dataset = Lee2019_MI(
        train_run=True,
        test_run=True,
        sessions=[session],
        subjects=[subject],
    )
    if subject not in dataset.subject_list or session not in dataset.sessions:
        raise RuntimeError("Local MOABB rejected the requested OpenBMI subject/session")
    source_path = canonical_session_path(cache_path, session, subject)
    source_url = (
        f"{LEE2019_URL}/session{session}/s{subject}/"
        f"sess{session:02d}_subj{subject:02d}_EEG_MI.mat"
    )
    source_path = download_with_resume(source_url, source_path)
    print(
        f"OpenBMI subject {subject:02d}, session {session}: {source_path}",
        flush=True,
    )
    if download_only:
        return

    payload = loadmat(source_path)
    phase_data = []
    phase_labels = []
    phase_groups = []
    phase_metadata = []
    for group, key in enumerate(PHASE_KEYS):
        if key not in payload:
            raise RuntimeError(f"{source_path} does not contain {key}")
        data, labels, metadata = extract_phase(payload[key][0, 0])
        metadata["phase"] = key
        phase_data.append(data)
        phase_labels.append(labels)
        phase_groups.append(np.full(len(labels), group, dtype=np.int64))
        phase_metadata.append(metadata)

    data = np.concatenate(phase_data, axis=0)
    labels = np.concatenate(phase_labels, axis=0)
    groups = np.concatenate(phase_groups, axis=0)
    split_suffix = "T" if role == "selection" else "E"
    name = f"O{subject:02d}{split_suffix}"
    output_path.mkdir(parents=True, exist_ok=True)
    atomic_save_npy(output_path / f"{name}_data.npy", data)
    atomic_save_npy(output_path / f"{name}_label.npy", labels[:, None])
    atomic_save_npy(output_path / f"{name}_group.npy", groups)
    manifest = {
        "dataset": "Lee2019_MI/OpenBMI",
        "subject": subject,
        "source_session": session,
        "protocol_role": role,
        "output_name": name,
        "source_file": str(source_path.resolve()),
        "source_file_bytes": source_path.stat().st_size,
        "phases": phase_metadata,
        "channels": list(OPENBMI_CHANNELS),
        "epoch_seconds": list(EPOCH_SECONDS),
        "sampling_rate": TARGET_SAMPLING_RATE,
        "scale": "microvolts",
        "shape": list(data.shape),
        "class_counts": np.bincount(labels, minlength=3)[1:3].tolist(),
        "group_counts": np.bincount(groups).tolist(),
        "group_definition": "offline_train_phase=0; online_test_phase=1",
    }
    atomic_save_json(output_path / f"O{subject:02d}_session{session}_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--role",
        choices=("selection", "evaluation"),
        required=True,
        help="selection exports session 1 as OxxT; evaluation exports session 2 as OxxE",
    )
    parser.add_argument(
        "--subjects", nargs="+", type=int, default=list(range(1, 55))
    )
    parser.add_argument("--cache-path", type=Path, default=Path("database"))
    parser.add_argument(
        "--output-path", type=Path, default=Path("database/processed/openbmi")
    )
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--unlock-file", type=Path, default=DEFAULT_UNLOCK)
    arguments = parser.parse_args()

    invalid = sorted(set(arguments.subjects) - set(range(1, 55)))
    if invalid:
        raise ValueError(f"OpenBMI subjects must be 1-54; invalid={invalid}")
    repository_root = Path(__file__).resolve().parents[2]
    if arguments.role == "evaluation":
        verify_evaluation_unlock(arguments.unlock_file, repository_root)
        print(
            f"Session-2 lock verified: {arguments.unlock_file}", flush=True
        )
    for subject in arguments.subjects:
        prepare_subject(
            subject,
            arguments.role,
            arguments.cache_path,
            arguments.output_path,
            arguments.download_only,
        )


if __name__ == "__main__":
    main()
