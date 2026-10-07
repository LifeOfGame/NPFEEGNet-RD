"""Build an isolated BCIC-IV-2a cue-aligned export from MOABB MAT caches.

The MAT ``trial`` field is the trial-start event. The published paradigm places
the motor-imagery cue 2 s later; at 250 Hz that is a 500-sample shift. This
script verifies the earlier 0--4 s export against the MAT source before making
new 2--6 s epochs. It never modifies the earlier export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


SAMPLE_RATE = 250
CUE_OFFSET = 2 * SAMPLE_RATE
EPOCH_SAMPLES = 4 * SAMPLE_RATE
EEG_CHANNELS = 22


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_session(mat_path: Path, old_dir: Path, output_dir: Path) -> dict:
    key = mat_path.stem
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    runs = np.atleast_1d(mat["data"]).reshape(-1)
    old_x = np.load(old_dir / f"{key}_data.npy", mmap_mode="r")
    old_y = np.load(old_dir / f"{key}_label.npy").reshape(-1)
    old_g = np.load(old_dir / f"{key}_group.npy").reshape(-1)
    new_x, new_y, new_g = [], [], []
    original_mismatches = []
    task_runs = [run for run in runs if np.atleast_1d(run.trial).size]
    assert len(task_runs) == 6, (key, len(task_runs))
    for group, run in enumerate(task_runs):
        signal = np.asarray(run.X)
        onsets = np.atleast_1d(run.trial).astype(np.int64).reshape(-1) - 1
        labels = np.atleast_1d(run.y).astype(np.int64).reshape(-1)
        assert len(onsets) == len(labels) == 48, (key, group, len(onsets))
        assert signal.shape[1] >= EEG_CHANNELS
        for onset, label in zip(onsets, labels, strict=True):
            idx = len(new_x)
            assert 0 <= onset and onset + CUE_OFFSET + EPOCH_SAMPLES <= len(signal), (key, idx)
            original = signal[onset : onset + EPOCH_SAMPLES, :EEG_CHANNELS].T.astype(np.float32)
            if not np.array_equal(original, old_x[idx]):
                original_mismatches.append(idx)
            new_x.append(signal[onset + CUE_OFFSET : onset + CUE_OFFSET + EPOCH_SAMPLES,
                                :EEG_CHANNELS].T.astype(np.float32))
            new_y.append(int(label))
            new_g.append(group)
    x = np.stack(new_x)
    y = np.asarray(new_y, dtype=np.int64)
    g = np.asarray(new_g, dtype=np.int64)
    assert x.shape == old_x.shape == (288, EEG_CHANNELS, EPOCH_SAMPLES)
    assert np.array_equal(y, old_y) and np.array_equal(g, old_g), key
    assert np.array_equal(np.bincount(y, minlength=5)[1:], [72] * 4), key
    assert np.isfinite(x).all(), key
    assert not original_mismatches, (key, original_mismatches[:10])
    output_paths = {}
    for suffix, arr in (("data", x), ("label", y[:, None]), ("group", g)):
        target = output_dir / f"{key}_{suffix}.npy"
        np.save(target, arr)
        output_paths[target.name] = sha256(target)
    return {"session": key, "source_sha256": sha256(mat_path),
            "old_export_matches_trial_start_exactly": True,
            "new_shape": list(x.shape), "class_counts": [72] * 4,
            "output_sha256": output_paths}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat-dir", type=Path, required=True)
    parser.add_argument("--old-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 10)))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sessions = []
    for subject in args.subjects:
        for session in ("T", "E"):
            key = f"A{subject:02d}{session}"
            result = export_session(args.mat_dir / f"{key}.mat", args.old_dir, args.output_dir)
            sessions.append(result)
            print(f"{key}: 288 verified trial-start epochs; cue-aligned export complete", flush=True)
    manifest = {"sample_rate_hz": SAMPLE_RATE, "cue_offset_samples": CUE_OFFSET,
                "epoch_samples": EPOCH_SAMPLES, "subjects": args.subjects,
                "source_interpretation": "MAT trial=start; cue at +2s per official BCIC-IV-2a protocol",
                "sessions": sessions}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(sessions)} sessions to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
