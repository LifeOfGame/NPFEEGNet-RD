"""Audit the complete corrected BCIC-IV-2a three-model, nine-subject grid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


MODELS = {"core": "NPFEEGNet", "eegnet": "EEGNet", "fbcnet": "FBCNet"}
EXPECTED = {(subject, seed) for subject in range(1, 10) for seed in range(3)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_complete_model(audit_dir: Path, model: str, network: str) -> tuple[list[dict], dict]:
    root = audit_dir / "output" / f"{model}_cue_aligned" / network
    candidates = []
    for path in root.glob("*/summary.csv"):
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        observed = {(int(row["subject"]), int(row["seed"])) for row in rows}
        if len(rows) == 27 and observed == EXPECTED:
            candidates.append((path, rows))
    if len(candidates) != 1:
        raise RuntimeError(f"{model}: expected one complete 27-run summary, found {len(candidates)}")
    path, rows = candidates[0]
    if len({(row["subject"], row["seed"]) for row in rows}) != 27:
        raise RuntimeError(f"{model}: duplicate subject/seed rows")
    if not all(0 <= float(row["test_accuracy"]) <= 1 and
               np.isfinite(float(row["test_kappa"])) for row in rows):
        raise RuntimeError(f"{model}: test accuracy outside [0, 1]")
    expected_data = str(audit_dir / "data" / "2a_cue_aligned")
    for subject, seed in sorted(EXPECTED):
        config = path.parent / f"sub{subject}" / f"seed{seed}" / "config.yaml"
        if not config.is_file() or f"data_path: {expected_data}" not in config.read_text():
            raise RuntimeError(f"{model}: missing or wrong data path in {config}")
    rows.sort(key=lambda row: (int(row["subject"]), int(row["seed"])))
    return rows, {"path": str(path), "sha256": sha256(path)}


def paired_interval(differences: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(2026)
    means = rng.choice(differences, size=(20_000, len(differences)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.percentile(means, [2.5, 97.5]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    args = parser.parse_args()
    audit = args.audit_dir.resolve()
    status = (audit / "results" / "cue_aligned_queue_status.tsv").read_text()
    if "\tALL_DONE" not in status:
        raise RuntimeError("Full queue has not finished successfully")
    manifest = json.loads((audit / "data" / "2a_cue_aligned" / "manifest.json").read_text())
    if len(manifest["sessions"]) != 18 or not all(
        session["old_export_matches_trial_start_exactly"] for session in manifest["sessions"]
    ):
        raise RuntimeError("Data manifest is incomplete")
    outputs = {}
    subject_rows = []
    for model, network in MODELS.items():
        rows, source = load_complete_model(audit, model, network)
        subjects = []
        for subject in range(1, 10):
            subset = [row for row in rows if int(row["subject"]) == subject]
            accuracies = np.array([float(row["test_accuracy"]) for row in subset])
            kappas = np.array([float(row["test_kappa"]) for row in subset])
            subjects.append({"model": model, "subject": subject,
                             "accuracy": float(accuracies.mean()),
                             "seed_sd": float(accuracies.std(ddof=1)),
                             "kappa": float(kappas.mean())})
        subject_rows.extend(subjects)
        accuracy_values = np.array([row["accuracy"] for row in subjects])
        outputs[model] = {
            "source": source,
            "accuracy_mean": float(accuracy_values.mean()),
            "accuracy_across_subject_sd": float(accuracy_values.std(ddof=1)),
            "kappa_mean": float(np.mean([row["kappa"] for row in subjects])),
            "seed_sd_mean": float(np.mean([row["seed_sd"] for row in subjects])),
            "subject_accuracies": accuracy_values.tolist(),
        }
    paired = {}
    for first, second in (("core", "eegnet"), ("core", "fbcnet"), ("fbcnet", "eegnet")):
        left = np.array(outputs[first]["subject_accuracies"])
        right = np.array(outputs[second]["subject_accuracies"])
        differences = left - right
        lower, upper = paired_interval(differences)
        test = wilcoxon(differences, alternative="two-sided")
        paired[f"{first}_minus_{second}"] = {
            "mean_delta": float(differences.mean()),
            "bootstrap_95ci": [lower, upper],
            "wilcoxon_w": float(test.statistic),
            "wilcoxon_p": float(test.pvalue),
            "wins": int((differences > 0).sum()),
        }
    report = {"data_manifest_sha256": sha256(audit / "data" / "2a_cue_aligned" / "manifest.json"),
              "models": outputs, "paired": paired,
              "caveat": "Audited rerun after earlier trial-start AxxE results had been viewed; not pristine prospective confirmation."}
    result_dir = audit / "results"
    (result_dir / "cue_aligned_three_model_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    with (result_dir / "cue_aligned_three_model_subjects.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "subject", "accuracy", "seed_sd", "kappa"])
        writer.writeheader()
        writer.writerows(subject_rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
