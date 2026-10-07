"""Compare exact-author-filter FBCNet against original local filter and Core."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from analyze_cue_aligned_2a import load_complete_model, paired_interval, sha256


def subject_values(rows: list[dict]) -> np.ndarray:
    return np.array([
        np.mean([float(row["test_accuracy"]) for row in rows
                 if int(row["subject"]) == subject])
        for subject in range(1, 10)
    ])


def comparison(left: np.ndarray, right: np.ndarray) -> dict:
    difference = left - right
    lower, upper = paired_interval(difference)
    test = wilcoxon(difference, alternative="two-sided")
    return {"mean_delta_pp": 100 * float(difference.mean()),
            "bootstrap_95ci_pp": [100 * lower, 100 * upper],
            "wins": int(np.sum(difference > 0)),
            "wilcoxon_p": float(test.pvalue)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    args = parser.parse_args()
    audit = args.audit_dir.resolve()
    status = (audit / "results" / "fbcnet_author_filter_status.tsv").read_text()
    if "\tDONE" not in status:
        raise RuntimeError("Filterfix 9 × 3 run has not finished")
    models = {}
    for name, network in (("core", "NPFEEGNet"), ("fbcnet", "FBCNet"),
                          ("fbcnet_author_filter", "FBCNet")):
        rows, source = load_complete_model(audit, name, network)
        scores = subject_values(rows)
        models[name] = {"source": source, "mean_percent": 100 * float(scores.mean()),
                        "subject_sd_percent": 100 * float(scores.std(ddof=1)),
                        "subject_accuracies_percent": (100 * scores).tolist()}
    new = np.array(models["fbcnet_author_filter"]["subject_accuracies_percent"]) / 100
    old = np.array(models["fbcnet"]["subject_accuracies_percent"]) / 100
    core = np.array(models["core"]["subject_accuracies_percent"]) / 100
    report = {
        "models": models,
        "author_filter_minus_old_filter": comparison(new, old),
        "core_minus_author_filter_fbcnet": comparison(core, new),
        "filterfix_code_sha256": sha256(audit / "local_protocol_filterfix" / "model" / "baselines.py"),
        "interpretation": "Only the FBCNet Chebyshev-II design Wn changed from cheb2ord ws to author fStop; training and cue-aligned data unchanged.",
    }
    target = audit / "results" / "fbcnet_author_filter_comparison.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
