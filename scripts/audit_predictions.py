"""Reconstruct descriptive metrics from anonymous frozen classifier outputs.

NLL and 10-bin ECE pool predictions; accuracy averages seeds within subject,
then subjects. This is an artifact audit, not a rerun or new hypothesis test.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def calibration(labels, probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if p.ndim != 2 or y.shape != (len(p),) or not len(y):
        raise ValueError('Invalid prediction dimensions')
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Invalid probability')
    if not np.allclose(p.sum(axis=1), 1, atol=1e-5, rtol=0):
        raise ValueError('Probabilities do not sum to one')
    if (y < 0).any() or (y >= p.shape[1]).any():
        raise ValueError('Invalid class label')
    nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == y
    ece = 0.0
    edges = np.linspace(0, 1, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        selected = (confidence > lo) & (confidence <= hi)
        if selected.any():
            ece += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return float(nll), float(ece)


def summarize(path):
    with np.load(path, allow_pickle=False) as archive:
        a = {k: archive[k] for k in archive.files}
    y, pred, p = a['labels'], a['predictions'], a['probabilities']
    if not np.array_equal(pred, p.argmax(axis=1)):
        raise ValueError(f'Argmax mismatch: {path}')
    nll, ece = calibration(y, p)
    runs = []
    for subject, seed in np.unique(np.column_stack([a['subject'], a['seed']]), axis=0):
        mask = (a['subject'] == subject) & (a['seed'] == seed)
        trials = a['trial'][mask]
        if not np.array_equal(trials, np.arange(len(trials))):
            raise ValueError(f'Duplicate or unordered run: {path}, {subject}, {seed}')
        runs.append(dict(subject=int(subject), seed=int(seed), trials=int(mask.sum()),
                         correct=int((pred[mask] == y[mask]).sum()),
                         accuracy=float((pred[mask] == y[mask]).mean()),
                         single_class_predictions=int(len(np.unique(pred[mask])) == 1)))
    subjects = sorted({r['subject'] for r in runs})
    macro = np.mean([np.mean([r['accuracy'] for r in runs if r['subject'] == s]) for s in subjects])
    return dict(subjects=len(subjects), runs=len(runs), trials=len(y),
                subject_macro_accuracy_pct=float(macro * 100), pooled_nll=nll,
                pooled_ece10_pct=ece * 100,
                single_class_prediction_runs=sum(r['single_class_predictions'] for r in runs)), runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = ROOT / 'evidence/predictions'
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    summaries, run_rows = [], []
    for group, entry in manifest['groups'].items():
        path = source / entry['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError(f'Hash mismatch: {path}')
        summary, runs = summarize(path)
        if summary['runs'] != entry['runs'] or summary['trials'] != entry['trials']:
            raise ValueError(f'Counts disagree: {group}')
        summaries.append(dict(group=group, **summary))
        run_rows.extend(dict(group=group, **r) for r in runs)
    for filename, rows in [('group_metrics.csv', summaries), ('run_metrics.csv', run_rows)]:
        with (args.output / filename).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f'PASS: {len(summaries)} groups, {len(run_rows)} runs; output: {args.output}')


if __name__ == '__main__':
    main()
