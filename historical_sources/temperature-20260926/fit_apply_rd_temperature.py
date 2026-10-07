"""Fit one temperature on S1 fold-out probabilities, then score frozen S2."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar


ROOT = Path('/mnt/helongfei/NPFEEGNet-RD-temperature-20260926')
RD = Path('/mnt/helongfei/NPFEEGNet-RD-unified-repro-20260918/output/openbmi_final_npf_raw_demean_3seed/NPFEEGNet/2026-08-31--14-52')
EXPECTED = {(s, z) for s in range(1, 55) for z in range(3)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path, expected_n):
    with np.load(path) as z:
        labels = z['labels'].astype(np.int64)
        probabilities = z['probabilities'].astype(np.float64)
    if labels.shape != (expected_n,) or probabilities.shape != (expected_n, 2):
        raise RuntimeError(f'Invalid shape: {path}')
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1, atol=2e-6):
        raise RuntimeError(f'Invalid probabilities: {path}')
    return labels, probabilities


def scale(probabilities, temperature):
    # Stable equivalent of softmax(logits/T) using saved softmax probabilities.
    scaled_logits = np.log(np.clip(probabilities, 1e-30, 1.0)) / temperature
    scaled_logits -= scaled_logits.max(axis=1, keepdims=True)
    exp = np.exp(scaled_logits)
    return exp / exp.sum(axis=1, keepdims=True)


def metrics(labels, probabilities):
    predicted = probabilities.argmax(1)
    confidence = probabilities.max(1)
    correct = predicted == labels
    nll = -np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1))
    ece = 0.0
    for low in np.linspace(0, 0.9, 10):
        mask = (confidence > low) & (confidence <= low + 0.1)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean() - confidence[mask].mean()))
    return {'accuracy': float(correct.mean()), 'nll': float(nll.mean()), 'ece': float(ece)}


def main():
    design_path = ROOT / 'results/design_lock.json'
    design = json.loads(design_path.read_text())
    if design['status'] != 'fixed_before_temperature_fit':
        raise RuntimeError('Missing design lock')
    for path, digest in design['sha256'].items():
        if sha(path) != digest:
            raise RuntimeError(f'Design artifact changed: {path}')
    candidates = list((ROOT / 'output/s1_oof/NPFEEGNet').glob('*/summary.csv'))
    if len(candidates) != 1:
        raise RuntimeError(f'Expected one S1 summary, got {candidates}')
    summary_path = candidates[0]
    with summary_path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    keys = [(int(r['subject']), int(r['seed'])) for r in rows]
    if len(rows) != 162 or set(keys) != EXPECTED or len(set(keys)) != 162:
        raise RuntimeError('Incomplete S1 54x3 selection grid')
    root = summary_path.parent
    source_labels, source_probs = [], []
    fold_files = []
    for subject, seed in sorted(EXPECTED):
        folds = sorted((root / f'sub{subject}' / f'seed{seed}' / 'selection').glob('fold*_groups*/test_auxiliary.npz'))
        if len(folds) != 2:
            raise RuntimeError(f'Expected two fold-out outputs: {(subject, seed)}')
        for path in folds:
            labels, probabilities = read(path, 100)
            source_labels.append(labels)
            source_probs.append(probabilities)
            fold_files.append(path)
    s1_labels = np.concatenate(source_labels)
    s1_probs = np.concatenate(source_probs)
    if s1_labels.shape != (32400,):
        raise RuntimeError('Incorrect S1 calibration sample size')
    def objective(log_t):
        return metrics(s1_labels, scale(s1_probs, np.exp(log_t)))['nll']
    result = minimize_scalar(objective, bounds=(np.log(0.5), np.log(5.0)),
                             method='bounded', options={'xatol': 1e-8})
    if not result.success:
        raise RuntimeError(result.message)
    temperature = float(np.exp(result.x))
    lock_path = ROOT / 'results/temperature_lock.json'
    if lock_path.exists():
        raise FileExistsError(lock_path)
    lock = {
        'status': 'temperature_fitted_on_S1_before_S2_scoring',
        'temperature': temperature,
        's1_oof_predictions': 32400,
        's1_before': metrics(s1_labels, s1_probs),
        's1_after': metrics(s1_labels, scale(s1_probs, temperature)),
        'design_lock_sha256': sha(design_path),
        's1_summary_sha256': sha(summary_path),
        'fold_file_count': len(fold_files),
        'fold_files_sha256': {str(p): sha(p) for p in fold_files},
        'session2_used_for_fitting': False,
    }
    lock_path.write_text(json.dumps(lock, indent=2) + '\n')
    # Only after persisting the S1-only fitted temperature may S2 be read.
    before_labels, before_probs, after_probs = [], [], []
    run_nll = []
    for subject, seed in sorted(EXPECTED):
        labels, probabilities = read(RD / f'sub{subject}' / f'seed{seed}' / 'test_auxiliary.npz', 200)
        calibrated = scale(probabilities, temperature)
        before_labels.append(labels)
        before_probs.append(probabilities)
        after_probs.append(calibrated)
        run_nll.append((subject, seed,
                        metrics(labels, probabilities)['nll'],
                        metrics(labels, calibrated)['nll']))
    labels = np.concatenate(before_labels)
    before = metrics(labels, np.concatenate(before_probs))
    after = metrics(labels, np.concatenate(after_probs))
    nll_by_subject = np.array([
        np.mean([row[3] - row[2] for row in run_nll if row[0] == s])
        for s in range(1, 55)
    ])
    rng = np.random.default_rng(20260926)
    draws = rng.integers(0, 54, (100000, 54))
    ci = np.quantile(nll_by_subject[draws].mean(1), [0.025, 0.975])
    report = {
        'status': 'complete', 'analysis_role': 'post-confirmatory calibration sensitivity',
        'temperature': temperature, 'temperature_lock_sha256': sha(lock_path),
        'session2_used_for_fitting': False,
        's1_oof': {'before': lock['s1_before'], 'after': lock['s1_after']},
        'session2': {'before': before, 'after': after},
        'subject_paired_nll_after_minus_before': {
            'mean': float(nll_by_subject.mean()),
            'bootstrap_95_ci': [float(v) for v in ci],
            'bootstrap_subjects': 54, 'bootstrap_draws': 100000,
        },
        'note': 'T is fitted on S1 fold-out probabilities from fold models, then applied to full-S1 retrained RD models; model differences and S1-to-S2 shift can limit transfer.',
    }
    (ROOT / 'results/temperature_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
