"""Verify and summarize the one-time BNCI2015-001 A->B external evaluation."""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

ROOT = Path('/mnt/helongfei/NPFEEGNet-RD-reviewer-controls-20260925')
DATA = ROOT / 'data/bnci2015_001'
MODELS = {'rd': 'NPFEEGNet', 'nodemean': 'NPFEEGNet', 'eegnet_demean': 'EEGNet'}
EXPECTED = {(subject, seed) for subject in range(1, 13) for seed in (0, 1, 2)}
per_subject = {}
paths = {}

for name, network in MODELS.items():
    candidates = list((ROOT / 'output' / f'bnci2015_001_{name}_final' / network).glob('*/summary.csv'))
    if len(candidates) != 1:
        raise RuntimeError(f'{name}: expected one final summary, got {candidates}')
    summary = candidates[0]
    with summary.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    keys = [(int(row['subject']), int(row['seed'])) for row in rows]
    if len(rows) != 36 or len(set(keys)) != 36 or set(keys) != EXPECTED:
        raise RuntimeError(f'{name}: incomplete final grid')
    mapping = json.loads((ROOT / 'results' / f'bnci2015_001_{name}_selected_epochs.json').read_text())['selected_epochs']
    subject_runs = {subject: [] for subject in range(1, 13)}
    for row in rows:
        subject, seed = int(row['subject']), int(row['seed'])
        if int(row['retrain_epochs']) != int(mapping[str(subject)][str(seed)]):
            raise RuntimeError(f'{name}: epoch map mismatch: {(subject, seed)}')
        auxiliary = summary.parent / f'sub{subject}' / f'seed{seed}' / 'test_auxiliary.npz'
        with np.load(auxiliary) as saved:
            labels = saved['labels']
            predictions = saved['predictions']
        expected_labels = np.load(DATA / f'N{subject:02d}E_label.npy').ravel() - 1
        if len(labels) != 200 or not np.array_equal(labels, expected_labels):
            raise RuntimeError(f'{name}: B labels do not match processed source: {(subject, seed)}')
        accuracy = float((predictions == labels).mean())
        if abs(accuracy - float(row['test_accuracy'])) > 1e-8:
            raise RuntimeError(f'{name}: summary versus prediction accuracy mismatch')
        subject_runs[subject].append((accuracy, float(balanced_accuracy_score(labels, predictions))))
    per_subject[name] = np.array([np.mean(subject_runs[s], axis=0) for s in range(1, 13)])
    paths[name] = str(summary)

rng = np.random.default_rng(20260926)
indices = rng.integers(0, 12, size=(100000, 12))


def compare(a, b):
    delta = 100 * (per_subject[a][:, 0] - per_subject[b][:, 0])
    return {'difference_pp': float(delta.mean()),
            'subject_bootstrap_95_ci_pp': np.quantile(delta[indices].mean(axis=1), [0.025, 0.975]).tolist(),
            'wins_ties_losses': [int((delta > 0).sum()), int((delta == 0).sum()), int((delta < 0).sum())]}


report = {
    'status': 'post-confirmatory external A-to-B result; all 12 subjects and 3 seeds',
    'dataset': 'BNCI2015-001', 'models': {},
    'contrasts': {'RD_minus_NoDemean': compare('rd', 'nodemean'),
                  'RD_minus_EEGNet_Demean': compare('rd', 'eegnet_demean')},
    'summary_paths': paths,
    'interpretation_limit': 'New dataset is only 12 subjects, binary right-hand versus feet imagery; the acquisition already has 0.5-Hz high-pass preprocessing.'
}
for name, array in per_subject.items():
    report['models'][name] = {'subject_macro_accuracy_pct': float(100 * array[:, 0].mean()),
                              'subject_macro_balanced_accuracy_pct': float(100 * array[:, 1].mean())}
(ROOT / 'results/bnci2015_001_external_report.json').write_text(json.dumps(report, indent=2) + '\n')
with (ROOT / 'results/bnci2015_001_external_subjects.csv').open('w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['subject'] + [f'{name}_{metric}' for name in MODELS for metric in ('accuracy', 'balanced_accuracy')])
    for subject in range(1, 13):
        writer.writerow([subject] + [float(value) for name in MODELS for value in per_subject[name][subject - 1]])
print(json.dumps(report, indent=2))
