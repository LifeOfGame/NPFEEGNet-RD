"""Audit frozen Zhou2020 four-class results with subjects as the inferential unit."""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

ROOT = Path('/mnt/helongfei/NPFEEGNet-RD-zhou2020-20260927')
MODELS = {'rd': 'NPFEEGNet', 'nodemean': 'NPFEEGNet', 'eegnet_demean': 'EEGNet'}
GROUPS = {'s': list(range(1, 13)), 'a': list(range(13, 21))}
subjects = {}
paths = {}
confusions = {}

def contrast(a, b, seed):
    delta = 100 * (subjects[a][:, 0] - subjects[b][:, 0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(100000, len(delta)))
    ci = np.quantile(delta[indices].mean(axis=1), [.025, .975]).tolist()
    observed = abs(delta.mean())
    count = 0
    for start in range(0, 1 << len(delta), 65536):
        nums = np.arange(start, min(start + 65536, 1 << len(delta)), dtype=np.uint32)
        signs = 2 * ((nums[:, None] >> np.arange(len(delta), dtype=np.uint32)) & 1).astype(np.int8) - 1
        count += int(np.count_nonzero(np.abs(signs @ delta / len(delta)) >= observed - 1e-12))
    return {'mean_pp': float(delta.mean()), 'subject_bootstrap_95_ci_pp': ci,
            'exact_two_sided_sign_flip_p': count / float(1 << len(delta)),
            'wins_ties_losses': [int((delta > 1e-12).sum()), int((np.abs(delta) <= 1e-12).sum()),
                                 int((delta < -1e-12).sum())],
            'by_hardware_group_mean_pp': {'s_41_channels': float(delta[:12].mean()),
                                          'a_26_channels': float(delta[12:].mean())}}

def main():
    if not (ROOT / 'results' / 'target_fetch_status.txt').is_file():
        raise RuntimeError('Target fetch incomplete')
    lock = json.loads((ROOT / 'results' / 'final_unlock_lock.json').read_text())
    if lock['status'] != 'ready_for_session2':
        raise RuntimeError('Invalid target release gate')
    for name, network in MODELS.items():
        subject_runs = {subject: [] for subject in range(1, 21)}
        confusion = np.zeros((4, 4), dtype=np.int64)
        paths[name] = {}
        for group, group_subjects in GROUPS.items():
            key = f'{group}_{name}'
            summaries = list((ROOT / 'output' / f'zhou2020_{key}_final' / network).glob('*/summary.csv'))
            if len(summaries) != 1:
                raise RuntimeError(f'{key}: expected one final summary')
            summary = summaries[0]
            paths[name][group] = str(summary)
            with summary.open(newline='') as fp:
                rows = list(csv.DictReader(fp))
            expected = {(subject, seed) for subject in group_subjects for seed in range(3)}
            keys = [(int(row['subject']), int(row['seed'])) for row in rows]
            if len(rows) != len(expected) or set(keys) != expected or len(set(keys)) != len(keys):
                raise RuntimeError(f'{key}: incomplete final grid')
            epochs = json.loads((ROOT / 'results' / f'zhou2020_{key}_selected_epochs.json').read_text())['selected_epochs']
            for row in rows:
                subject, seed = int(row['subject']), int(row['seed'])
                if int(row['retrain_epochs']) != epochs[str(subject)][str(seed)]:
                    raise RuntimeError(f'{key}: retraining duration differs from source-only selection')
                with np.load(summary.parent / f'sub{subject}' / f'seed{seed}' / 'test_auxiliary.npz') as saved:
                    y = saved['labels']
                    pred = saved['predictions']
                source_y = np.load(ROOT / 'data' / f'Z{subject:02d}E_label.npy') - 1
                if not np.array_equal(y, source_y):
                    raise RuntimeError(f'{key}: target labels mismatch')
                accuracy = float(np.mean(y == pred))
                if abs(accuracy - float(row['test_accuracy'])) > 1e-8:
                    raise RuntimeError(f'{key}: accuracy does not match saved predictions')
                subject_runs[subject].append((accuracy, float(balanced_accuracy_score(y, pred))))
                confusion += confusion_matrix(y, pred, labels=list(range(4)))
        if any(len(subject_runs[s]) != 3 for s in subject_runs):
            raise RuntimeError(f'{name}: missing seeds')
        subjects[name] = np.asarray([np.mean(subject_runs[s], axis=0) for s in range(1, 21)])
        confusions[name] = confusion.tolist()
    report = {
        'status': 'complete', 'dataset': 'Zhou2020',
        'role': 'later independently locked four-class cross-day audit',
        'subjects': 20, 'seeds': 3, 'source_session': 1, 'target_session': 2,
        'classes': ['left_hand', 'right_hand', 'feet', 'rest'],
        'hardware_groups': {'s': 'subjects 1-12, 41 EEG channels', 'a': 'subjects 13-20, 26 EEG channels'},
        'model_subject_macro': {
            name: {'accuracy_percent': float(100 * values[:, 0].mean()),
                   'balanced_accuracy_percent': float(100 * values[:, 1].mean()),
                   's_group_accuracy_percent': float(100 * values[:12, 0].mean()),
                   'a_group_accuracy_percent': float(100 * values[12:, 0].mean())}
            for name, values in subjects.items()},
        'primary_rd_minus_nodemean': contrast('rd', 'nodemean', 20260927),
        'secondary_rd_minus_eegnet_demean': contrast('rd', 'eegnet_demean', 20260928),
        'pooled_confusion_counts_descriptive': confusions,
        'final_summary_csvs': paths,
        'target_release_lock': str(ROOT / 'results/final_unlock_lock.json'),
    }
    with (ROOT / 'results' / 'final_subjects.csv').open('w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['subject', 'hardware_group', 'rd_accuracy_percent', 'nodemean_accuracy_percent',
                         'eegnet_demean_accuracy_percent', 'rd_minus_nodemean_pp'])
        for i in range(20):
            writer.writerow([i + 1, 's_41_channels' if i < 12 else 'a_26_channels',
                             100 * subjects['rd'][i, 0], 100 * subjects['nodemean'][i, 0],
                             100 * subjects['eegnet_demean'][i, 0],
                             100 * (subjects['rd'][i, 0] - subjects['nodemean'][i, 0])])
    (ROOT / 'results' / 'final_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__':
    main()
