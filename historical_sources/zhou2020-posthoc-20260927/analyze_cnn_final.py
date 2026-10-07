"""Audit post-hoc four-class CNN runs under matched 8-30 Hz preprocessing."""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

ROOT = Path('/mnt/helongfei/NPFEEGNet-RD-zhou2020-posthoc-20260927')
MODELS = {'rd': 'NPFEEGNet', 'nodemean': 'NPFEEGNet', 'eegnet_demean': 'EEGNet'}
GROUPS = {'s': list(range(1, 13)), 'a': list(range(13, 21))}

def exact_signflip(delta):
    observed = abs(delta.mean())
    count = 0
    for start in range(0, 1 << len(delta), 65536):
        nums = np.arange(start, min(start + 65536, 1 << len(delta)), dtype=np.uint32)
        signs = 2 * ((nums[:, None] >> np.arange(len(delta), dtype=np.uint32)) & 1).astype(np.int8) - 1
        count += int(np.count_nonzero(np.abs(signs @ delta / len(delta)) >= observed - 1e-12))
    return count / float(1 << len(delta))

def contrast(a, b, seed):
    delta = 100 * (values[a][:, 0] - values[b][:, 0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(100000, len(delta)))
    return {'mean_pp': float(delta.mean()),
            'subject_bootstrap_95_ci_pp': np.quantile(delta[idx].mean(1), [.025, .975]).tolist(),
            'exact_two_sided_sign_flip_p': exact_signflip(delta),
            'wins_ties_losses': [int((delta > 1e-12).sum()), int((np.abs(delta) <= 1e-12).sum()),
                                 int((delta < -1e-12).sum())]}

values = {}
def main():
    freeze = json.loads((ROOT / 'results/selection_freeze.json').read_text())
    if freeze['status'] != 'posthoc_source_selection_frozen_before_matched_target_evaluation':
        raise RuntimeError('Source selection freeze absent')
    summaries = {}
    for name, network in MODELS.items():
        subject_runs = {i: [] for i in range(1, 21)}
        summaries[name] = {}
        for group, group_subjects in GROUPS.items():
            key = f'{group}_{name}'
            paths = list((ROOT / 'output' / f'zhou2020_{key}_final' / network).glob('*/summary.csv'))
            if len(paths) != 1:
                raise RuntimeError(f'{key}: missing final summary')
            summary = paths[0]
            summaries[name][group] = str(summary)
            with summary.open(newline='') as fp:
                rows = list(csv.DictReader(fp))
            expected = {(s, seed) for s in group_subjects for seed in range(3)}
            keys = [(int(r['subject']), int(r['seed'])) for r in rows]
            if len(rows) != len(expected) or set(keys) != expected or len(set(keys)) != len(keys):
                raise RuntimeError(f'{key}: incomplete final grid')
            epochs = json.loads((ROOT / 'results' / f'zhou2020_{key}_selected_epochs.json').read_text())['selected_epochs']
            for row in rows:
                subject, seed = int(row['subject']), int(row['seed'])
                if int(row['retrain_epochs']) != epochs[str(subject)][str(seed)]:
                    raise RuntimeError(f'{key}: retrain duration mismatch')
                with np.load(summary.parent / f'sub{subject}' / f'seed{seed}' / 'test_auxiliary.npz') as saved:
                    y = saved['labels']
                    pred = saved['predictions']
                source_y = np.load(ROOT / 'data' / f'Z{subject:02d}E_label.npy') - 1
                if not np.array_equal(y, source_y):
                    raise RuntimeError(f'{key}: label mismatch')
                accuracy = float(np.mean(y == pred))
                if abs(accuracy - float(row['test_accuracy'])) > 1e-8:
                    raise RuntimeError(f'{key}: saved prediction accuracy mismatch')
                subject_runs[subject].append((accuracy, float(balanced_accuracy_score(y, pred))))
        values[name] = np.asarray([np.mean(subject_runs[i], axis=0) for i in range(1, 21)])
    report = {'status': 'complete', 'role': 'post-hoc exploratory matched-preprocessing four-class control',
              'subjects': 20, 'seeds': 3, 'source_session': 1, 'target_session': 2,
              'preprocessing': 'continuous zero-phase IIR 8-30Hz, class [0.5,4.5)s',
              'model_subject_macro': {name: {'accuracy_percent': float(100 * x[:, 0].mean()),
                                             'balanced_accuracy_percent': float(100 * x[:, 1].mean())}
                                      for name, x in values.items()},
              'rd_minus_nodemean': contrast('rd', 'nodemean', 20260927),
              'rd_minus_eegnet_demean': contrast('rd', 'eegnet_demean', 20260928),
              'final_summary_csvs': summaries}
    with (ROOT / 'results/final_subjects.csv').open('w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['subject', 'rd_accuracy_percent', 'nodemean_accuracy_percent',
                         'eegnet_demean_accuracy_percent', 'rd_minus_nodemean_pp'])
        for i in range(20):
            writer.writerow([i + 1, 100 * values['rd'][i, 0], 100 * values['nodemean'][i, 0],
                             100 * values['eegnet_demean'][i, 0],
                             100 * (values['rd'][i, 0] - values['nodemean'][i, 0])])
    (ROOT / 'results/final_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__':
    main()
