"""Read-only Zhou2020 accuracy audit; no model retraining or result replacement."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path('/mnt/helongfei/NPFEEGNet-RD-zhou2020-20260927')
report = json.loads((ROOT / 'results/final_report.json').read_text())
source = json.loads((ROOT / 'data/session_1_report.json').read_text())
target = json.loads((ROOT / 'data/session_2_report.json').read_text())
result = {'models': {}, 'subject_accuracy': {}, 'target_class_counts': {}, 'source_q99_uv': {}}

for name in ('rd', 'nodemean', 'eegnet_demean'):
    rows = []
    for group in ('s', 'a'):
        network = 'EEGNet' if name == 'eegnet_demean' else 'NPFEEGNet'
        summaries = list((ROOT / 'output' / f'zhou2020_{group}_{name}_selection' / network).glob('*/summary.csv'))
        if len(summaries) != 1:
            raise RuntimeError(f'{group}/{name}: expected one source CV summary')
        with summaries[0].open(newline='') as fp:
            rows.extend(csv.DictReader(fp))
    cv = np.asarray([float(row['best_val_accuracy']) for row in rows])
    epochs = np.asarray([int(row['best_val_epoch']) for row in rows])
    result['models'][name] = {
        'selection_rows': len(rows),
        'source_session_run_group_cv_accuracy_mean_percent': float(cv.mean() * 100),
        'source_session_run_group_cv_accuracy_median_percent': float(np.median(cv) * 100),
        'source_session_run_group_cv_accuracy_range_percent': [float(cv.min() * 100), float(cv.max() * 100)],
        'selected_epoch_median': float(np.median(epochs)),
        'selected_epoch_range': [int(epochs.min()), int(epochs.max())],
        'selected_epoch_one_count': int((epochs == 1).sum()),
        'target_session_accuracy_percent': report['model_subject_macro'][name]['accuracy_percent'],
    }

with (ROOT / 'results/final_subjects.csv').open(newline='') as fp:
    subject_rows = list(csv.DictReader(fp))
for row in subject_rows:
    result['subject_accuracy'][row['subject']] = {key: float(row[key]) for key in (
        'rd_accuracy_percent', 'nodemean_accuracy_percent', 'eegnet_demean_accuracy_percent')}
for item in target['subjects']:
    result['target_class_counts'][str(item['subject'])] = item['class_counts']
for item in source['subjects']:
    result['source_q99_uv'][str(item['subject'])] = float(np.mean([r['raw_q99_abs_uv'] for r in item['runs']]))

output = ROOT / 'results/accuracy_diagnostic.json'
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result['models'], indent=2))
print('source_q99_uv_median', np.median(list(result['source_q99_uv'].values())))
print('target_class_counts_total', np.sum(list(result['target_class_counts'].values()), axis=0).tolist())
