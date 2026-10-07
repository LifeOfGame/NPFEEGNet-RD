"""Freeze source-only epochs for the post-hoc matched-preprocessing CNN runs."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path('/mnt/helongfei/NPFEEGNet-RD-zhou2020-posthoc-20260927')
MODELS = {'rd': 'NPFEEGNet', 'nodemean': 'NPFEEGNet', 'eegnet_demean': 'EEGNet'}
GROUPS = {'s': list(range(1, 13)), 'a': list(range(13, 21))}

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as fp:
        for block in iter(lambda: fp.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    records = []
    for group, subjects in GROUPS.items():
        expected = {(subject, seed) for subject in subjects for seed in range(3)}
        for name, network in MODELS.items():
            key = f'{group}_{name}'
            config = ROOT / 'config' / f'zhou2020_{key}_selection.yaml'
            summaries = list((ROOT / 'output' / f'zhou2020_{key}_selection' / network).glob('*/summary.csv'))
            if len(summaries) != 1:
                raise RuntimeError(f'{key}: expected one selection summary')
            summary = summaries[0]
            with summary.open(newline='') as fp:
                rows = list(csv.DictReader(fp))
            keys = [(int(r['subject']), int(r['seed'])) for r in rows]
            if len(rows) != len(expected) or set(keys) != expected or len(set(keys)) != len(keys):
                raise RuntimeError(f'{key}: incomplete source selection')
            epochs = {}
            fold_paths = []
            for row in rows:
                subject, seed = int(row['subject']), int(row['seed'])
                if row['selection_strategy'] != 'group_cv' or int(row['cv_num_folds']) != 2:
                    raise RuntimeError(f'{key}: invalid CV method')
                path = summary.parent / f'sub{subject}' / f'seed{seed}' / 'selection/fold_summary.csv'
                with path.open(newline='') as fp:
                    folds = list(csv.DictReader(fp))
                if len(folds) != 2:
                    raise RuntimeError(f'{key}: incomplete fold summaries')
                groups = np.load(ROOT / 'data' / f'Z{subject:02d}T_group.npy')
                seen = set()
                for fold in folds:
                    val = {int(x) for x in fold['validation_groups'].split(';')}
                    if not val or seen & val:
                        raise RuntimeError(f'{key}: overlapping validation runs')
                    seen |= val
                    n_val = int(np.isin(groups, list(val)).sum())
                    if n_val != int(fold['validation_samples']) or len(groups)-n_val != int(fold['train_samples']):
                        raise RuntimeError(f'{key}: fold size mismatch')
                if seen != set(np.unique(groups).tolist()):
                    raise RuntimeError(f'{key}: incomplete run coverage')
                selected = max(1, int(np.rint(np.median([int(f['best_val_epoch']) for f in folds]))))
                if selected != int(row['best_val_epoch']):
                    raise RuntimeError(f'{key}: epoch mismatch')
                epochs.setdefault(str(subject), {})[str(seed)] = selected
                fold_paths.append(path)
            epoch_path = ROOT / 'results' / f'zhou2020_{key}_selected_epochs.json'
            epoch_path.write_text(json.dumps({'source_summary': str(summary), 'selected_epochs': epochs}, indent=2) + '\n')
            final = yaml.safe_load(config.read_text())
            final.update(out_folder=str(ROOT / 'output' / f'zhou2020_{key}_final'),
                         selected_epochs_file=str(epoch_path), fixed_epoch_training=True,
                         two_stage_training=False, selection_only=False, export_aux=True,
                         analysis_status=f'posthoc_zhou2020_{key}_target_final')
            final_path = ROOT / 'config' / f'zhou2020_{key}_final.yaml'
            final_path.write_text(yaml.safe_dump(final, sort_keys=False))
            records.append({'model': key, 'summary': str(summary), 'summary_sha256': digest(summary),
                            'selection_config_sha256': digest(config), 'selected_epochs_sha256': digest(epoch_path),
                            'final_config_sha256': digest(final_path), 'fold_summary_sha256': [digest(p) for p in fold_paths]})
    lock = {'status': 'posthoc_source_selection_frozen_before_matched_target_evaluation',
            'dataset': 'Zhou2020', 'selection_runs': 180, 'records': records}
    (ROOT / 'results' / 'selection_freeze.json').write_text(json.dumps(lock, indent=2) + '\n')
    print('POSTHOC SOURCE SELECTION FROZEN: 180 durations', flush=True)

if __name__ == '__main__':
    main()
