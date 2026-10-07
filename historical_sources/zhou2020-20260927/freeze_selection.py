"""Audit all source-only selected epochs and lock artifacts before target access."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path('/mnt/helongfei/NPFEEGNet-RD-zhou2020-20260927')
PROJECT = Path('/mnt/helongfei/NPFEEGNet-RD-paper-20260917')
RAW = Path('/mnt/helongfei/database/raw/zhou2020_zenodo_v1')
MODELS = {'rd': 'NPFEEGNet', 'nodemean': 'NPFEEGNet', 'eegnet_demean': 'EEGNet'}
GROUPS = {'s': list(range(1, 13)), 'a': list(range(13, 21))}

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as fp:
        for block in iter(lambda: fp.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    if (ROOT / 'results' / 'fetch_status.txt').read_text().strip() != 'source_session_1_download_complete':
        raise RuntimeError('All source-session runs must be fetched first')
    artifacts = []
    def record(path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        artifacts.append({'path': str(path), 'sha256': digest(path)})
    for path in (
        ROOT / 'results/protocol_lock.json', ROOT / 'results/source_only_addendum.json',
        ROOT / 'results/source_probe.json', ROOT / 'preprocess_sessions.py',
        ROOT / 'fetch_source_session.py', ROOT / 'fetch_target_session.py',
        ROOT / 'build_selection_configs.py', ROOT / 'freeze_selection.py',
        ROOT / 'analyze_final.py', ROOT / 'run_pipeline.sh',
        ROOT / 'data/session_1_report.json',
        PROJECT / 'model/NPF_EEGNet.py', PROJECT / 'model/openbmi_baselines.py',
        PROJECT / 'repro/frozen_engines/openbmi_dd010/train_dcsa_2a.py',
        PROJECT / 'train_dcsa_2a.py', PROJECT / 'train_openbmi_baselines.py',
    ):
        record(path)
    for subject in range(1, 21):
        record(ROOT / 'results' / f'S{subject:02d}_session_1_source.json')
        for path in sorted((RAW / f'S{subject:02d}' / 'session_1').glob('run_*.npz')):
            record(path)
        for suffix in ('data', 'label', 'group'):
            record(ROOT / 'data' / f'Z{subject:02d}T_{suffix}.npy')
    selection = {}
    for group, subjects in GROUPS.items():
        expected = {(subject, seed) for subject in subjects for seed in (0, 1, 2)}
        for name, network in MODELS.items():
            key = f'{group}_{name}'
            config_path = ROOT / 'config' / f'zhou2020_{key}_selection.yaml'
            record(config_path)
            summaries = list((ROOT / 'output' / f'zhou2020_{key}_selection' / network).glob('*/summary.csv'))
            if len(summaries) != 1:
                raise RuntimeError(f'{key}: expected one summary, got {summaries}')
            summary = summaries[0]
            with summary.open(newline='') as fp:
                rows = list(csv.DictReader(fp))
            keys = [(int(row['subject']), int(row['seed'])) for row in rows]
            if len(rows) != len(expected) or set(keys) != expected or len(set(keys)) != len(keys):
                raise RuntimeError(f'{key}: incomplete subject/seed grid')
            epochs = {}
            for row in rows:
                subject, seed = int(row['subject']), int(row['seed'])
                if row['selection_strategy'] != 'group_cv' or int(row['cv_num_folds']) != 2:
                    raise RuntimeError(f'{key}: invalid selection protocol')
                fold_path = summary.parent / f'sub{subject}' / f'seed{seed}' / 'selection/fold_summary.csv'
                with fold_path.open(newline='') as fp:
                    folds = list(csv.DictReader(fp))
                if len(folds) != 2 or {int(f['fold']) for f in folds} != {0, 1}:
                    raise RuntimeError(f'{key}: incomplete folds, subject {subject}, seed {seed}')
                groups = np.load(ROOT / 'data' / f'Z{subject:02d}T_group.npy')
                seen = set()
                for fold in folds:
                    val_groups = {int(v) for v in fold['validation_groups'].split(';')}
                    if not val_groups or seen & val_groups:
                        raise RuntimeError(f'{key}: invalid run partition')
                    seen |= val_groups
                    n_val = int(np.isin(groups, list(val_groups)).sum())
                    if n_val != int(fold['validation_samples']) or len(groups) - n_val != int(fold['train_samples']):
                        raise RuntimeError(f'{key}: fold sample counts mismatch')
                if seen != set(np.unique(groups).tolist()):
                    raise RuntimeError(f'{key}: incomplete run partition')
                chosen = max(1, int(np.rint(np.median([int(f['best_val_epoch']) for f in folds]))))
                if chosen != int(row['best_val_epoch']):
                    raise RuntimeError(f'{key}: selected epoch mismatch')
                epochs.setdefault(str(subject), {})[str(seed)] = chosen
                record(fold_path)
            record(summary)
            epoch_path = ROOT / 'results' / f'zhou2020_{key}_selected_epochs.json'
            epoch_path.write_text(json.dumps({'source_summary': str(summary), 'selected_epochs': epochs}, indent=2) + '\n')
            record(epoch_path)
            final = yaml.safe_load(config_path.read_text())
            final.update(out_folder=str(ROOT / 'output' / f'zhou2020_{key}_final'),
                         selected_epochs_file=str(epoch_path), fixed_epoch_training=True,
                         two_stage_training=False, selection_only=False, export_aux=True,
                         analysis_status=f'zhou2020_{key}_session2_final')
            if name != 'eegnet_demean':
                final['protocol_lock_file'] = str(ROOT / 'results/final_unlock_lock.json')
            final_path = ROOT / 'config' / f'zhou2020_{key}_final.yaml'
            final_path.write_text(yaml.safe_dump(final, sort_keys=False))
            record(final_path)
            selection[key] = {'epochs': epochs, 'summary': str(summary)}
    lock = {'status': 'ready_for_session2', 'dataset': 'Zhou2020', 'target_session': 'session_2',
            'source_only_selection': selection, 'artifacts': artifacts}
    (ROOT / 'results/final_unlock_lock.json').write_text(json.dumps(lock, indent=2) + '\n')
    print(f'FROZEN {len(artifacts)} artifacts and all 180 selected epochs before target access', flush=True)

if __name__ == '__main__':
    main()
