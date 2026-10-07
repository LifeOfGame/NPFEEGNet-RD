"""Audit all A-only CV results and freeze epoch maps before reading Session B."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path('/mnt/helongfei/NPFEEGNet-RD-reviewer-controls-20260925')
PROJECT = Path('/mnt/helongfei/NPFEEGNet-RD-paper-20260917')
DATA = ROOT / 'data/bnci2015_001'
MODELS = {'rd': 'NPFEEGNet', 'nodemean': 'NPFEEGNet', 'eegnet_demean': 'EEGNet'}
EXPECTED = {(subject, seed) for subject in range(1, 13) for seed in (0, 1, 2)}


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


artifacts = []
def record(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    artifacts.append({'path': str(path), 'sha256': digest(path)})


for subject in range(1, 13):
    for suffix in ('data', 'label', 'group'):
        record(DATA / f'N{subject:02d}T_{suffix}.npy')
record(DATA / 'session_a_preprocess_report.json')
record(ROOT / 'preprocess_bnci2015_a.py')
record(ROOT / 'build_bnci2015_selection_configs.py')
record(ROOT / 'check_bnci2015_model_shapes.py')
record(ROOT / 'freeze_bnci2015_selection.py')
record(ROOT / 'preprocess_bnci2015_b_after_lock.py')
record(ROOT / 'run_bnci2015_final_after_lock.sh')
record(ROOT / 'analyze_bnci2015_final.py')
record(ROOT / 'run_bnci2015_pipeline_after_selection.sh')
record(PROJECT / 'model/NPF_EEGNet.py')
record(PROJECT / 'model/openbmi_baselines.py')
record(PROJECT / 'repro/frozen_engines/openbmi_dd010/train_dcsa_2a.py')
record(PROJECT / 'train_dcsa_2a.py')
record(PROJECT / 'train_openbmi_baselines.py')

selection = {}
for name, network in MODELS.items():
    config = ROOT / 'config' / f'bnci2015_001_{name}_selection.yaml'
    record(config)
    candidates = list((ROOT / 'output' / f'bnci2015_001_{name}_selection' / network).glob('*/summary.csv'))
    if len(candidates) != 1:
        raise RuntimeError(f'{name}: expected exactly one selection summary, found {candidates}')
    summary = candidates[0]
    with summary.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    keys = [(int(row['subject']), int(row['seed'])) for row in rows]
    if len(rows) != 36 or len(set(keys)) != 36 or set(keys) != EXPECTED:
        raise RuntimeError(f'{name}: incomplete subject/seed grid')
    epochs = {}
    for row in rows:
        subject, seed = int(row['subject']), int(row['seed'])
        if row['selection_strategy'] != 'group_cv' or int(row['cv_num_folds']) != 5:
            raise RuntimeError(f'{name}: unexpected CV rule: {(subject, seed)}')
        if row['cv_group_partitions'] != '0|1|2|3|4':
            raise RuntimeError(f'{name}: unexpected fold membership')
        fold_path = summary.parent / f'sub{subject}' / f'seed{seed}' / 'selection/fold_summary.csv'
        with fold_path.open(newline='') as stream:
            folds = list(csv.DictReader(stream))
        if len(folds) != 5 or [int(f['fold']) for f in folds] != list(range(5)):
            raise RuntimeError(f'{name}: incomplete folds for {(subject, seed)}')
        if any(int(f['train_samples']) != 160 or int(f['validation_samples']) != 40 for f in folds):
            raise RuntimeError(f'{name}: wrong fold trial counts')
        chosen = max(1, int(np.rint(np.median([int(f['best_val_epoch']) for f in folds]))))
        if chosen != int(row['best_val_epoch']):
            raise RuntimeError(f'{name}: epoch mismatch for {(subject, seed)}')
        epochs.setdefault(str(subject), {})[str(seed)] = chosen
        record(fold_path)
    record(summary)
    target = ROOT / 'results' / f'bnci2015_001_{name}_selected_epochs.json'
    target.write_text(json.dumps({'source_summary': str(summary), 'selected_epochs': epochs}, indent=2) + '\n')
    record(target)
    final_config = yaml.safe_load(config.read_text())
    final_config['out_folder'] = str(ROOT / 'output' / f'bnci2015_001_{name}_final')
    final_config['selected_epochs_file'] = str(target)
    final_config['fixed_epoch_training'] = True
    final_config['two_stage_training'] = False
    final_config['selection_only'] = False
    final_config['export_aux'] = True
    final_config['analysis_status'] = 'bnci2015_001_external_B_final'
    # The baseline wrapper only accepts its historical config paths when a
    # protocol_lock_file key is set. The external runner verifies this lock
    # before calling that wrapper; NPF models also verify it internally.
    if name != 'eegnet_demean':
        final_config['protocol_lock_file'] = str(ROOT / 'results/bnci2015_001_protocol_lock.json')
    final_path = ROOT / 'config' / f'bnci2015_001_{name}_final.yaml'
    final_path.write_text(yaml.safe_dump(final_config, sort_keys=False))
    record(final_path)
    selection[name] = {'n_subject_seed': len(rows), 'selected_epoch_range': [min(v for m in epochs.values() for v in m.values()), max(v for m in epochs.values() for v in m.values())], 'summary': str(summary)}

raw_provenance = json.loads((Path('/mnt/helongfei/database/raw/bnci2015_001_nemar_v1.0.2') / 'sourcedata_provenance.json').read_text())
expected_raw = {f'S{subject:02d}{session}.mat' for subject in range(1, 13) for session in 'AB'}
raw_map = {item['file']: item for item in raw_provenance['files'] if item['file'] in expected_raw}
if set(raw_map) != expected_raw:
    raise RuntimeError('Raw source manifest incomplete')
for subject in range(1, 13):
    path = Path('/mnt/helongfei/database/raw/bnci2015_001_nemar_v1.0.2') / f'S{subject:02d}B.mat'
    if path.stat().st_size != raw_map[path.name]['bytes'] or digest(path) != raw_map[path.name]['sha256']:
        raise RuntimeError(f'B source file checksum mismatch: {path}')

lock = {'status': 'ready_for_session2', 'dataset': 'BNCI2015-001',
        'protocol': '12 subjects, A-only stratified 5-fold median epoch selection, 3 seeds, one B evaluation',
        'source': 'NEMAR nm000140 v1.0.2 unmodified MAT files',
        'selection': selection, 'artifacts': artifacts,
        'session_b_sha256': {name: raw_map[name]['sha256'] for name in sorted(expected_raw) if name.endswith('B.mat')}}
target = ROOT / 'results/bnci2015_001_protocol_lock.json'
target.write_text(json.dumps(lock, indent=2) + '\n')
print(json.dumps({'lock': str(target), 'status': lock['status'], 'selection': selection, 'artifact_count': len(artifacts)}, indent=2))
