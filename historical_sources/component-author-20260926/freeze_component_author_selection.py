"""Validate all Session-1 CV rows and freeze epochs before any Session-2 read."""

import csv
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path('/mnt/helongfei/NPFEEGNet-RD-component-author-20260926')
SOURCE = Path('/mnt/helongfei/NPFEEGNet-RD-paper-20260917')
FBC = Path('/mnt/helongfei/NPFEEGNet-RD-openbmi-fbcnet-audit-20260922')
EXPECTED = {(s, z) for s in range(1, 55) for z in range(3)}
VARIANTS = ('spectral_only', 'no_spectral_prior', 'fbcnet_author_training')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    design = ROOT / 'results/design_lock.json'
    lock = json.loads(design.read_text())
    for path, checksum in lock['sha256'].items():
        if sha(path) != checksum:
            raise RuntimeError(f'Selection design changed: {path}')
    all_summaries = {}
    for name in VARIANTS:
        selection = ROOT / 'config' / f'{name}_selection.yaml'
        config = yaml.safe_load(selection.read_text())
        summaries = list((Path(config['out_folder']) / config['network']).glob('*/summary.csv'))
        if len(summaries) != 1:
            raise RuntimeError(f'{name}: expected exactly one summary, found {summaries}')
        summary = summaries[0]
        with summary.open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        keys = [(int(r['subject']), int(r['seed'])) for r in rows]
        if len(rows) != 162 or set(keys) != EXPECTED or len(set(keys)) != 162:
            raise RuntimeError(f'{name}: incomplete S1 grid')
        if any(r['selection_strategy'] != 'group_cv' or int(r['cv_num_folds']) != 2
               for r in rows):
            raise RuntimeError(f'{name}: unexpected CV strategy')
        if any(k.startswith('test_') for k in rows[0]):
            raise RuntimeError(f'{name}: test metric in selection summary')
        epochs = {(int(r['subject']), int(r['seed'])): int(r['best_val_epoch']) for r in rows}
        if not all(1 <= v <= int(config['epochs']) for v in epochs.values()):
            raise RuntimeError(f'{name}: out-of-range epoch')
        epoch_path = ROOT / 'results' / f'{name}_selected_epochs.json'
        if epoch_path.exists():
            raise FileExistsError(epoch_path)
        epoch_path.write_text(json.dumps({
            'protocol': 'OpenBMI Session-1 two-phase group-CV median epoch',
            'variant': name,
            'official_evaluation_used_for_selection': False,
            'selection_summary': str(summary),
            'selection_summary_sha256': sha(summary),
            'selected_epochs': {str(s): {str(z): epochs[s, z] for z in range(3)}
                                for s in range(1, 55)},
        }, indent=2) + '\n')
        final = dict(config)
        for field in ('validation_strategy', 'epoch_selection_strategy', 'cv_num_folds',
                      'cv_epoch_aggregation', 'export_selection_aux'):
            final.pop(field, None)
        final.update(
            out_folder=str(ROOT / 'output' / name / 'final'),
            selected_epochs_file=str(epoch_path),
            protocol_lock_file=str(ROOT / 'results/session2_lock.json'),
            fixed_epoch_training=True, two_stage_training=False,
            selection_only=False, early_stopping_patience=None, export_aux=True,
        )
        final_path = ROOT / 'config' / f'{name}_final.yaml'
        final_path.write_text(yaml.safe_dump(final, sort_keys=False))
        all_summaries[name] = {'summary': str(summary), 'summary_sha256': sha(summary),
                               'epoch_map': str(epoch_path), 'epoch_map_sha256': sha(epoch_path),
                               'final_config': str(final_path), 'final_config_sha256': sha(final_path)}
    lock_path = ROOT / 'results/session2_lock.json'
    if lock_path.exists():
        raise FileExistsError(lock_path)
    payload = {
        'status': 'ready_for_session2',
        'analysis_role': 'post-confirmatory',
        'official_evaluation_used_for_selection': False,
        'expected_runs_per_variant': 162,
        'selection': all_summaries,
        'sha256': {str(path): sha(path) for path in (
            design, ROOT / 'component_variants.py', ROOT / 'run_component_variant.py',
            ROOT / 'run_fbcnet_author_training.py',
            FBC / 'author_filter_fbcnet.py',
            *[ROOT / 'config' / f'{name}_final.yaml' for name in VARIANTS],
            *[ROOT / 'results' / f'{name}_selected_epochs.json' for name in VARIANTS],
        )},
    }
    lock_path.write_text(json.dumps(payload, indent=2) + '\n')
    print(f'Frozen {len(VARIANTS)} x 162 S1 epochs: {lock_path}')


if __name__ == '__main__':
    main()
