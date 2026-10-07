"""Freeze post-confirmatory selection configurations before running S1 CV."""

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path('/mnt/helongfei/NPFEEGNet-RD-component-author-20260926')
SOURCE = Path('/mnt/helongfei/NPFEEGNet-RD-paper-20260917')
AUDIT_FBC = Path('/mnt/helongfei/NPFEEGNet-RD-openbmi-fbcnet-audit-20260922')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    (ROOT / 'config').mkdir(parents=True, exist_ok=True)
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'logs').mkdir(exist_ok=True)
    base = yaml.safe_load((SOURCE / 'config/openbmi_selection_npf_raw_demean_3seed.yaml').read_text())
    base['data_path'] = '/mnt/helongfei/database/processed/openbmi'
    # CUDA_VISIBLE_DEVICES exposes one device; the engine addresses it as cuda:0.
    base['nGPU'] = 0
    base['analysis_status'] = 'post-confirmatory independent component ablation'
    for name in ('spectral_only', 'no_spectral_prior'):
        config = yaml.safe_load(yaml.safe_dump(base))
        config['out_folder'] = str(ROOT / 'output' / name / 'selection_dualgpu_v3')
        config['publication_model_name'] = name
        if name == 'spectral_only':
            config['raw_branch_loss_weight'] = 0.0
            config['spectral_branch_loss_weight'] = 0.0
        (ROOT / 'config' / f'{name}_selection.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        smoke = yaml.safe_load(yaml.safe_dump(config))
        smoke['subjects'] = [1]
        smoke['random_seeds'] = [0]
        smoke['out_folder'] = str(ROOT / 'output' / name / 'smoke')
        (ROOT / 'config' / f'{name}_smoke.yaml').write_text(yaml.safe_dump(smoke, sort_keys=False))

    fbc = yaml.safe_load((AUDIT_FBC / 'config/fbcnet_selection.yaml').read_text())
    fbc['out_folder'] = str(ROOT / 'output/fbcnet_author_training/selection')
    fbc['analysis_status'] = 'post-confirmatory author-training sensitivity'
    fbc['publication_model_name'] = 'FBCNet-author-filter-author-training'
    # Author repository classify/ho.py: batch 16, Adam 1e-3, max 1500,
    # NoDecrease 200, no LR schedule. Preserve S1 phase-group validation.
    fbc['batch_size'] = 16
    fbc['train_microbatch_size'] = 16
    fbc['epochs'] = 1500
    fbc['early_stopping_patience'] = 200
    fbc['use_lr_scheduler'] = False
    fbc['weight_decay'] = 0.0
    fbc['use_data_augmentation'] = False
    (ROOT / 'config/fbcnet_author_training_selection.yaml').write_text(
        yaml.safe_dump(fbc, sort_keys=False))
    smoke = yaml.safe_load(yaml.safe_dump(fbc))
    smoke['subjects'] = [1]
    smoke['random_seeds'] = [0]
    smoke['out_folder'] = str(ROOT / 'output/fbcnet_author_training/smoke')
    (ROOT / 'config/fbcnet_author_training_smoke.yaml').write_text(
        yaml.safe_dump(smoke, sort_keys=False))
    manifest = {
        'status': 'selection_config_frozen_before_new_S2_evaluation',
        'analysis_role': 'post-confirmatory; earlier OpenBMI S2 results were seen',
        'subjects': list(range(1, 55)), 'seeds': [0, 1, 2],
        'selection': 'Session-1 offline/online phase group CV, two folds, median epoch',
        'final': 'retrain full Session-1 for selected epoch, then one Session-2 evaluation',
        'fbcnet_note': 'Author-recommended training hyperparameters with strict local S1/S2 protocol; not framework-exact author reproduction',
        'sha256': {str(p): sha(p) for p in (ROOT / 'config').glob('*.yaml')},
    }
    (ROOT / 'results/design_lock.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
