"""Adapt frozen OpenBMI recipes to BNCI2015-001 A-only 5-fold selection."""
from pathlib import Path
import yaml

PROJECT = Path('/mnt/helongfei/NPFEEGNet-RD-paper-20260917')
AUDIT = Path('/mnt/helongfei/NPFEEGNet-RD-reviewer-controls-20260925')
DATA = AUDIT / 'data/bnci2015_001'
CONFIG = AUDIT / 'config'
SOURCES = {
    'rd': 'openbmi_selection_npf_raw_demean_3seed.yaml',
    'nodemean': 'openbmi_selection_npf_nodemean_3seed.yaml',
    'eegnet_demean': 'openbmi_baseline_selection_eegnet_demean_3seed.yaml',
}

for name, source in SOURCES.items():
    config = yaml.safe_load((PROJECT / 'config' / source).read_text())
    config['network_args']['num_channels'] = 13
    config['data_path'] = str(DATA)
    config['data_prefix'] = 'N'
    config['out_folder'] = str(AUDIT / 'output' / f'bnci2015_001_{name}_selection')
    config['subjects'] = list(range(1, 13))
    config['random_seeds'] = [0, 1, 2]
    config['validation_strategy'] = 'group'
    config['two_stage_training'] = True
    config['selection_only'] = True
    config['epoch_selection_strategy'] = 'group_cv'
    config['cv_num_folds'] = 5
    config['cv_epoch_aggregation'] = 'median'
    config['analysis_status'] = 'bnci2015_001_external_A_only_selection'
    config['nGPU'] = 1 if name == 'nodemean' else 0
    path = CONFIG / f'bnci2015_001_{name}_selection.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(path)
    smoke = dict(config)
    smoke['subjects'] = [1]
    smoke['random_seeds'] = [0]
    smoke['out_folder'] = str(AUDIT / 'output' / f'bnci2015_001_{name}_smoke')
    smoke_path = CONFIG / f'bnci2015_001_{name}_smoke.yaml'
    smoke_path.write_text(yaml.safe_dump(smoke, sort_keys=False))
    print(smoke_path)
