"""Matched post-hoc CNN configurations; same network recipes as locked audit."""
from pathlib import Path
import yaml

PROJECT = Path('/mnt/helongfei/NPFEEGNet-RD-paper-20260917')
ROOT = Path('/mnt/helongfei/NPFEEGNet-RD-zhou2020-posthoc-20260927')
SOURCES = {'rd': 'openbmi_selection_npf_raw_demean_3seed.yaml',
           'nodemean': 'openbmi_selection_npf_nodemean_3seed.yaml',
           'eegnet_demean': 'openbmi_baseline_selection_eegnet_demean_3seed.yaml'}
GROUPS = {'s': (list(range(1, 13)), 41), 'a': (list(range(13, 21)), 26)}

def main():
    (ROOT / 'config').mkdir(exist_ok=True)
    for group, (subjects, channels) in GROUPS.items():
        for name, source in SOURCES.items():
            config = yaml.safe_load((PROJECT / 'config' / source).read_text())
            config['network_args']['num_channels'] = channels
            config['network_args']['classes'] = 4
            config['num_classes'] = 4
            config['data_path'] = str(ROOT / 'data')
            config['data_prefix'] = 'Z'
            config['out_folder'] = str(ROOT / 'output' / f'zhou2020_{group}_{name}_selection')
            config['subjects'] = subjects
            config['random_seeds'] = [0, 1, 2]
            config['validation_strategy'] = 'group'
            config['two_stage_training'] = True
            config['selection_only'] = True
            config['epoch_selection_strategy'] = 'group_cv'
            config['cv_num_folds'] = 2
            config['cv_epoch_aggregation'] = 'median'
            config['nGPU'] = 0 if name != 'nodemean' else 1
            config['analysis_status'] = f'posthoc_zhou2020_{group}_source_selection'
            path = ROOT / 'config' / f'zhou2020_{group}_{name}_selection.yaml'
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            smoke = dict(config)
            smoke['subjects'] = [subjects[0]]
            smoke['random_seeds'] = [0]
            smoke['out_folder'] = str(ROOT / 'output' / f'zhou2020_{group}_{name}_smoke')
            (ROOT / 'config' / f'zhou2020_{group}_{name}_smoke.yaml').write_text(yaml.safe_dump(smoke, sort_keys=False))

if __name__ == '__main__':
    main()
