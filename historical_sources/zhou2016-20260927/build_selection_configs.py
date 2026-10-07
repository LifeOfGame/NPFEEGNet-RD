"""Adapt the frozen OpenBMI recipe only for Zhou2016 dimensions and paths."""
from pathlib import Path
import yaml

PROJECT=Path('/mnt/helongfei/NPFEEGNet-RD-paper-20260917')
ROOT=Path('/mnt/helongfei/NPFEEGNet-RD-zhou2016-20260927')
SOURCES={'rd':'openbmi_selection_npf_raw_demean_3seed.yaml',
         'nodemean':'openbmi_selection_npf_nodemean_3seed.yaml',
         'eegnet_demean':'openbmi_baseline_selection_eegnet_demean_3seed.yaml'}
for name,source in SOURCES.items():
    config=yaml.safe_load((PROJECT/'config'/source).read_text())
    config['network_args']['num_channels']=14
    config['network_args']['classes']=3
    config['num_classes']=3
    config['data_path']=str(ROOT/'data')
    config['data_prefix']='Z'
    config['out_folder']=str(ROOT/'output'/f'zhou2016_{name}_selection')
    config['subjects']=[1,2,3,4]
    config['random_seeds']=[0,1,2]
    config['validation_strategy']='group'
    config['two_stage_training']=True
    config['selection_only']=True
    config['epoch_selection_strategy']='group_cv'
    config['cv_num_folds']=2
    config['cv_epoch_aggregation']='median'
    config['nGPU']=0 if name=='rd' else 1
    config['analysis_status']='zhou2016_session0_only_selection'
    path=ROOT/'config'/f'zhou2016_{name}_selection.yaml'
    path.write_text(yaml.safe_dump(config,sort_keys=False))
    print(path)
    smoke=dict(config)
    smoke['subjects']=[1];smoke['random_seeds']=[0]
    smoke['out_folder']=str(ROOT/'output'/f'zhou2016_{name}_smoke')
    (ROOT/'config'/f'zhou2016_{name}_smoke.yaml').write_text(yaml.safe_dump(smoke,sort_keys=False))
