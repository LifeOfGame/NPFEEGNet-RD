"""Freeze S1-only RD temperature-scaling inputs before fitting or S2 scoring."""

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path('/mnt/helongfei/NPFEEGNet-RD-temperature-20260926')
SOURCE = Path('/mnt/helongfei/NPFEEGNet-RD-paper-20260917')
ENGINE = SOURCE / 'repro/frozen_engines/openbmi_dd010/train_dcsa_2a.py'
BASE = SOURCE / 'config/openbmi_selection_npf_raw_demean_3seed.yaml'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    for folder in ('config', 'results', 'logs'):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    if sha(ENGINE) != 'dd010b3c64a1ec25e10b1fe539e439a0e1f3305942327e4baea49ef7e6485e15':
        raise RuntimeError('Historical OpenBMI engine changed')
    base = yaml.safe_load(BASE.read_text())
    if base['network'] != 'NPFEEGNet' or base['network_args']['input_standardization_scope'] != 'raw':
        raise RuntimeError('Not the original RD selection configuration')
    base['data_path'] = '/mnt/helongfei/database/processed/openbmi'
    base['out_folder'] = str(ROOT / 'output/s1_oof')
    base['nGPU'] = 0
    base['export_selection_aux'] = True
    base['analysis_status'] = 'post-confirmatory S1-only global temperature calibration'
    full_path = ROOT / 'config/rd_s1_oof.yaml'
    full_path.write_text(yaml.safe_dump(base, sort_keys=False))
    smoke = yaml.safe_load(yaml.safe_dump(base))
    smoke['subjects'] = [1]
    smoke['random_seeds'] = [0]
    smoke['out_folder'] = str(ROOT / 'output/smoke')
    smoke_path = ROOT / 'config/rd_s1_oof_smoke.yaml'
    smoke_path.write_text(yaml.safe_dump(smoke, sort_keys=False))
    design = {
        'status': 'fixed_before_temperature_fit',
        'analysis_role': 'post-confirmatory; historical S2 results already known',
        'calibrator': 'one global positive T; minimize pooled S1 fold-out NLL; scipy bounded scalar log-temperature',
        'temperature_range': [0.5, 5.0],
        'application': 'softmax(log(S2 probabilities)/T), without refitting RD',
        'ece': 'pooled 32400 predictions, ten fixed equal-width confidence bins',
        'subjects': list(range(1, 55)), 'seeds': [0, 1, 2],
        'sha256': {str(p): sha(p) for p in (
            ENGINE, BASE, full_path, smoke_path,
            ROOT / 'fit_apply_rd_temperature.py',
        )},
    }
    (ROOT / 'results/design_lock.json').write_text(json.dumps(design, indent=2) + '\n')
    print(json.dumps(design, indent=2))


if __name__ == '__main__':
    main()
