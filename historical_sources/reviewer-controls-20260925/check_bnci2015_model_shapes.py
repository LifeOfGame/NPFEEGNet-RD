"""CPU-only shape check for the three frozen BNCI2015-001 model configurations."""
import sys
from pathlib import Path

import torch
import yaml

PROJECT = Path('/mnt/helongfei/NPFEEGNet-RD-paper-20260917')
AUDIT = Path('/mnt/helongfei/NPFEEGNet-RD-reviewer-controls-20260925')
sys.path.insert(0, str(PROJECT))
from model.NPF_EEGNet import NPFEEGNet  # noqa: E402
from model.openbmi_baselines import OpenBMIEEGNet  # noqa: E402

for name in ['rd', 'nodemean', 'eegnet_demean']:
    config = yaml.safe_load((AUDIT / 'config' / f'bnci2015_001_{name}_selection.yaml').read_text())
    model = (OpenBMIEEGNet if config['network'] == 'EEGNet' else NPFEEGNet)(**config['network_args'])
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(2, 1, 13, 1000))
    print(name, 'parameters', sum(p.numel() for p in model.parameters() if p.requires_grad),
          'output', [tuple(x.shape) for x in output] if isinstance(output, tuple) else tuple(output.shape))
