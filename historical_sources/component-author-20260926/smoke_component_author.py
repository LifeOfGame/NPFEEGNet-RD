"""One minibatch forward/backward sanity check for all three new models."""

import sys
from pathlib import Path

import torch

source = Path('/mnt/helongfei/NPFEEGNet-RD-paper-20260917')
sys.path.insert(0, str(source))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, '/mnt/helongfei/NPFEEGNet-RD-openbmi-fbcnet-audit-20260922')

from component_variants import SpectralOnly, NoSpectralPrior
from author_filter_fbcnet import AuthorFilterFBCNet

common = dict(num_channels=20, classes=2, n_times=1000,
              adaptive_band_routing=False, learnable_band_edges=False,
              input_standardization='demean', input_standardization_scope='raw')
models = [
    ('spectral_only', SpectralOnly(**common)),
    ('no_spectral_prior', NoSpectralPrior(**common)),
    ('fbcnet_author_training', AuthorFilterFBCNet(
        num_channels=20, classes=2, n_times=1000, sampling_rate=250,
        bands=[[low, low+4] for low in range(4, 40, 4)],
        spatial_filters=32, temporal_segments=4,
        input_standardization='demean')),
]
x = torch.randn(3, 1, 20, 1000)
y = torch.tensor([0, 1, 0])
for name, model in models:
    logits, aux = model(x, return_aux=True)
    if logits.shape != (3, 2) or not torch.isfinite(logits).all():
        raise RuntimeError(f'{name}: bad logits')
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise RuntimeError(f'{name}: bad gradients')
    print(name, 'parameters', sum(p.numel() for p in model.parameters()),
          'loss', round(loss.item(), 5), flush=True)
