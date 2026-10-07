"""One-factor boundary modifications of the frozen NPFEEGNet topology."""
import torch.nn.functional as F
from model.NPF_EEGNet import NPFEEGNet, ResponseNormalizedFilterBank

class ReflectionFFT(ResponseNormalizedFilterBank):
    def forward(self, x):
        # Fixed 1 s each side at 250 Hz; no labels or between-trial statistics.
        n = x.shape[-1]
        padded = F.pad(x.squeeze(1), (250, 250), mode='reflect').unsqueeze(1)
        y = super().forward(padded)
        return y[..., 250:250+n]

def make_model(boundary):
    class BoundaryNPF(NPFEEGNet):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            if boundary == 'fft_reflect':
                # Same class state/parameters; only boundary treatment changes.
                self.filter_bank.__class__ = ReflectionFFT
            elif boundary == 'raw_reflect':
                # Only raw first temporal convolution. Same weight, width,
                # asymmetric 31-left/32-right same padding, and initialization.
                self.raw_features[0].padding_mode = 'reflect'
            elif boundary != 'current':
                raise ValueError(boundary)
    return BoundaryNPF
