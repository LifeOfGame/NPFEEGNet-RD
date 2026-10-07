"""Independently trained OpenBMI component controls for the frozen NPF model."""

import torch

from model.NPF_EEGNet import NPFEEGNet


class SpectralOnly(NPFEEGNet):
    """Train the five-band shared encoder without a raw-path contribution."""

    def forward(self, x, return_aux=False):
        _, aux = super().forward(x, return_aux=True)
        logits = aux["spectral_logits"]
        return (logits, aux) if return_aux else logits


class NoSpectralPrior(NPFEEGNet):
    """Keep branch capacity but feed five identical, unfiltered raw views."""

    def forward(self, x, return_aux=False):
        old_forward = self.filter_bank.forward
        self.filter_bank.forward = lambda data: data.squeeze(1)[:, None].expand(
            -1, self.num_bands, -1, -1
        )
        try:
            return super().forward(x, return_aux=return_aux)
        finally:
            self.filter_bank.forward = old_forward
