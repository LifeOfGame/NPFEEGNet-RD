"""Predeclared, label-free OpenBMI preprocessing controls.

The primary OpenBMI experiment subtracts the temporal mean from only the raw
NPFEEGNet branch.  This module implements the conventional broadband high-pass
control used to determine whether that branch-specific operation merely
duplicates ordinary signal preprocessing.

Filtering is deliberately performed on an array shaped
``(trials, channels, samples)`` along its final axis.  SciPy therefore treats
every trial/channel trace independently: trials are never concatenated, and
Session 1 and Session 2 are loaded and filtered in separate calls.  No labels,
training-set statistics, or evaluation-set statistics are consumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt


@dataclass(frozen=True)
class HighPassSpec:
    """Fixed offline high-pass definition for the OpenBMI control."""

    sampling_rate: float = 250.0
    cutoff_hz: float = 1.0
    order: int = 4

    def validate(self) -> None:
        if not np.isfinite(self.sampling_rate) or self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be finite and positive")
        if (
            not np.isfinite(self.cutoff_hz)
            or self.cutoff_hz <= 0
            or self.cutoff_hz >= self.sampling_rate / 2
        ):
            raise ValueError("cutoff_hz must lie strictly between 0 and Nyquist")
        if not isinstance(self.order, int) or isinstance(self.order, bool):
            raise TypeError("order must be an integer")
        if self.order < 1:
            raise ValueError("order must be positive")

    def design_sos(self) -> np.ndarray:
        self.validate()
        return butter(
            self.order,
            self.cutoff_hz,
            btype="highpass",
            fs=self.sampling_rate,
            output="sos",
        )


def highpass_trials(
    data: np.ndarray,
    *,
    sampling_rate: float = 250.0,
    cutoff_hz: float = 1.0,
    order: int = 4,
    padtype: str = "odd",
    padlen_samples: int = 15,
) -> np.ndarray:
    """Zero-phase high-pass each trial/channel independently.

    ``sosfiltfilt`` is appropriate here because the experiment is offline.  It
    is applied only along the last axis of the already separated trials, so
    its forward/backward padding cannot cross a trial, phase, or session
    boundary.  The fixed filter uses no fitted parameters and is consequently
    identical for training, validation, and evaluation data.
    """

    array = np.asarray(data)
    if array.ndim != 3:
        raise ValueError(
            "OpenBMI high-pass expects (trials, channels, samples), "
            f"got {array.shape}"
        )
    if array.shape[-1] < 16:
        raise ValueError("OpenBMI high-pass requires at least 16 time samples")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError("OpenBMI high-pass input must have a floating dtype")
    if not np.isfinite(array).all():
        raise ValueError("OpenBMI high-pass input contains non-finite values")
    if padtype != "odd":
        raise ValueError("The frozen OpenBMI high-pass requires odd padding")
    if (
        not isinstance(padlen_samples, int)
        or isinstance(padlen_samples, bool)
        or not 0 <= padlen_samples < array.shape[-1] - 1
    ):
        raise ValueError("padlen_samples must be an integer in [0, samples - 2]")

    specification = HighPassSpec(
        sampling_rate=float(sampling_rate),
        cutoff_hz=float(cutoff_hz),
        order=order,
    )
    filtered = sosfiltfilt(
        specification.design_sos(),
        array,
        axis=-1,
        padtype=padtype,
        padlen=padlen_samples,
    )
    if not np.isfinite(filtered).all():
        raise FloatingPointError("OpenBMI high-pass produced non-finite values")
    return np.ascontiguousarray(filtered.astype(array.dtype, copy=False))


def highpass_from_config(data: np.ndarray, preprocessing: dict) -> np.ndarray:
    """Apply the exact filter declared in a preprocessing-control YAML."""

    if preprocessing.get("method") != "butterworth_highpass":
        raise ValueError("Expected preprocessing method 'butterworth_highpass'")
    if preprocessing.get("application") != "per_trial_per_channel":
        raise ValueError("High-pass must remain per_trial_per_channel")
    if preprocessing.get("phase") != "zero_phase_sosfiltfilt":
        raise ValueError("High-pass phase must remain zero_phase_sosfiltfilt")
    if preprocessing.get("scope") != "all_model_branches":
        raise ValueError("High-pass must feed all model branches")
    return highpass_trials(
        data,
        sampling_rate=preprocessing["sampling_rate"],
        cutoff_hz=preprocessing["cutoff_hz"],
        order=preprocessing["order"],
        padtype=preprocessing["padtype"],
        padlen_samples=preprocessing["padlen_samples"],
    )
