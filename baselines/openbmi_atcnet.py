# SPDX-License-Identifier: Apache-2.0
# Adapted from Copyright (C) 2022 King Saud University, Saudi Arabia.
"""Paper-era ATCNet port for the matched-protocol OpenBMI control.

The topology in this file is a dependency-free PyTorch translation of the
author-maintained EEG-ATCNet implementation at commit
``d8fb0c10abc14b796dfbd317f53f56a68ebe29df`` (2022-09-08).  That source is
licensed under Apache-2.0.  The corresponding paper is:

    Altaheri et al., "Physics-Informed Attention Temporal Convolutional
    Network for EEG-Based Motor Imagery Classification," IEEE Transactions
    on Industrial Informatics, 2023. doi:10.1109/TII.2022.3197419.

This is a controlled architecture reproduction, not a claim that the original
authors used our OpenBMI preprocessing or training/evaluation protocol.  The
only wrapper-specific operation is optional per-trial, per-channel temporal
demeaning.  The default OpenBMI shape (20 channels, 1000 samples, 2 classes)
has exactly 113,338 trainable parameters, matching the translated topology.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


_SOURCE_COMMIT = "d8fb0c10abc14b796dfbd317f53f56a68ebe29df"
_OPENBMI_EXPECTED_PARAMETERS = 113_338


def _limit_output_unit_norm_(weight: torch.Tensor, maximum: float) -> None:
    """Apply a Keras-style per-output-unit max-norm constraint in place."""
    with torch.no_grad():
        rows = weight.reshape(weight.shape[0], -1)
        norms = rows.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        scales = (maximum / norms).clamp(max=1.0)
        weight.mul_(scales.reshape((-1,) + (1,) * (weight.ndim - 1)))


class _TemporalSameConv2d(nn.Conv2d):
    """Stride-one temporal convolution with TensorFlow ``same`` padding."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            stride=1,
            padding=0,
            bias=False,
        )
        self.temporal_kernel_size = int(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total = self.temporal_kernel_size - 1
        left = total // 2
        right = total - left
        return super().forward(F.pad(x, (left, right, 0, 0)))


class _SpatialDepthwiseConv2d(nn.Conv2d):
    """EEGNet spatial depthwise convolution with max-norm 1.0."""

    def __init__(
        self,
        temporal_filters: int,
        depth_multiplier: int,
        num_channels: int,
    ) -> None:
        super().__init__(
            temporal_filters,
            temporal_filters * depth_multiplier,
            kernel_size=(num_channels, 1),
            groups=temporal_filters,
            bias=False,
        )
        self.maximum_norm = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _limit_output_unit_norm_(self.weight, self.maximum_norm)
        return super().forward(x)


class _MaxNormLinear(nn.Linear):
    """Dense output layer reproducing Keras ``max_norm(axis=0)``."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        maximum_norm: float,
    ) -> None:
        super().__init__(in_features, out_features)
        if maximum_norm <= 0:
            raise ValueError("classifier_max_norm must be positive")
        self.maximum_norm = float(maximum_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _limit_output_unit_norm_(self.weight, self.maximum_norm)
        return super().forward(x)


class _MultiHeadSelfAttention(nn.Module):
    """Keras-MHA-compatible projections with independent head width."""

    def __init__(
        self,
        input_dim: int,
        head_dim: int,
        num_heads: int,
        score_dropout: float,
    ) -> None:
        super().__init__()
        if head_dim < 1 or num_heads < 1:
            raise ValueError("head_dim and num_heads must be positive")
        self.head_dim = int(head_dim)
        self.num_heads = int(num_heads)
        projected_dim = self.head_dim * self.num_heads
        self.query = nn.Linear(input_dim, projected_dim)
        self.key = nn.Linear(input_dim, projected_dim)
        self.value = nn.Linear(input_dim, projected_dim)
        self.output = nn.Linear(projected_dim, input_dim)
        self.score_dropout = nn.Dropout(score_dropout)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for layer in (self.query, self.key, self.value, self.output):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tokens.shape

        def project(layer: nn.Linear) -> torch.Tensor:
            return (
                layer(tokens)
                .reshape(batch, length, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )

        query = project(self.query)
        key = project(self.key)
        value = project(self.value)
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)
        weights = self.score_dropout(scores.softmax(dim=-1))
        attended = torch.matmul(weights, value)
        attended = attended.transpose(1, 2).reshape(batch, length, -1)
        return self.output(attended)


class _AttentionResidual(nn.Module):
    """Pre-normalized temporal attention from the paper-era implementation."""

    def __init__(
        self,
        channels: int,
        head_dim: int,
        num_heads: int,
        score_dropout: float,
        output_dropout: float,
    ) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(channels, eps=1e-6)
        self.attention = _MultiHeadSelfAttention(
            channels,
            head_dim,
            num_heads,
            score_dropout,
        )
        self.output_dropout = nn.Dropout(output_dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attended = self.attention(self.normalization(tokens))
        return tokens + self.output_dropout(attended)


class _CausalConv1d(nn.Conv1d):
    """Keras-style causal Conv1D with explicit left padding."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
            bias=True,
        )
        self.left_padding = (kernel_size - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, (self.left_padding, 0)))


class _TCNResidualBlock(nn.Module):
    """Two-convolution dilated residual unit from ATCNet's TCN block."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.first_conv = _CausalConv1d(channels, kernel_size, dilation)
        self.first_norm = nn.BatchNorm1d(
            channels, eps=1e-3, momentum=0.01
        )
        self.first_dropout = nn.Dropout(dropout)
        self.second_conv = _CausalConv1d(channels, kernel_size, dilation)
        self.second_norm = nn.BatchNorm1d(
            channels, eps=1e-3, momentum=0.01
        )
        self.second_dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for layer in (self.first_conv, self.second_conv):
            nn.init.kaiming_uniform_(layer.weight, a=0.0, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        for layer in (self.first_norm, self.second_norm):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.first_conv(x)
        x = self.first_norm(x)
        x = self.activation(x)
        x = self.first_dropout(x)
        x = self.second_conv(x)
        x = self.second_norm(x)
        x = self.activation(x)
        x = self.second_dropout(x)
        return self.activation(x + residual)


class _ATCStem(nn.Module):
    """The EEGNet-like convolutional stem used by the published ATCNet."""

    def __init__(
        self,
        num_channels: int,
        temporal_filters: int,
        depth_multiplier: int,
        temporal_kernel: int,
        refine_kernel: int,
        pool_sizes: tuple[int, int],
        dropout: float,
    ) -> None:
        super().__init__()
        feature_filters = temporal_filters * depth_multiplier
        self.temporal_conv = _TemporalSameConv2d(
            1, temporal_filters, temporal_kernel
        )
        self.temporal_norm = nn.BatchNorm2d(
            temporal_filters, eps=1e-3, momentum=0.01
        )
        self.spatial_conv = _SpatialDepthwiseConv2d(
            temporal_filters,
            depth_multiplier,
            num_channels,
        )
        self.spatial_norm = nn.BatchNorm2d(
            feature_filters, eps=1e-3, momentum=0.01
        )
        self.first_pool = nn.AvgPool2d(kernel_size=(1, pool_sizes[0]))
        self.first_dropout = nn.Dropout(dropout)

        # This is deliberately an ordinary channel-mixing convolution.  It is
        # not the separable/depthwise substitute used in some later ports.
        self.refine_conv = _TemporalSameConv2d(
            feature_filters,
            feature_filters,
            refine_kernel,
        )
        self.refine_norm = nn.BatchNorm2d(
            feature_filters, eps=1e-3, momentum=0.01
        )
        self.second_pool = nn.AvgPool2d(kernel_size=(1, pool_sizes[1]))
        self.second_dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for layer in (
            self.temporal_conv,
            self.spatial_conv,
            self.refine_conv,
        ):
            nn.init.xavier_uniform_(layer.weight)
        for layer in (
            self.temporal_norm,
            self.spatial_norm,
            self.refine_norm,
        ):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_norm(self.temporal_conv(x))
        x = self.spatial_norm(self.spatial_conv(x))
        x = self.activation(x)
        x = self.first_pool(x)
        x = self.first_dropout(x)
        x = self.refine_norm(self.refine_conv(x))
        x = self.activation(x)
        x = self.second_pool(x)
        x = self.second_dropout(x)
        return x.squeeze(2)


class OpenBMIATCNet(nn.Module):
    """Paper-era ATCNet under the project's matched OpenBMI protocol."""

    paper_model_name = "ATCNet (matched-protocol reproduction)"
    architecture_source_commit = _SOURCE_COMMIT
    architecture_source_license = "Apache-2.0"
    expected_openbmi_trainable_parameters = _OPENBMI_EXPECTED_PARAMETERS

    def __init__(
        self,
        num_channels: int,
        classes: int,
        n_times: int = 1000,
        temporal_filters: int = 16,
        depth_multiplier: int = 2,
        temporal_kernel: int = 64,
        refine_kernel: int = 16,
        pool_sizes: tuple[int, int] = (8, 7),
        conv_dropout: float = 0.3,
        windows: int = 5,
        head_dim: int = 8,
        num_heads: int = 2,
        attention_dropout: float = 0.5,
        attention_residual_dropout: float = 0.3,
        tcn_depth: int = 2,
        tcn_kernel: int = 4,
        tcn_dropout: float = 0.3,
        classifier_max_norm: float = 0.25,
        input_standardization: str = "demean",
    ) -> None:
        super().__init__()
        if num_channels < 1 or classes < 2 or n_times < 1:
            raise ValueError("num_channels, classes, and n_times are invalid")
        if temporal_filters < 1 or depth_multiplier < 1:
            raise ValueError("temporal_filters and depth_multiplier must be positive")
        if len(pool_sizes) != 2 or any(int(value) < 1 for value in pool_sizes):
            raise ValueError("pool_sizes must contain two positive integers")
        if windows < 1 or tcn_depth < 1 or tcn_kernel < 1:
            raise ValueError("windows, tcn_depth, and tcn_kernel must be positive")
        for name, probability in (
            ("conv_dropout", conv_dropout),
            ("attention_dropout", attention_dropout),
            ("attention_residual_dropout", attention_residual_dropout),
            ("tcn_dropout", tcn_dropout),
        ):
            if not 0.0 <= probability < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")

        self.num_channels = int(num_channels)
        self.classes = int(classes)
        self.n_times = int(n_times)
        self.windows = int(windows)
        self.input_standardization = input_standardization.lower()
        if self.input_standardization not in {"none", "demean"}:
            raise ValueError("input_standardization must be 'none' or 'demean'")

        first_pool, second_pool = (int(value) for value in pool_sizes)
        self.condensed_times = (self.n_times // first_pool) // second_pool
        self.window_length = self.condensed_times - self.windows + 1
        self.tcn_receptive_field = 1 + 2 * (int(tcn_kernel) - 1) * sum(
            2**depth for depth in range(int(tcn_depth))
        )
        if self.window_length < 1:
            raise ValueError(
                "Input length and pooling leave no valid ATCNet windows: "
                f"condensed_times={self.condensed_times}, windows={self.windows}"
            )

        feature_filters = int(temporal_filters) * int(depth_multiplier)
        self.stem = _ATCStem(
            self.num_channels,
            int(temporal_filters),
            int(depth_multiplier),
            int(temporal_kernel),
            int(refine_kernel),
            (first_pool, second_pool),
            float(conv_dropout),
        )
        self.attention_blocks = nn.ModuleList(
            [
                _AttentionResidual(
                    feature_filters,
                    int(head_dim),
                    int(num_heads),
                    float(attention_dropout),
                    float(attention_residual_dropout),
                )
                for _ in range(self.windows)
            ]
        )
        self.tcn_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        _TCNResidualBlock(
                            feature_filters,
                            int(tcn_kernel),
                            dilation=2**depth,
                            dropout=float(tcn_dropout),
                        )
                        for depth in range(int(tcn_depth))
                    ]
                )
                for _ in range(self.windows)
            ]
        )
        self.classifiers = nn.ModuleList(
            [
                _MaxNormLinear(
                    feature_filters,
                    self.classes,
                    float(classifier_max_norm),
                )
                for _ in range(self.windows)
            ]
        )
        for classifier in self.classifiers:
            nn.init.xavier_uniform_(classifier.weight)
            nn.init.zeros_(classifier.bias)

        if self._is_reference_openbmi_shape(
            temporal_filters=int(temporal_filters),
            depth_multiplier=int(depth_multiplier),
            temporal_kernel=int(temporal_kernel),
            refine_kernel=int(refine_kernel),
            pool_sizes=(first_pool, second_pool),
            windows=self.windows,
            head_dim=int(head_dim),
            num_heads=int(num_heads),
            tcn_depth=int(tcn_depth),
            tcn_kernel=int(tcn_kernel),
        ):
            observed = sum(parameter.numel() for parameter in self.parameters())
            if observed != self.expected_openbmi_trainable_parameters:
                raise RuntimeError(
                    "ATCNet topology regression: expected "
                    f"{self.expected_openbmi_trainable_parameters:,} trainable "
                    f"parameters, observed {observed:,}."
                )

    def _is_reference_openbmi_shape(self, **topology: Any) -> bool:
        return (
            self.num_channels == 20
            and self.classes == 2
            and self.n_times == 1000
            and topology
            == {
                "temporal_filters": 16,
                "depth_multiplier": 2,
                "temporal_kernel": 64,
                "refine_kernel": 16,
                "pool_sizes": (8, 7),
                "windows": 5,
                "head_dim": 8,
                "num_heads": 2,
                "tcn_depth": 2,
                "tcn_kernel": 4,
            }
        )

    @property
    def model_metadata(self) -> dict[str, Any]:
        """Immutable-by-convention provenance for result manifests."""
        return {
            "paper_model_name": self.paper_model_name,
            "architecture_source_commit": self.architecture_source_commit,
            "architecture_source_license": self.architecture_source_license,
            "trainable_parameters": sum(
                parameter.numel() for parameter in self.parameters()
            ),
            "condensed_times": self.condensed_times,
            "window_length": self.window_length,
            "tcn_receptive_field": self.tcn_receptive_field,
            "input_standardization": self.input_standardization,
        }

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(
                "OpenBMIATCNet expects (batch, 1, channels, samples)"
            )
        if x.shape[2] != self.num_channels or x.shape[3] != self.n_times:
            raise ValueError(
                "OpenBMIATCNet input shape does not match its constructor: "
                f"expected channels/time=({self.num_channels}, {self.n_times}), "
                f"got ({x.shape[2]}, {x.shape[3]})"
            )
        if self.input_standardization == "demean":
            x = x - x.mean(dim=-1, keepdim=True)

        features = self.stem(x)
        window_logits = []
        for index, (attention, tcn, classifier) in enumerate(
            zip(self.attention_blocks, self.tcn_blocks, self.classifiers)
        ):
            window = features[
                ..., index : index + self.window_length
            ].transpose(1, 2)
            window = attention(window).transpose(1, 2)
            encoded = tcn(window)[..., -1]
            window_logits.append(classifier(encoded))
        logits = torch.stack(window_logits, dim=1).mean(dim=1)
        if return_aux:
            return logits, {}
        return logits


__all__ = ["OpenBMIATCNet"]
