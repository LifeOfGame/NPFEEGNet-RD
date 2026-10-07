"""Clean-room EEG Conformer architecture for the matched OpenBMI control.

This module implements the network topology described by Song et al., "EEG
Conformer: Convolutional Transformer for EEG Decoding and Visualization",
IEEE TNSRE 31 (2023), 710--719, doi:10.1109/TNSRE.2022.3230250.  It was written
against the paper and public architecture documentation using only native
PyTorch layers; no source code from the authors' GPL-3.0 repository is copied.

The model is deliberately an architecture reproduction under this project's
common training protocol, not a claim to reproduce the paper's published
accuracy.  The OpenBMI control adds only trial-wise/channel-wise temporal
demeaning before the published convolutional tokenizer.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class _ConvolutionalTokenizer(nn.Module):
    """Temporal-spatial convolution followed by overlapping temporal tokens."""

    def __init__(
        self,
        num_channels: int,
        embedding_dim: int,
        temporal_kernel: int,
        pool_length: int,
        pool_stride: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.temporal = nn.Conv2d(
            1,
            embedding_dim,
            kernel_size=(1, temporal_kernel),
            bias=True,
        )
        self.spatial = nn.Conv2d(
            embedding_dim,
            embedding_dim,
            kernel_size=(num_channels, 1),
            bias=True,
        )
        self.normalization = nn.BatchNorm2d(embedding_dim)
        self.activation = nn.ELU()
        self.pool = nn.AvgPool2d(
            kernel_size=(1, pool_length),
            stride=(1, pool_stride),
        )
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv2d(
            embedding_dim,
            embedding_dim,
            kernel_size=(1, 1),
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.normalization(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.projection(x)
        return x.squeeze(2).transpose(1, 2)


class _PaperScaledMultiheadAttention(nn.Module):
    """Multi-head attention with the EEG Conformer embedding-width scale.

    EEG Conformer's published implementation divides attention logits by
    ``sqrt(embedding_dim)``.  Native ``nn.MultiheadAttention`` instead uses
    ``sqrt(head_dim)``; an explicit implementation avoids that silent topology
    change while remaining independently written.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.embedding_dim = int(embedding_dim)
        self.num_heads = int(num_heads)
        self.head_dim = embedding_dim // num_heads
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.output = nn.Linear(embedding_dim, embedding_dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        return x.reshape(
            batch, tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self._split_heads(self.query(x))
        key = self._split_heads(self.key(x))
        value = self._split_heads(self.value(x))
        scores = query @ key.transpose(-2, -1)
        weights = self.attention_dropout(
            (scores / math.sqrt(self.embedding_dim)).softmax(dim=-1)
        )
        attended = weights @ value
        batch, _, tokens, _ = attended.shape
        merged = attended.transpose(1, 2).reshape(
            batch, tokens, self.embedding_dim
        )
        return self.output(merged)


class _TransformerBlock(nn.Module):
    """Pre-normalized self-attention and fourfold feed-forward residuals."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        attention_dropout: float,
        feedforward_expansion: int,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if feedforward_expansion < 1:
            raise ValueError("feedforward_expansion must be positive")

        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.attention = _PaperScaledMultiheadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
        )
        self.attention_residual_dropout = nn.Dropout(attention_dropout)

        hidden_dim = embedding_dim * feedforward_expansion
        self.feedforward_norm = nn.LayerNorm(embedding_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(attention_dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.feedforward_residual_dropout = nn.Dropout(attention_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(x)
        attended = self.attention(normalized)
        x = x + self.attention_residual_dropout(attended)
        feedforward = self.feedforward(self.feedforward_norm(x))
        return x + self.feedforward_residual_dropout(feedforward)


class OpenBMIEEGConformer(nn.Module):
    """EEG Conformer paper architecture under the matched OpenBMI protocol.

    Parameters retain the published defaults: 40 convolutional/token features,
    a 25-sample temporal kernel, 75/15 average pooling, six transformer blocks,
    ten attention heads, and the 256-to-32 classification head.
    """

    paper_model_name = "EEGConformer-Matched-Demean"
    paper_doi = "10.1109/TNSRE.2022.3230250"
    implementation_provenance = (
        "clean-room native-PyTorch implementation from the paper and public "
        "architecture documentation; no GPL source copied"
    )

    def __init__(
        self,
        num_channels: int,
        classes: int,
        n_times: int = 1000,
        embedding_dim: int = 40,
        temporal_kernel: int = 25,
        pool_length: int = 75,
        pool_stride: int = 15,
        convolution_dropout: float = 0.5,
        transformer_depth: int = 6,
        num_heads: int = 10,
        attention_dropout: float = 0.5,
        feedforward_expansion: int = 4,
        classifier_hidden: tuple[int, int] = (256, 32),
        classifier_dropouts: tuple[float, float] = (0.5, 0.3),
        input_standardization: str = "demean",
    ) -> None:
        super().__init__()
        if min(num_channels, classes, n_times, embedding_dim) < 1:
            raise ValueError("channel, class, time, and embedding sizes must be positive")
        if temporal_kernel > n_times:
            raise ValueError("temporal_kernel cannot exceed n_times")
        if pool_length > n_times - temporal_kernel + 1:
            raise ValueError("pool_length leaves no convolutional token")
        if pool_stride < 1 or transformer_depth < 1:
            raise ValueError("pool_stride and transformer_depth must be positive")
        if len(classifier_hidden) != 2 or min(classifier_hidden) < 1:
            raise ValueError("classifier_hidden must contain two positive widths")
        if len(classifier_dropouts) != 2:
            raise ValueError("classifier_dropouts must contain two probabilities")
        probabilities = (
            convolution_dropout,
            attention_dropout,
            *classifier_dropouts,
        )
        if any(not 0.0 <= probability < 1.0 for probability in probabilities):
            raise ValueError("dropout probabilities must lie in [0, 1)")

        self.input_standardization = input_standardization.lower()
        if self.input_standardization not in {"none", "demean"}:
            raise ValueError("input_standardization must be 'none' or 'demean'")
        self.num_channels = int(num_channels)
        self.n_times = int(n_times)
        self.embedding_dim = int(embedding_dim)
        self.num_tokens = (
            (n_times - temporal_kernel + 1 - pool_length) // pool_stride + 1
        )
        if self.num_tokens < 1:
            raise ValueError("tokenizer produces no temporal tokens")

        self.tokenizer = _ConvolutionalTokenizer(
            num_channels=num_channels,
            embedding_dim=embedding_dim,
            temporal_kernel=temporal_kernel,
            pool_length=pool_length,
            pool_stride=pool_stride,
            dropout=convolution_dropout,
        )
        self.transformer = nn.Sequential(
            *[
                _TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    feedforward_expansion=feedforward_expansion,
                )
                for _ in range(transformer_depth)
            ]
        )
        first_hidden, second_hidden = classifier_hidden
        first_dropout, second_dropout = classifier_dropouts
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.num_tokens * embedding_dim, first_hidden),
            nn.ELU(),
            nn.Dropout(first_dropout),
            nn.Linear(first_hidden, second_hidden),
            nn.ELU(),
            nn.Dropout(second_dropout),
            nn.Linear(second_hidden, classes),
        )

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(
                "OpenBMIEEGConformer expects (batch, 1, channels, samples)"
            )
        if x.shape[2:] != (self.num_channels, self.n_times):
            raise ValueError(
                "OpenBMIEEGConformer input shape does not match its constructor: "
                f"expected channels/time {(self.num_channels, self.n_times)}, "
                f"received {tuple(x.shape[2:])}"
            )
        if self.input_standardization == "demean":
            x = x - x.mean(dim=-1, keepdim=True)
        tokens = self.tokenizer(x)
        if tokens.shape[1:] != (self.num_tokens, self.embedding_dim):
            raise RuntimeError(
                "EEG Conformer tokenizer topology drift: expected "
                f"{(self.num_tokens, self.embedding_dim)}, got "
                f"{tuple(tokens.shape[1:])}"
            )
        logits = self.classifier(self.transformer(tokens))
        if return_aux:
            return logits, {}
        return logits


__all__ = ["OpenBMIEEGConformer"]
