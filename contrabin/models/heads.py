"""Projection heads that map encoder outputs into the shared manifold M."""

from __future__ import annotations

import torch
from torch import nn


class LinearProjectionHead(nn.Module):
    """Linear projection head with a residual GELU block and LayerNorm."""

    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.projection(x)
        h = self.gelu(projected)
        h = self.fc(h)
        h = self.dropout(h)
        h = h + projected
        return self.layer_norm(h)


class NonLinearProjectionHead(nn.Module):
    """Deeper non-linear projection head (two FC layers + residual + LayerNorm)."""

    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.fc1 = nn.Linear(projection_dim, projection_dim)
        self.fc2 = nn.Linear(projection_dim, projection_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.projection(x)
        h = self.gelu(projected)
        h = self.fc1(h)
        h = self.gelu(h)
        h = self.fc2(h)
        h = self.dropout(h)
        h = h + projected
        return self.layer_norm(h)


def build_head(head_type: str, embedding_dim: int, projection_dim: int, dropout: float) -> nn.Module:
    if head_type == "linear":
        return LinearProjectionHead(embedding_dim, projection_dim, dropout)
    if head_type == "nonlinear":
        return NonLinearProjectionHead(embedding_dim, projection_dim, dropout)
    raise ValueError(f"Unknown head_type={head_type!r}; expected 'linear' or 'nonlinear'.")
