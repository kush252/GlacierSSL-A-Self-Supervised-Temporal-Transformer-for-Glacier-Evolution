"""Phase 7 helper — swappable scalar->768 temporal encodings (Temptrans_aarchi.md §1).

A common interface ``encode(scalar) -> (..., dim)`` with three families selectable by
config (sinusoidal | mlp | linear), kept swappable for later ablation. All temporal
scalars are computed from dates upstream; these modules only map a scalar to a vector.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalEncoding(nn.Module):
    """Continuous sinusoidal encoding of a scalar value (no fixed position table)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        self.register_buffer("div_term", div, persistent=False)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        # v: (...,) continuous scalar (years). -> (..., dim)
        v = v.unsqueeze(-1)                          # (...,1)
        args = v * self.div_term                     # (..., dim/2)
        enc = torch.zeros(*v.shape[:-1], self.dim, device=v.device, dtype=torch.float32)
        enc[..., 0::2] = torch.sin(args)
        enc[..., 1::2] = torch.cos(args)
        return enc


class MLPEncoding(nn.Module):
    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return self.net(v.unsqueeze(-1).float())


class LinearEncoding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(1, dim)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return self.proj(v.unsqueeze(-1).float())


def make_encoding(kind: str, dim: int) -> nn.Module:
    kind = kind.lower()
    if kind == "sinusoidal":
        return SinusoidalEncoding(dim)
    if kind == "mlp":
        return MLPEncoding(dim)
    if kind == "linear":
        return LinearEncoding(dim)
    raise ValueError(f"unknown time encoding '{kind}'")
