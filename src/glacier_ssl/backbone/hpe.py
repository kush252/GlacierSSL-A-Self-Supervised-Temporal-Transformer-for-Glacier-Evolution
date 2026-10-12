"""Phase 6 — Hierarchical Patch Embedding (hpe.md).

Compresses each timestamp's 32x32 patch grid to an 8x8 regional grid with two learned
2x2 patch-merge stages (width preserved at 768), then adds a learned regional positional
embedding. Each timestamp is an independent batch element; features are NEVER merged across
timestamps. Input (N,1024,768) -> output (N,64,768).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import Config


class PatchMerge(nn.Module):
    """One 2x2 learned merge stage: concat(TL,TR,BL,BR) -> LN(4C) -> Linear(4C->C)."""

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, G, G, C) -> (N, G/2, G/2, C). Fixed neighborhood order TL,TR,BL,BR."""
        n, g, _, c = x.shape
        assert g % 2 == 0, f"grid {g} not divisible by 2"
        x00 = x[:, 0::2, 0::2, :]     # top-left
        x01 = x[:, 0::2, 1::2, :]     # top-right
        x10 = x[:, 1::2, 0::2, :]     # bottom-left
        x11 = x[:, 1::2, 1::2, :]     # bottom-right
        merged = torch.cat([x00, x01, x10, x11], dim=-1)   # (N, G/2, G/2, 4C)
        merged = self.norm(merged)
        merged = self.reduction(merged)                    # (N, G/2, G/2, C)
        return self.drop(merged)


class HierarchicalPatchEmbedding(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        h = config.hpe
        self.dim = int(h.embed_dim)
        self.input_grid = int(h.input_grid)                # 32
        self.num_stages = int(h.num_merge_stages)          # 2 (fixed -> 8x8)
        dropout = float(h.dropout)

        out_grid = self.input_grid
        stages = []
        for _ in range(self.num_stages):
            stages.append(PatchMerge(self.dim, dropout))
            out_grid //= 2
        self.stages = nn.ModuleList(stages)
        self.out_grid = out_grid                           # 8
        self.num_regions = out_grid * out_grid             # 64

        # Learned 2D regional positional embedding (1, 64, 768), broadcast over timestamps.
        self.regional_pos_embed = nn.Parameter(torch.zeros(1, self.num_regions, self.dim))
        nn.init.trunc_normal_(self.regional_pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 1024, 768) -> (N, 64, 768). N folds (B,T); no cross-timestamp mixing."""
        n, tokens, c = x.shape
        g = self.input_grid
        assert tokens == g * g, f"expected {g*g} tokens, got {tokens}"
        assert c == self.dim, f"expected dim {self.dim}, got {c}"
        x = x.view(n, g, g, c)                              # row-major reshape
        for stage in self.stages:
            x = stage(x)                                    # (N, g/2, g/2, C)
        x = x.reshape(n, self.num_regions, c)               # flatten row-major -> (N,64,C)
        x = x + self.regional_pos_embed
        return x
