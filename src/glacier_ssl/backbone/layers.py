"""Shared transformer primitives (pre-LN attention, MLP, DropPath).

Used by the ViT (Phase 5) and TET (Phase 7). Kept minimal and standard; all widths and
depths come from config, never hardcoded.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()
    return x.div(keep) * mask


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Attention(nn.Module):
    """Multi-head self-attention with an optional key-padding mask.

    ``key_padding_mask`` is True at positions to KEEP (valid); invalid keys get -inf so
    padding is inert (build plan Cross-cutting rule #5).
    """

    def __init__(self, dim: int, num_heads: int, attn_dropout: float = 0.0,
                 proj_dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)            # (3,B,heads,N,head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B,heads,N,N)
        if key_padding_mask is not None:
            # True = keep. Expand to (B,1,1,N).
            invalid = ~key_padding_mask.view(B, 1, 1, N)
            attn = attn.masked_fill(invalid, float("-inf"))
        attn = attn.softmax(dim=-1)
        # rows that were all -inf (query at a fully-masked step) -> nan; zero them.
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))


class TransformerBlock(nn.Module):
    """Standard pre-LN transformer encoder block."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, attn_dropout: float = 0.0, drop_path_p: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, attn_dropout, dropout)
        self.drop_path = DropPath(drop_path_p)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x, key_padding_mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), key_padding_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
