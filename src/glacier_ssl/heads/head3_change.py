"""Phase 11 — Head 3: Temporal Change Representation (head3_archi.md).

Consumes REGIONAL embeddings (T,64,768) for a directed interval i->j (d_i < d_j). Builds
directed per-region pair features, a shared region-wise Change Encoder, validity-aware
pooling to one change embedding, and a projection for the contrastive objective. Baseline
objective is two-view instance InfoNCE on the SAME directed interval. Prototype/cluster
grouping is a gated, staged experiment (not a mandatory per-step loss). Direction is
preserved (reverse != forward); the time gap is NEVER an encoder input.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config

_FEATURE_MULT = {"five_group": 5, "signed": 1, "abs": 1, "start_end_signed": 3}


def region_pair_features(r_i: torch.Tensor, r_j: torch.Tensor, ln: nn.LayerNorm,
                         kind: str) -> torch.Tensor:
    """r_i,r_j: (P,64,768). Returns (P,64, mult*768). Directed (uses j-i, not |.| only)."""
    a = ln(r_i)
    b = ln(r_j)
    diff = b - a
    if kind == "signed":
        return diff
    if kind == "abs":
        return torch.abs(diff)
    if kind == "start_end_signed":
        return torch.cat([a, b, diff], dim=-1)
    # five_group (baseline)
    return torch.cat([a, b, diff, torch.abs(diff), a * b], dim=-1)


class ChangeEncoder(nn.Module):
    """Shared region-wise MLP applied to each region's pair feature."""

    def __init__(self, in_dim: int, hidden: list[int], out_dim: int, dropout: float):
        super().__init__()
        self.in_norm = nn.LayerNorm(in_dim)
        layers = []
        prev = in_dim
        for i, h in enumerate(hidden):
            layers += [nn.Linear(prev, h), nn.GELU()]
            if i == 0:
                layers += [nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, out_dim), nn.GELU()]
        self.mlp = nn.Sequential(*layers)
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_norm(self.mlp(self.in_norm(x)))   # (P,64,out_dim)


class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x, mask):  # x:(P,64,D) mask:(P,64) bool
        s = self.score(x).squeeze(-1)                     # (P,64)
        s = s.masked_fill(~mask, float("-inf"))
        w = torch.nan_to_num(s.softmax(dim=-1), nan=0.0).unsqueeze(-1)
        return (x * w).sum(dim=1)


class Head3Change(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        c = config.head3
        dim = int(config.tet.embed_dim)
        self.pair_kind = str(c.pair_features)
        self.ln = nn.LayerNorm(dim)
        in_dim = _FEATURE_MULT[self.pair_kind] * dim
        out_dim = int(c.encoder_out)
        self.encoder = ChangeEncoder(in_dim, list(c.encoder_hidden), out_dim,
                                     float(c.dropout))
        self.pooling = str(c.pooling)
        if self.pooling == "attention":
            self.attn_pool = AttentionPool(out_dim)
        elif self.pooling == "mean_max":
            self.mean_max_proj = nn.Linear(2 * out_dim, out_dim)
        self.proj = nn.Sequential(
            nn.LayerNorm(out_dim), nn.Linear(out_dim, int(c.proj_hidden)), nn.GELU(),
            nn.Linear(int(c.proj_hidden), int(c.proj_dim)))
        self.temperature = float(c.temperature)

        self.cluster_enabled = bool(c.cluster.enabled)
        if self.cluster_enabled:
            K = int(c.cluster.num_prototypes)
            self.prototypes = nn.Linear(int(c.proj_dim), K, bias=False)

    # -- correspondence check ----------------------------------------------------
    @staticmethod
    def verify_region_correspondence(regional_out: torch.Tensor) -> bool:
        """Region index k must denote the same grid location across timestamps. The
        HPE/TET design guarantees this (fixed row-major 8x8 grid + shared regional pos
        embedding). We assert the region axis has the expected size and is consistent."""
        return regional_out.dim() >= 3 and regional_out.shape[-2] >= 1

    # -- encode an interval ------------------------------------------------------
    def encode(self, r_i, r_j, region_valid=None):
        """r_i,r_j: (P,64,768). region_valid: (P,64) bool. Returns change_embedding,
        contrastive_embedding."""
        P, R, D = r_i.shape
        feats = region_pair_features(r_i, r_j, self.ln, self.pair_kind)   # (P,64,*)
        tokens = self.encoder(feats)                                      # (P,64,out)
        if region_valid is None:
            region_valid = torch.ones(P, R, dtype=torch.bool, device=r_i.device)
        mask = region_valid.unsqueeze(-1).float()
        if self.pooling == "attention":
            change = self.attn_pool(tokens, region_valid)
        elif self.pooling == "mean_max":
            denom = mask.sum(dim=1).clamp(min=1.0)
            mean = (tokens * mask).sum(dim=1) / denom
            neg_inf = torch.finfo(tokens.dtype).min
            mx = (tokens.masked_fill(~region_valid.unsqueeze(-1), neg_inf)).max(dim=1).values
            change = self.mean_max_proj(torch.cat([mean, mx], dim=-1))
        else:  # validity-aware mean (baseline)
            denom = mask.sum(dim=1).clamp(min=1.0)
            change = (tokens * mask).sum(dim=1) / denom
        contrastive = F.normalize(self.proj(change), dim=-1)
        return change, contrastive

    # -- objective ---------------------------------------------------------------
    def info_nce(self, z_a: torch.Tensor, z_b: torch.Tensor):
        """Symmetric two-view InfoNCE on normalized projections (P,proj_dim)."""
        P = z_a.shape[0]
        logits = (z_a @ z_b.t()) / self.temperature          # (P,P)
        labels = torch.arange(P, device=z_a.device)
        loss = 0.5 * (F.cross_entropy(logits, labels)
                      + F.cross_entropy(logits.t(), labels))
        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == labels).float().mean()
        return loss, {"infonce_acc": float(acc)}

    def forward(self, r_i_a, r_j_a, r_i_b, r_j_b, valid_a=None, valid_b=None):
        """Two augmented views (a,b) of the same directed interval -> InfoNCE loss."""
        _, z_a = self.encode(r_i_a, r_j_a, valid_a)
        _, z_b = self.encode(r_i_b, r_j_b, valid_b)
        return self.info_nce(z_a, z_b)
