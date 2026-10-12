"""Phase 10 — Head 2: Temporal Distance Prediction (head2_archi.md).

Symmetric elapsed-time regression on pairs of TIMESTAMP embeddings from the same glacier.
Pair feature = concat(|u_i-u_j|, u_i*u_j) with u = LayerNorm(z) — symmetric, so swapping
i/j leaves the prediction identical. The predictor NEVER sees delta_years / date diffs /
index diffs. Loss = SmoothL1 in log1p-years space; metrics convert back with expm1.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config


class Head2Distance(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        c = config.head2
        dim = int(config.tet.embed_dim)
        self.l2_normalize = bool(c.l2_normalize)
        self.ln = nn.LayerNorm(dim)
        in_dim = 2 * dim                       # concat(abs diff, product)
        hidden = list(c.mlp_hidden)
        self.input_norm = nn.LayerNorm(in_dim)
        layers = []
        prev = in_dim
        for i, h in enumerate(hidden):
            layers += [nn.Linear(prev, h), nn.GELU()]
            if i == 0:
                layers += [nn.Dropout(float(c.dropout))]
            prev = h
        layers += [nn.Linear(prev, 1), nn.Softplus()]
        self.mlp = nn.Sequential(*layers)
        self.loss_kind = str(c.loss)

    def pair_features(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        u_i = self.ln(z_i)
        u_j = self.ln(z_j)
        if self.l2_normalize:
            u_i = F.normalize(u_i, dim=-1)
            u_j = F.normalize(u_j, dim=-1)
        return torch.cat([torch.abs(u_i - u_j), u_i * u_j], dim=-1)   # (P, 2*dim)

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        feats = self.input_norm(self.pair_features(z_i, z_j))
        return self.mlp(feats).squeeze(-1)     # (P,) predicted years, non-negative

    def compute_loss(self, pred_years: torch.Tensor, target_years: torch.Tensor):
        yp = torch.log1p(pred_years)
        yt = torch.log1p(target_years)
        if self.loss_kind == "mae":
            loss = F.l1_loss(yp, yt)
        elif self.loss_kind == "mse":
            loss = F.mse_loss(yp, yt)
        else:
            loss = F.smooth_l1_loss(yp, yt)
        with torch.no_grad():
            mae_years = (pred_years - target_years).abs().mean()
        return loss, {"mae_years": float(mae_years)}
