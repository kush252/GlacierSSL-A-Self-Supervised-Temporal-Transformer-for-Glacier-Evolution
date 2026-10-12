"""Phase 8 — Backbone integration (ViT -> HPE -> TET).

One wired backbone honoring the Phase 0 tensor-contract map. Accepts padded variable-length
batches (B,T,6,H,W) with a timestamp mask and date-derived temporal metadata, folds the
(B,T) timestamps through the shared per-timestamp ViT+HPE, then runs the temporal TET. The
same frozen-interface backbone is shared by all three heads.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import Config
from .hpe import HierarchicalPatchEmbedding
from .tet import TemporalEvolutionTransformer, seasons_to_ids
from .vit import MultispectralViT


class GlacierBackbone(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.cfg = config
        self.vit = MultispectralViT(config)
        self.hpe = HierarchicalPatchEmbedding(config)
        self.tet = TemporalEvolutionTransformer(config)
        self.num_regions = self.hpe.num_regions
        self.embed_dim = self.vit.embed_dim

    # -- per-timestamp spatial path (shared weights, no temporal op) -------------
    def encode_regional(self, imgs: torch.Tensor) -> torch.Tensor:
        """imgs: (B,T,6,H,W) -> regional tokens (B,T,64,768) via ViT+HPE.

        (B,T) are folded into the batch dimension; each timestamp is independent.
        """
        B, T, C, H, W = imgs.shape
        folded = imgs.reshape(B * T, C, H, W)
        tokens = self.vit(folded)                 # (B*T, 1024, 768)
        regional = self.hpe(tokens)               # (B*T, 64, 768)
        return regional.view(B, T, self.num_regions, self.embed_dim)

    # -- full temporal forward ---------------------------------------------------
    def forward(self, imgs, timestamp_mask, elapsed_years, span_years, seasons,
                region_mask=None):
        """
        imgs:           (B,T,6,H,W) padded
        timestamp_mask: (B,T) bool, True=valid (real) timestamp
        elapsed_years:  (B,T) float from dates
        span_years:     (B,) float
        seasons:        list[list[str]] length B
        """
        regional = self.encode_regional(imgs)                     # (B,T,64,768)
        season_ids = seasons_to_ids(seasons, imgs.shape[1], imgs.device)
        return self.tet(regional, season_ids, elapsed_years, span_years,
                        timestamp_mask, region_mask=region_mask)

    # -- convenience for a single view already carrying its own T ----------------
    def forward_view(self, view: dict, elapsed_years, span_years, seasons,
                     region_mask=None):
        return self.forward(view["img"], view["timestamp_mask"], elapsed_years,
                            span_years, seasons, region_mask=region_mask)
