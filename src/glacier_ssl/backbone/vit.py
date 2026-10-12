"""Phase 5 — Multispectral Spatial ViT (VIT_aarchi.md).

Encodes ONE Sentinel-2 timestamp at a time into spatially contextualized patch tokens.
No temporal operation, no CLS token, no pooling, no prediction head. Shared weights across
all timestamps. Output contract: (B, 1024, 768).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import Config
from ..utils.logging_utils import get_logger
from .layers import TransformerBlock


class MultispectralViT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        c = config.vit
        self.image_size = int(c.image_size)
        self.patch_size = int(c.patch_size)
        self.in_chans = int(c.in_chans)
        self.embed_dim = int(c.embed_dim)
        self.depth = int(c.depth)
        self.num_heads = int(c.num_heads)
        self.mlp_ratio = float(c.mlp_ratio)
        assert self.image_size % self.patch_size == 0, "image_size must divide patch_size"
        self.grid = self.image_size // self.patch_size
        self.num_patches = self.grid * self.grid

        # Single Conv2D projection fusing all six spectral channels per patch.
        self.patch_embed = nn.Conv2d(self.in_chans, self.embed_dim,
                                     kernel_size=self.patch_size, stride=self.patch_size)
        # Learnable absolute 2D spatial positional embeddings (1, num_patches, dim).
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        dpr = torch.linspace(0, float(c.drop_path), self.depth).tolist()
        self.blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, self.num_heads, self.mlp_ratio,
                             dropout=float(c.dropout), attn_dropout=float(c.dropout),
                             drop_path_p=dpr[i])
            for i in range(self.depth)
        ])
        self.norm = nn.LayerNorm(self.embed_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 6, H, W) -> (B, 1024, 768). Each timestamp processed independently."""
        assert x.shape[1] == self.in_chans, \
            f"expected {self.in_chans} channels, got {x.shape[1]}"
        assert x.shape[-2] == self.image_size and x.shape[-1] == self.image_size, \
            f"expected {self.image_size}x{self.image_size}, got {tuple(x.shape[-2:])}"
        x = self.patch_embed(x)                      # (B, dim, grid, grid)
        x = x.flatten(2).transpose(1, 2)             # (B, num_patches, dim)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)                               # no mask: spatial tokens all valid
        x = self.norm(x)
        return x                                     # (B, 1024, 768)


def load_pretrained(vit: MultispectralViT, ckpt_path: str, config: Config) -> dict:
    """Checkpoint-compatibility guards (VIT_aarchi.md §7).

    Checks patch size, dim, depth, in-channels, and positional-embedding format BEFORE
    loading. Incompatible parts are reported and skipped (adapt/reinitialize deliberately)
    rather than silently assumed compatible. Returns a report dict.
    """
    log = get_logger("vit")
    state = torch.load(ckpt_path, map_location="cpu")
    state = state.get("model", state.get("state_dict", state))
    report = {"loaded": [], "skipped": [], "reasons": {}}

    target = vit.state_dict()
    to_load = {}
    for k, v in state.items():
        key = k.replace("module.", "")
        if key not in target:
            report["skipped"].append(key)
            report["reasons"][key] = "absent in model"
            continue
        if tuple(target[key].shape) != tuple(v.shape):
            report["skipped"].append(key)
            report["reasons"][key] = (f"shape {tuple(v.shape)} != {tuple(target[key].shape)}")
            continue
        to_load[key] = v
        report["loaded"].append(key)

    missing, unexpected = vit.load_state_dict(to_load, strict=False)
    report["missing_after_load"] = list(missing)
    log.info("Pretrained load: %d tensors loaded, %d skipped (incompatible/absent).",
             len(report["loaded"]), len(report["skipped"]))
    if "patch_embed.weight" in report["reasons"]:
        log.warning("Patch projection incompatible (%s) -> reinitialized deliberately.",
                    report["reasons"]["patch_embed.weight"])
    return report
