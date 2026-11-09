"""Phase 5 — Multispectral Spatial ViT (VIT_aarchi.md).

Encodes ONE Sentinel-2 timestamp at a time into spatially contextualized patch tokens.
No temporal operation, no CLS token, no pooling, no prediction head. Shared weights across
all timestamps. Output contract: (B, 1024, 768).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

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
        # Recompute block activations during backward instead of storing them. One training
        # step runs this backbone seven times (base + 4 DINO views + 2 Head-3 views) and
        # every call's activations stay alive until backward, so stored activations — not
        # parameters — are what exhausts a 16 GB GPU. Identical math, ~30% more compute.
        self.grad_checkpoint = bool(config.get("vit.grad_checkpoint", False))
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
        ckpt = self.grad_checkpoint and self.training and torch.is_grad_enabled()
        for blk in self.blocks:
            if ckpt:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)                           # no mask: spatial tokens all valid
        x = self.norm(x)
        return x                                     # (B, 1024, 768)


_PREFIXES = ("module.", "encoder.", "backbone.", "model.", "vit.")
_IGNORE = ("cls_token", "mask_token", "decoder", "head.", "fc_norm", "dist_token")


def _normalise_key(k: str) -> str:
    changed = True
    while changed:
        changed = False
        for pre in _PREFIXES:
            if k.startswith(pre):
                k, changed = k[len(pre):], True
    return k.replace("patch_embed.proj.", "patch_embed.")


def _read_state(ckpt_path: str) -> dict:
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:                    # MAE-style files also pickle their argparse args
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for key in ("model", "state_dict", "teacher", "student"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break
    return state


def _adapt_patch_embed(w: torch.Tensor, vit: "MultispectralViT", config: Config,
                       report: dict) -> torch.Tensor | None:
    """(D,C,p,p) or (D,C,t,p,p) -> (D,in_chans,p,p), or None if it cannot be adapted."""
    if w.dim() == 5:                     # 3-D tubelet embedding (e.g. Prithvi): one frame
        report["notes"].append(f"patch_embed: 3-D kernel {tuple(w.shape)} averaged over "
                               "its temporal axis")
        w = w.mean(dim=2)
    idx = config.get("vit.pretrained_band_indices", None)
    if idx is not None:
        w = w[:, list(idx)]
        report["notes"].append(f"patch_embed: input bands selected {list(idx)}")
    if w.shape[1] != vit.in_chans:
        report["reasons"]["patch_embed.weight"] = (
            f"checkpoint has {w.shape[1]} input bands, model {vit.in_chans}; set "
            "vit.pretrained_band_indices to map them")
        return None
    if w.shape[-1] != vit.patch_size:
        report["reasons"]["patch_embed.weight"] = (
            f"patch size {w.shape[-1]} != {vit.patch_size}")
        return None
    return w


def _adapt_pos_embed(pos: torch.Tensor, vit: "MultispectralViT", config: Config,
                     report: dict, has_cls: bool) -> torch.Tensor | None:
    """Any (1,N,D) ViT positional table -> (1, grid*grid, D).

    Handles a leading cls position, multi-frame tables (num_frames * g*g, as in video/
    temporal MAEs; frames are averaged, which keeps the spatial component) and a different
    grid size (bicubic resampling, the standard ViT resolution change).

    Whether position 0 is a cls slot is taken from the checkpoint itself (a ``cls_token``
    tensor), never guessed from the count: 1 + 3*4*4 = 49 is also a 7x7 grid, and guessing
    would silently blend the cls position and three frames into one spatial table.
    """
    n, d = pos.shape[1], pos.shape[2]
    if d != vit.embed_dim:
        return None
    frames = config.get("vit.pretrained_num_frames", None)

    def split(m):
        if frames:
            f = int(frames)
            return (f, int(round((m / f) ** 0.5))) if m % f == 0 else (None, None)
        for f in range(1, 9):            # prefer a single frame when the table is square
            if m % f == 0:
                g = int(round((m / f) ** 0.5))
                if g * g * f == m:
                    return f, g
        return None, None

    m = n - 1 if has_cls else n
    body = pos[:, 1:] if has_cls else pos
    f, g = split(m)
    if f is None or g * g * f != m:
        report["reasons"]["pos_embed"] = (f"cannot factor {m} positions into frames x grid "
                                          f"(cls slot: {has_cls})")
        return None
    if has_cls:
        report["notes"].append("pos_embed: leading cls position dropped")
    grid = body.reshape(f, g, g, d).mean(dim=0)                     # (g,g,D)
    if f > 1:
        report["notes"].append(f"pos_embed: {f} frames averaged into one spatial table")
    if g != vit.grid:
        grid = torch.nn.functional.interpolate(
            grid.permute(2, 0, 1).unsqueeze(0).float(), size=(vit.grid, vit.grid),
            mode="bicubic", align_corners=False).squeeze(0).permute(1, 2, 0)
        report["notes"].append(f"pos_embed: resampled {g}x{g} -> {vit.grid}x{vit.grid}")
    return grid.reshape(1, vit.grid * vit.grid, d).to(pos.dtype)


def load_pretrained(vit: MultispectralViT, ckpt_path: str, config: Config) -> dict:
    """Initialise the spatial encoder from a pretrained ViT (VIT_aarchi.md section 7).

    Accepts timm-style ViT state dicts, including those wrapped in an MAE/DINO checkpoint
    or saved with ``encoder.``/``module.`` prefixes. Adaptations are explicit and reported:
    3-D tubelet patch kernels, band selection (``vit.pretrained_band_indices``), cls /
    multi-frame positional tables and grid-size resampling. Anything whose shape still
    differs is skipped (and therefore keeps its fresh initialisation) rather than forced.

    A compatible match for this architecture is Prithvi-EO-1.0-100M (ViT-B/16, 768-d,
    12 blocks, 6 HLS bands B02 B03 B04 B8A B11 B12 -- this model's band order, with B8A
    standing in for B08).
    """
    log = get_logger("vit")
    state = _read_state(ckpt_path)
    report = {"loaded": [], "skipped": [], "reasons": {}, "notes": [], "path": ckpt_path}
    target = vit.state_dict()
    has_cls = any(_normalise_key(k).endswith("cls_token") for k in state)
    to_load = {}
    for raw_key, v in state.items():
        if not torch.is_tensor(v):
            continue
        key = _normalise_key(raw_key)
        if any(tok in key for tok in _IGNORE):
            continue
        if key == "patch_embed.weight":
            v = _adapt_patch_embed(v, vit, config, report)
        elif key == "pos_embed":
            v = _adapt_pos_embed(v, vit, config, report, has_cls)
        if v is None or key not in target:
            report["skipped"].append(key)
            report["reasons"].setdefault(key, "absent in model" if v is not None
                                         else "could not be adapted")
            continue
        if tuple(target[key].shape) != tuple(v.shape):
            report["skipped"].append(key)
            report["reasons"][key] = f"shape {tuple(v.shape)} != {tuple(target[key].shape)}"
            continue
        to_load[key] = v.to(target[key].dtype)
        report["loaded"].append(key)

    if "patch_embed.bias" in to_load and "patch_embed.weight" not in to_load:
        # a bias without its kernel is meaningless; keep both fresh together
        to_load.pop("patch_embed.bias")
        report["loaded"].remove("patch_embed.bias")
        report["notes"].append("patch_embed.bias kept fresh because its kernel was skipped")
    missing, _ = vit.load_state_dict(to_load, strict=False)
    report["missing_after_load"] = list(missing)
    n_target = len(target)
    log.info("Pretrained ViT from %s: %d/%d tensors loaded, %d still fresh.", ckpt_path,
             len(report["loaded"]), n_target, len(missing))
    for note in report["notes"]:
        log.info("  %s", note)
    for k, why in report["reasons"].items():
        if k in ("patch_embed.weight", "pos_embed") or not k.startswith("blocks"):
            log.warning("  skipped %s: %s", k, why)
    if len(report["loaded"]) < n_target // 2:
        log.warning("Fewer than half of the ViT tensors loaded -- this is probably not a "
                    "compatible checkpoint; check the report before training on it.")
    return report
