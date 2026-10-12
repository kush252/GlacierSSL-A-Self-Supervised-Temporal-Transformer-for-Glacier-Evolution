"""Phase 4 — applying views (crop geometry + spatial + radiometric aug) to a window.

Given a preprocessed window tensor (T,6,H,W) + validity (T,H,W) and a ViewSpec, produce
the view tensor (T,6,out,out) + per-timestamp crop validity. The three augmentation
families are applied in a fixed, separable order:

    1. crop geometry  (where)  -> same box for all T in the view
    2. spatial aug     (flips/rotations) -> same for all T in the view
    3. radiometric aug (brightness/noise) -> same params for all T in the view

Augmentations are kept physically plausible (mild brightness + light noise, flips/rot90);
they must not erase snowline/terminus/fragmentation evidence (head1 §6, head3 §3).
Padded/invalid pixels never acquire signal: the validity mask is cropped/flipped the same
way, and a view/timestamp pair whose valid fraction falls below the threshold is flagged.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..config import Config
from .sampling import ViewSpec


def _crop_resize(x: torch.Tensor, geom, mode: str) -> torch.Tensor:
    """x: (T,C,H,W). Crop the box then resize to (out_size,out_size)."""
    t, c, h, w = x.shape
    top = min(max(geom.top, 0), h - 1)
    left = min(max(geom.left, 0), w - 1)
    bot = min(top + geom.height, h)
    right = min(left + geom.width, w)
    cropped = x[:, :, top:bot, left:right]
    if cropped.shape[-2:] == (geom.out_size, geom.out_size):
        return cropped
    interp = "nearest" if mode == "mask" else "bilinear"
    kwargs = {} if interp == "nearest" else {"align_corners": False}
    return F.interpolate(cropped, size=(geom.out_size, geom.out_size),
                         mode=interp, **kwargs)


def _apply_spatial(x: torch.Tensor, spatial) -> torch.Tensor:
    if spatial.hflip:
        x = torch.flip(x, dims=[-1])
    if spatial.vflip:
        x = torch.flip(x, dims=[-2])
    if spatial.rot90_k:
        x = torch.rot90(x, k=spatial.rot90_k, dims=[-2, -1])
    return x


def apply_view(window: torch.Tensor, valid: torch.Tensor, view: ViewSpec,
               config: Config) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (view_img (T,6,out,out) float32, crop_valid (T,) bool).

    ``window`` is (T,6,H,W) normalized reflectance; ``valid`` is (T,H,W) bool.
    """
    geom = view.geometry
    img = _crop_resize(window, geom, mode="img")
    mask = valid.unsqueeze(1).float()                       # (T,1,H,W)
    mask = _crop_resize(mask, geom, mode="mask")            # (T,1,out,out)

    img = _apply_spatial(img, view.spatial)
    mask = _apply_spatial(mask, view.spatial)

    # radiometric: multiplicative brightness + additive gaussian noise on reflectance.
    rad = view.radiometric
    if rad.brightness != 1.0:
        img = img * rad.brightness
    if rad.noise_std > 0:
        g = torch.Generator(device=img.device).manual_seed(int(rad.noise_seed))
        noise = torch.randn(img.shape, generator=g, device=img.device) * rad.noise_std
        img = img + noise
    # invalid pixels carry no signal.
    mask_b = mask.squeeze(1) > 0.5                          # (T,out,out)
    img = img * mask_b.unsqueeze(1).float()

    min_frac = float(config.data.min_valid_fraction)
    valid_frac = mask_b.float().mean(dim=(-1, -2))          # (T,)
    crop_valid = valid_frac >= min_frac
    return img.float(), crop_valid
