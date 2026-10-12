"""Phase 3 — Preprocessing and quality control.

Deterministic, leakage-free image preprocessing matching the ViT's input contract
(VIT_aarchi.md §2):
* six-band order B02,B03,B04,B08,B11,B12 enforced; SCL is quality info, not a channel;
* reflectance scaling (uint16 DN / scale);
* validity masking from SCL-derived quality + nodata, per the heads' mask references;
* band-alignment / extent / dimension checks before stacking;
* band-wise normalization using TRAIN-ONLY statistics (fit elsewhere, applied here);
  never per-image normalization (it would erase between-date reflectance differences).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import rasterio
except Exception:  # pragma: no cover - rasterio always present in this env
    rasterio = None

from ..config import Config
from ..utils.logging_utils import get_logger
from .manifest import Observation


class BandAlignmentError(ValueError):
    """Raised when bands of one timestamp do not share grid/extent/size."""


def _bounds_from(transform, shape) -> tuple[float, float, float, float]:
    h, w = shape
    x0, y0 = transform.c, transform.f
    x1 = x0 + transform.a * w + transform.b * h
    y1 = y0 + transform.d * w + transform.e * h
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


class Preprocessor:
    """Loads one observation into a normalized (6,H,W) tensor + validity mask."""

    def __init__(self, config: Config, norm_stats: Optional[dict] = None,
                 project_root: Optional[Path] = None):
        self.cfg = config
        self.bands = list(config.data.band_order)
        self.scale = float(config.data.reflectance_scale)
        self.invalid_scl = set(config.data.scl_invalid_classes)
        self.image_size = int(config.data.image_size)
        self.project_root = project_root or Path(__file__).resolve().parents[3]
        self.norm_stats = norm_stats  # dict band -> {mean,std} in reflectance units
        # Alignment tolerance: bands are each warped to 512x512 over the glacier bbox;
        # 10m vs resampled-20m grids co-register to within ~1 pixel. Require identical
        # shape+CRS always; allow sub-tolerance origin/extent drift unless strict.
        self.strict_alignment = bool(config.get("data.strict_alignment", False))
        self.alignment_tol_m = float(config.get("data.alignment_tolerance_m", 30.0))
        self.log = get_logger("preprocessing")
        self._warned_align = False

    def _resolve(self, rel: str) -> Path:
        return self.project_root / rel.replace("\\", "/")

    # -- raw IO ------------------------------------------------------------------
    def _read_band(self, path: Path):
        with rasterio.open(path) as ds:
            arr = ds.read(1).astype(np.float32)
            meta = {"shape": arr.shape, "transform": ds.transform,
                    "crs": str(ds.crs), "nodata": ds.nodata}
        return arr, meta

    def load_raw(self, obs: Observation) -> tuple[np.ndarray, np.ndarray]:
        """Return (bands_reflectance (6,H,W) float32, scl (H,W) or None).

        Performs band-alignment checks before stacking.
        """
        arrays, metas = [], []
        for band in self.bands:
            path = self._resolve(obs.band_paths[band])
            arr, meta = self._read_band(path)
            arrays.append(arr)
            metas.append(meta)

        ref = metas[0]
        ref_bounds = _bounds_from(ref["transform"], ref["shape"])
        tag = f"{obs.glims_id} {obs.year}_{obs.season}"
        for band, meta in zip(self.bands, metas):
            # Identical pixel dimensions and CRS are always required.
            if meta["shape"] != ref["shape"]:
                raise BandAlignmentError(
                    f"{tag}: band {band} shape {meta['shape']} != {ref['shape']}")
            if meta["crs"] != ref["crs"]:
                raise BandAlignmentError(f"{tag}: band {band} CRS mismatch")
            # Extent/origin: 10m vs resampled-20m grids may differ by ~1 pixel. Enforce
            # exact equality only in strict mode; otherwise require sub-tolerance drift.
            bounds = _bounds_from(meta["transform"], meta["shape"])
            drift = max(abs(a - b) for a, b in zip(bounds, ref_bounds))
            if drift > self.alignment_tol_m:
                raise BandAlignmentError(
                    f"{tag}: band {band} extent drift {drift:.1f}m "
                    f"> tol {self.alignment_tol_m}m")
            if self.strict_alignment and meta["transform"] != ref["transform"]:
                raise BandAlignmentError(f"{tag}: band {band} transform mismatch (strict)")
            if drift > 0 and not self._warned_align:
                self.log.info(
                    "Band extent drift up to %.1fm (<= tol %.1fm) accepted; bands "
                    "co-registered on identical %sx%s grids.", drift,
                    self.alignment_tol_m, ref["shape"][0], ref["shape"][1])
                self._warned_align = True

        stack = np.stack(arrays, axis=0)  # (6,H,W) raw DN
        scl = None
        if obs.scl_path:
            spath = self._resolve(obs.scl_path)
            if spath.exists():
                scl_arr, scl_meta = self._read_band(spath)
                if scl_meta["shape"] == ref["shape"]:
                    scl = scl_arr.astype(np.int16)
        return stack, scl

    # -- validity ----------------------------------------------------------------
    def validity_mask(self, raw_bands: np.ndarray, scl: Optional[np.ndarray]) -> np.ndarray:
        """Per-pixel validity (H,W) bool, True = usable.

        Invalid where SCL class is in the configured invalid set, or where all bands
        are nodata(0). SCL class IDs are never used as reflectance values.
        """
        h, w = raw_bands.shape[1:]
        valid = np.ones((h, w), dtype=bool)
        # nodata: Sentinel-2 L2A uses 0 as nodata across reflectance bands.
        nodata = np.all(raw_bands <= 0, axis=0)
        valid &= ~nodata
        if scl is not None:
            invalid = np.isin(scl, list(self.invalid_scl))
            valid &= ~invalid
        return valid

    # -- normalization -----------------------------------------------------------
    def to_reflectance(self, raw_bands: np.ndarray) -> np.ndarray:
        return (raw_bands / self.scale).astype(np.float32)

    def normalize(self, reflectance: np.ndarray) -> np.ndarray:
        """Band-wise (x-mean)/std using persisted TRAIN stats. No per-image stats."""
        if self.norm_stats is None:
            return reflectance
        out = np.empty_like(reflectance)
        for i, band in enumerate(self.bands):
            st = self.norm_stats[band]
            std = st["std"] if st["std"] > 1e-6 else 1.0
            out[i] = (reflectance[i] - st["mean"]) / std
        return out

    def process(self, obs: Observation) -> tuple[np.ndarray, np.ndarray]:
        """Full path: load -> reflectance -> validity -> zero invalid -> normalize.

        Returns (normalized (6,H,W) float32, validity (H,W) bool). Invalid pixels are
        zeroed in reflectance space *before* normalization so they carry no reflectance
        signal; the mask records them for downstream exclusion.
        """
        raw, scl = self.load_raw(obs)
        valid = self.validity_mask(raw, scl)
        refl = self.to_reflectance(raw)
        refl[:, ~valid] = 0.0
        norm = self.normalize(refl)
        return norm, valid
