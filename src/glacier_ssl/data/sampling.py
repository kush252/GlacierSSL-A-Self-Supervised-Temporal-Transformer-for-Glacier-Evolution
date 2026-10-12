"""Phase 4 — window sampling + view/crop geometry + metadata tracking.

This module produces the *specifications* for training examples before any backbone runs.
It enforces the core rules from head1_archi.md §2-§3 and head3_archi.md §8, and the review
brief's three-way split of augmentation (build plan Phase 4):

* Variable-length window sampler over a glacier's chronological observations (min/max
  length and short/medium/long frequency are config, never hardcoded).
* A *view* owns ONE crop geometry applied to every timestamp in the window (same geographic
  area through time); different views get different geometries. Never one identical crop for
  all views; never independent per-timestamp crops within a view.
* Three distinct, separately tracked things: crop geometry (where), spatial augmentation
  (flips/rotations), radiometric augmentation (reflectance/noise).
* Everything needed to reproduce a view (dates, original indices, crop coords, aug params,
  glacier id, seed) is captured in the spec so examples are reproducible from metadata.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Optional

import numpy as np

from ..config import Config
from .manifest import GlacierSequence

_DAYS_PER_YEAR = 365.2425


@dataclass
class CropGeometry:
    """A crop box in source-pixel coordinates, resized to ``out_size``.

    The SAME geometry is applied to every timestamp within its view.
    """
    top: int
    left: int
    height: int
    width: int
    out_size: int
    kind: str  # "global" | "local" | "base"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SpatialAug:
    hflip: bool = False
    vflip: bool = False
    rot90_k: int = 0  # number of 90-degree rotations

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class RadiometricAug:
    brightness: float = 1.0        # multiplicative factor
    noise_std: float = 0.0         # additive gaussian std (reflectance units)
    noise_seed: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ViewSpec:
    """A full, reproducible view: geometry + spatial aug + radiometric aug."""
    geometry: CropGeometry
    spatial: SpatialAug
    radiometric: RadiometricAug
    view_type: str  # "global" | "local" | "base"

    def as_dict(self) -> dict:
        return {"geometry": self.geometry.as_dict(), "spatial": self.spatial.as_dict(),
                "radiometric": self.radiometric.as_dict(), "view_type": self.view_type}


@dataclass
class WindowSample:
    """A sampled chronological window + all derived specs and metadata."""
    glims_id: str
    split: str
    obs_indices: list[int]           # indices into the glacier's chronological list
    observation_ids: list[int]       # stable DB ids (reproducibility)
    dates: list[str]                 # ISO acquisition dates (temporal truth)
    seasons: list[str]
    elapsed_years: list[float]       # from first obs in window, from dates
    span_years: float
    seed: int
    dino_views: list[ViewSpec] = field(default_factory=list)
    base_view: Optional[ViewSpec] = None
    # Head 2 pairs: index pairs into the window (computed on base view embeddings)
    distance_pairs: list[tuple[int, int]] = field(default_factory=list)
    # Head 3 intervals: (i,j) directed with d_i<d_j, each gets two views
    change_intervals: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["dino_views"] = [v.as_dict() for v in self.dino_views]
        d["base_view"] = self.base_view.as_dict() if self.base_view else None
        return d


def elapsed_years_from_dates(dates: list[date]) -> list[float]:
    base = dates[0]
    return [(d - base).days / _DAYS_PER_YEAR for d in dates]


class WindowSampler:
    """Samples variable-length chronological windows and all per-head view specs."""

    def __init__(self, config: Config):
        self.cfg = config
        w = config.sampling.window
        self.min_len = int(w.min_length)
        self.max_len = int(w.max_length)
        self.short_max = int(w.short_max)
        self.medium_max = int(w.medium_max)
        freq = w.length_bucket_freq
        self.bucket_freq = (float(freq["short"]), float(freq["medium"]), float(freq["long"]))
        s = config.sampling
        self.image_size = int(config.data.image_size)
        self.global_scale = tuple(s.global_scale)
        self.local_scale = tuple(s.local_scale)
        self.global_crop_size = int(s.global_crop_size)
        self.local_crop_size = int(s.local_crop_size)
        self.n_global = int(config.head1.num_global_student)   # distinct global views built
        self.n_local = int(config.head1.num_local_student)
        self.aug_cfg = s.augmentation

    # -- window length -----------------------------------------------------------
    def _sample_length(self, rng: np.random.Generator, available: int) -> int:
        lo = min(self.min_len, available)
        hi = min(self.max_len, available)
        if hi <= lo:
            return hi
        # Pick a bucket, then a length within it (dates make the actual span irregular).
        buckets = []
        short_hi = min(self.short_max, hi)
        if short_hi >= lo:
            buckets.append(("short", lo, short_hi, self.bucket_freq[0]))
        med_lo = max(lo, self.short_max + 1)
        med_hi = min(self.medium_max, hi)
        if med_hi >= med_lo:
            buckets.append(("medium", med_lo, med_hi, self.bucket_freq[1]))
        long_lo = max(lo, self.medium_max + 1)
        if hi >= long_lo:
            buckets.append(("long", long_lo, hi, self.bucket_freq[2]))
        if not buckets:
            return rng.integers(lo, hi + 1)
        probs = np.array([b[3] for b in buckets], dtype=float)
        probs /= probs.sum()
        choice = buckets[rng.choice(len(buckets), p=probs)]
        return int(rng.integers(choice[1], choice[2] + 1))

    def _sample_window_indices(self, rng: np.random.Generator, n_obs: int) -> list[int]:
        length = self._sample_length(rng, n_obs)
        if length >= n_obs:
            return list(range(n_obs))
        start = int(rng.integers(0, n_obs - length + 1))
        return list(range(start, start + length))

    # -- geometry ----------------------------------------------------------------
    def _sample_geometry(self, rng: np.random.Generator, src: int,
                         scale_range: tuple[float, float], out_size: int,
                         kind: str) -> CropGeometry:
        scale = rng.uniform(scale_range[0], scale_range[1])
        side = int(round(src * math.sqrt(scale)))
        side = max(8, min(side, src))
        top = int(rng.integers(0, src - side + 1)) if side < src else 0
        left = int(rng.integers(0, src - side + 1)) if side < src else 0
        return CropGeometry(top=top, left=left, height=side, width=side,
                            out_size=out_size, kind=kind)

    def _sample_spatial(self, rng: np.random.Generator) -> SpatialAug:
        a = self.aug_cfg
        if not bool(a.enabled):
            return SpatialAug()
        return SpatialAug(
            hflip=bool(rng.random() < float(a.hflip_prob)),
            vflip=bool(rng.random() < float(a.vflip_prob)),
            rot90_k=int(rng.integers(0, 4)) if rng.random() < float(a.rot90_prob) else 0,
        )

    def _sample_radiometric(self, rng: np.random.Generator) -> RadiometricAug:
        a = self.aug_cfg
        if not bool(a.enabled):
            return RadiometricAug()
        r = a.radiometric
        b = float(r.brightness)
        return RadiometricAug(
            brightness=float(rng.uniform(1.0 - b, 1.0 + b)),
            noise_std=float(r.gaussian_noise_std),
            noise_seed=int(rng.integers(0, 2 ** 31 - 1)),
        )

    def _make_view(self, rng, src, scale_range, out_size, kind) -> ViewSpec:
        return ViewSpec(
            geometry=self._sample_geometry(rng, src, scale_range, out_size, kind),
            spatial=self._sample_spatial(rng),
            radiometric=self._sample_radiometric(rng),
            view_type=kind,
        )

    # -- top-level ---------------------------------------------------------------
    def sample(self, seq: GlacierSequence, seed: int) -> WindowSample:
        rng = np.random.default_rng(seed)
        n_obs = len(seq.observations)
        idxs = self._sample_window_indices(rng, n_obs)
        obs = [seq.observations[i] for i in idxs]
        dates = [o.date_obj() for o in obs]
        elapsed = elapsed_years_from_dates(dates)
        span = elapsed[-1] - elapsed[0] if len(elapsed) > 1 else 0.0
        src = self.image_size

        # Base view = identity crop (full scene) resized to image size. Head 2 and the
        # canonical sequence use this.
        base = ViewSpec(
            geometry=CropGeometry(0, 0, src, src, src, "base"),
            spatial=SpatialAug(), radiometric=RadiometricAug(), view_type="base")

        # DINO views: distinct global + local geometries (different across views).
        dino_views = [self._make_view(rng, src, self.global_scale,
                                      self.global_crop_size, "global")
                      for _ in range(self.n_global)]
        dino_views += [self._make_view(rng, src, self.local_scale,
                                       self.local_crop_size, "local")
                       for _ in range(self.n_local)]

        sample = WindowSample(
            glims_id=seq.glims_id, split=seq.split, obs_indices=idxs,
            observation_ids=[o.observation_id for o in obs],
            dates=[d.isoformat() for d in dates],
            seasons=[o.season for o in obs],
            elapsed_years=elapsed, span_years=span, seed=seed,
            dino_views=dino_views, base_view=base,
        )
        return sample

    # -- Head 2 pair sampling (gap-stratified over window dates) -----------------
    def sample_distance_pairs(self, sample: WindowSample, n_pairs: int,
                              rng: np.random.Generator) -> list[tuple[int, int]]:
        T = len(sample.obs_indices)
        if T < 2:
            return []
        edges = [float(e) for e in self.cfg.head2.gap_bins_years]
        strategy = str(self.cfg.head2.pair_sampling)
        all_pairs = [(i, j) for i in range(T) for j in range(i + 1, T)]
        gaps = {(i, j): abs(sample.elapsed_years[j] - sample.elapsed_years[i])
                for (i, j) in all_pairs}
        # remove duplicate-date pairs
        all_pairs = [p for p in all_pairs if gaps[p] > 1e-9]
        if not all_pairs:
            return []
        if strategy == "uniform":
            sel = [all_pairs[k] for k in rng.choice(len(all_pairs),
                   size=min(n_pairs, len(all_pairs)), replace=False)]
            return sel
        # gap-stratified: bucket pairs by gap bin, draw round-robin across non-empty bins
        bins: dict[int, list] = {}
        for p in all_pairs:
            b = np.digitize(gaps[p], edges)
            bins.setdefault(int(b), []).append(p)
        order = [b for b in sorted(bins) if bins[b]]
        sel = []
        bi = 0
        while len(sel) < n_pairs and order:
            b = order[bi % len(order)]
            lst = bins[b]
            sel.append(lst[int(rng.integers(0, len(lst)))])
            bi += 1
        return sel[:n_pairs]

    # -- Head 3 interval sampling (directed, gap-stratified, two views each) -----
    def sample_change_intervals(self, sample: WindowSample, n_intervals: int,
                                rng: np.random.Generator) -> list[dict]:
        T = len(sample.obs_indices)
        if T < 2:
            return []
        max_gap = self.cfg.head3.max_gap_years
        edges = [float(e) for e in self.cfg.head2.gap_bins_years]
        src = self.image_size
        pairs = []
        for i in range(T):
            for j in range(i + 1, T):  # ensures date_i < date_j (chronological order)
                gap = sample.elapsed_years[j] - sample.elapsed_years[i]
                if gap <= 1e-9:
                    continue
                if max_gap is not None and gap > float(max_gap):
                    continue
                pairs.append((i, j, gap))
        if not pairs:
            return []
        # gap-stratified selection
        bins: dict[int, list] = {}
        for p in pairs:
            b = int(np.digitize(p[2], edges))
            bins.setdefault(b, []).append(p)
        order = [b for b in sorted(bins) if bins[b]]
        intervals = []
        bi = 0
        while len(intervals) < n_intervals and order:
            b = order[bi % len(order)]
            lst = bins[b]
            i, j, gap = lst[int(rng.integers(0, len(lst)))]
            # two views of the SAME directed interval, different crop geometries,
            # each applied consistently to both endpoints.
            view_a = self._make_view(rng, src, self.global_scale, self.global_crop_size, "global")
            view_b = self._make_view(rng, src, self.global_scale, self.global_crop_size, "global")
            intervals.append({"i": i, "j": j, "gap_years": gap,
                              "view_a": view_a, "view_b": view_b})
            bi += 1
        return intervals
