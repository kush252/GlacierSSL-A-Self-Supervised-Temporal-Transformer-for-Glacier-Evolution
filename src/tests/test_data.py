"""Phase 2-4 tests: manifest/sequence, preprocessing math, and the view/crop pipeline.

Crop-geometry-through-time consistency, cross-view geometry difference, reproducibility
from stored specs, and crop-validity flagging (build plan Phase 4 exit criteria). Uses
synthetic tensors so the tests run without disk IO; a disk-backed smoke test is guarded.
"""
import numpy as np
import torch

from _helpers import small_config
from glacier_ssl.config import load_config
from glacier_ssl.data.sampling import (WindowSampler, elapsed_years_from_dates,
                                        CropGeometry, SpatialAug, RadiometricAug, ViewSpec)
from glacier_ssl.data.augmentation import apply_view
from glacier_ssl.data.manifest import GlacierSequence, Observation
from datetime import date


# ----------------------------------------------------------- elapsed-time from dates
def test_elapsed_years_from_dates_irregular_gaps():
    dates = [date(2016, 1, 1), date(2016, 4, 1), date(2018, 1, 1)]
    e = elapsed_years_from_dates(dates)
    assert abs(e[0]) < 1e-9
    assert abs(e[1] - 0.2491) < 1e-3          # ~91 days
    assert abs(e[2] - 2.0014) < 1e-2          # the 2-year gap stays visible


# ----------------------------------------------------------- window sampler
def _toy_sequence(n=12):
    obs = []
    for k in range(n):
        y = 2016 + k // 4
        s = ["DJF", "MAM", "JJA", "SON"][k % 4]
        obs.append(Observation(glims_id="G1", observation_id=k, year=y, season=s,
                               acquisition_date=f"{y}-{(k%4)*3+1:02d}-15",
                               band_paths={}, scl_path=None, cloud_pct=0.0))
    seq = GlacierSequence("G1", "r1", "train", 1.0, (0.0, 0.0), obs)
    seq.sort_chronologically()
    return seq


def test_window_variable_length_within_bounds():
    cfg = small_config(sampling={"window": {"min_length": 2, "max_length": 8,
                                            "short_max": 3, "medium_max": 6,
                                            "length_bucket_freq": {"short": 0.34,
                                                                   "medium": 0.33,
                                                                   "long": 0.33}}})
    sampler = WindowSampler(cfg)
    seq = _toy_sequence(12)
    lengths = set()
    for seed in range(40):
        s = sampler.sample(seq, seed=seed)
        lengths.add(len(s.obs_indices))
        assert 2 <= len(s.obs_indices) <= 8
        # chronological + dated
        assert s.dates == sorted(s.dates)
    assert len(lengths) > 1                   # variable length actually happens


def test_dino_views_geometry_distinct_and_counts():
    cfg = small_config()
    sampler = WindowSampler(cfg)
    s = sampler.sample(_toy_sequence(10), seed=3)
    assert len(s.dino_views) == (cfg.head1.num_global_student
                                 + cfg.head1.num_local_student)
    geoms = [(v.geometry.top, v.geometry.left, v.geometry.height) for v in s.dino_views]
    assert len(set(geoms)) > 1                # different geometry across views


# ----------------------------------------------------------- crop geometry through time
def test_crop_geometry_constant_through_time():
    """Identical spatial content across timestamps -> identical per-timestamp view output,
    which is only true if the SAME crop box + aug is applied to every timestamp."""
    cfg = small_config()
    T, C, H, W = 5, 6, 128, 128
    base = torch.randn(C, H, W)
    window = base.unsqueeze(0).repeat(T, 1, 1, 1)      # same content each timestamp
    valid = torch.ones(T, H, W, dtype=torch.bool)
    view = ViewSpec(CropGeometry(10, 20, 60, 60, 128, "global"),
                    SpatialAug(hflip=True, rot90_k=1), RadiometricAug(1.1, 0.0, 0),
                    "global")
    img, crop_valid = apply_view(window, valid, view, cfg)
    assert img.shape == (T, C, 128, 128)
    for t in range(1, T):
        assert torch.allclose(img[0], img[t], atol=1e-6)   # same box/aug every timestamp
    assert crop_valid.all()


def test_independent_timestamp_content_shares_one_box():
    """Different content per timestamp must still be cropped by ONE shared box: the output
    at each timestamp equals cropping that timestamp's own image with the same geometry."""
    cfg = small_config()
    T, C, H, W = 3, 6, 128, 128
    window = torch.randn(T, C, H, W)
    valid = torch.ones(T, H, W, dtype=torch.bool)
    geom = CropGeometry(0, 0, 128, 128, 128, "base")       # identity box, no resize
    view = ViewSpec(geom, SpatialAug(), RadiometricAug(), "base")
    img, _ = apply_view(window, valid, view, cfg)
    assert torch.allclose(img, window, atol=1e-6)          # identity crop == input


def test_view_reproducible_from_spec():
    cfg = small_config()
    sampler = WindowSampler(cfg)
    s1 = sampler.sample(_toy_sequence(10), seed=123)
    s2 = sampler.sample(_toy_sequence(10), seed=123)
    g1 = [v.geometry.as_dict() for v in s1.dino_views]
    g2 = [v.geometry.as_dict() for v in s2.dino_views]
    assert g1 == g2                            # deterministic from (seq, seed)
    assert s1.obs_indices == s2.obs_indices


def test_crop_validity_flags_low_coverage():
    cfg = small_config(data={"min_valid_fraction": 0.5})
    T, C, H, W = 2, 6, 128, 128
    window = torch.randn(T, C, H, W)
    valid = torch.zeros(T, H, W, dtype=torch.bool)
    valid[0] = True                            # timestamp 0 fully valid
    valid[1, :10, :10] = True                  # timestamp 1 almost entirely invalid
    view = ViewSpec(CropGeometry(0, 0, 128, 128, 128, "base"),
                    SpatialAug(), RadiometricAug(), "base")
    _, crop_valid = apply_view(window, valid, view, cfg)
    assert bool(crop_valid[0]) and not bool(crop_valid[1])


def test_head2_pairs_and_head3_intervals_direction():
    cfg = small_config()
    sampler = WindowSampler(cfg)
    s = sampler.sample(_toy_sequence(12), seed=9)
    rng = np.random.default_rng(0)
    pairs = sampler.sample_distance_pairs(s, 4, rng)
    for (i, j) in pairs:
        assert i != j
    intervals = sampler.sample_change_intervals(s, 4, rng)
    for iv in intervals:
        assert iv["i"] < iv["j"]               # directed, d_i < d_j
        assert iv["gap_years"] > 0
