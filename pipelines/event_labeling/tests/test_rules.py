"""Synthetic checks of the labelling rules (no network)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glacier_gt.config import load_config  # noqa: E402
from glacier_gt.labels import change_test, detect_splits, label_window, sample_windows  # noqa: E402
from glacier_gt.measure import Measurement, components  # noqa: E402

CFG = load_config()
PX = 10.0


def body_one():
    m = np.zeros((60, 120), bool)
    m[10:50, 10:110] = True          # 0.4 km2 single body
    return m


def body_split(gap=6):
    m = body_one()
    m[:, 58:58 + gap] = False        # 60 m gap -> two bodies
    return m


def comps(masks):
    out = [components(m, PX, CFG) for m in masks]
    return [c for c, _ in out], [a for _, a in out]


def ep(year, area, status="ok", sigma=None):
    return Measurement(item_id=f"i{year}", date=f"{year}-09-01", year=year, source="t", baseline=0,
                       area_km2=area, sigma_km2=sigma if sigma else 0.02 * area, area_ndsi_km2=area,
                       sigma_ndsi_km2=sigma if sigma else 0.02 * area, status=status, outline_coverage=0.9,
                       mask_iou=0.95, beyond_outline_km2=0.0)


def test_narrow_gap_does_not_split():
    c, a = components(body_split(gap=1), PX, CFG)
    assert len(a) == 1                   # 10 m gap closed (closing_px=1)


def test_persistent_split_confirmed_and_pre_verified():
    maps, areas = comps([body_one(), body_one(), body_split(), body_split(), body_split()])
    ev = detect_splits([2016, 2017, 2018, 2019, 2020], maps, areas, CFG)
    assert len(ev) == 1 and ev[0]["year"] == 2018
    assert ev[0]["confirmed"] and ev[0]["pre_verified"]


def test_transient_split_not_confirmed():
    maps, areas = comps([body_one(), body_one(), body_split(), body_one(), body_one()])
    ev = detect_splits([2016, 2017, 2018, 2019, 2020], maps, areas, CFG)
    assert ev and not ev[0]["confirmed"]          # re-merged next year (snow bridge / noise)


def test_pre_existing_tongues_are_not_a_split():
    maps, areas = comps([body_split()] * 4)
    assert detect_splits([2016, 2017, 2018, 2019], maps, areas, CFG) == []


def test_change_test_lengths():
    s = [0.01] * 5
    r = change_test([0, 1, 2, 3, 4], [1.0, 0.97, 0.93, 0.9, 0.86], s, CFG)
    assert r["significant"] and r["direction"] == -1 and r["mk_p"] < 0.05
    r = change_test([0, 1], [1.0, 0.99], [0.01, 0.01], CFG)
    assert not r["significant"]                  # 0.01 < 2 * 0.014
    r = change_test([0, 1, 2], [1.0, 1.05, 0.9], [0.01] * 3, CFG)
    assert not r["significant"]                  # not monotonic


def test_window_retreat_and_stable_and_insufficient():
    eps = [ep(2016 + i, 1.0 - 0.04 * i) for i in range(9)]
    r = label_window(eps, 2016, 2024, [], [], CFG)
    assert r["primary_label"] == "Retreat" and r["auto_tier"] in ("A", "B")
    eps = [ep(2016 + i, 1.0 + (0.003 if i % 2 else -0.003)) for i in range(9)]
    assert label_window(eps, 2016, 2024, [], [], CFG)["primary_label"] == "Stable"
    eps = [ep(2016, 1.0), ep(2017, 0.9, status="seasonal_snow")]
    assert label_window(eps, 2016, 2017, [], [], CFG)["primary_label"] == "Insufficient"


def test_disappearance_needs_persistence():
    eps = [ep(2016, 0.05), ep(2017, 0.03), ep(2018, 0.004), ep(2019, 0.002)]
    r = label_window(eps, 2016, 2019, [], [], CFG)
    assert r["primary_label"] == "Disappearance"
    eps = [ep(2016, 0.05), ep(2017, 0.03), ep(2018, 0.004), ep(2019, 0.03)]
    assert label_window(eps, 2016, 2019, [], [], CFG)["primary_label"] != "Disappearance"
    # comes back later (look-ahead beyond the window) -> not gone for good
    eps = [ep(2016, 0.05), ep(2017, 0.004), ep(2018, 0.004), ep(2019, 0.04)]
    assert label_window(eps, 2016, 2018, [], [], CFG)["primary_label"] != "Disappearance"
    # starts at the detection limit -> never a disappearance event
    eps = [ep(2016, 0.015), ep(2017, 0.004), ep(2018, 0.003)]
    r = label_window(eps, 2016, 2018, [], [], CFG)
    assert r["primary_label"] != "Disappearance" and "near_mmu_at_start" in r["flags"]


def test_windows_sampler():
    w = sample_windows(list(range(2016, 2025)), 4, np.random.default_rng(0))
    assert w[0] == (2016, 2024) and len(set(w)) == len(w) == 5
    assert all(b - a >= 1 for a, b in w)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def test_marginal_patch_is_not_fragmentation():
    split_small = body_one()
    split_small[10:50, 95:97] = False             # 0.02 km2 strip detaches at the margin
    maps, areas = comps([body_one(), body_one(), split_small, split_small])
    assert detect_splits([2016, 2017, 2018, 2019], maps, areas, CFG) == []


def test_terminus_alone_gives_retreat_but_not_tier_a():
    eps = [ep(2016 + i, 1.0) for i in range(6)]
    for i, e in enumerate(eps):
        e.z_terminus_m = 2000 + 15 * i            # +75 m terminus rise, area flat
    r = label_window(eps, 2016, 2021, [], [], CFG)
    assert r["primary_label"] == "Retreat" and r["agree"] == ["terminus"] and r["auto_tier"] == "B"
