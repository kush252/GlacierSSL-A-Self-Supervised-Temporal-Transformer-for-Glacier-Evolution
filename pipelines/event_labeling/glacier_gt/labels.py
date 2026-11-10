"""Window labels for sequences of any length, derived only from the per-epoch measurements.

A window is a run of annual end-of-season epochs [y0, y1]. Its label is decided by
independent *evidence lines*; each votes -1/0/+1 (or event/no-event) with its own
significance test. The label's tier says how much independent evidence agrees.

Lines:
  area_ratio   ice-body area, red/SWIR mask        (primary)
  area_ndsi    ice-body area, NDSI mask            (different spectral algorithm)
  terminus     2nd-percentile ice elevation (DEM)  (geometry, insensitive to upper-glacier snow)
  glims        independent GLIMS outlines dated inside the window (other analysts)
  split_ratio / split_ndsi / glims_children        (fragmentation)
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np
from scipy.stats import kendalltau, theilslopes

from . import PRECEDENCE


def decimal_year(d: str) -> float:
    y, m, dd = (int(x) for x in d[:10].split("-"))
    return y + (date(y, m, dd).timetuple().tm_yday - 1) / 365.25


# ----------------------------------------------------------------------------- trend test
def change_test(t, a, s, cfg) -> dict | None:
    """Significance of change in a series with per-point 1-sigma uncertainty s.

    n >= 4: Theil-Sen change over the window + Mann-Kendall p < alpha + |change| > k*sigma.
    n = 3 : endpoint change > k*sigma and monotonic.
    n = 2 : endpoint change > k*sigma (low power; capped at tier B).
    """
    t, a, s = (np.asarray(x, float) for x in (t, a, s))
    n = len(t)
    if n < 2:
        return None
    span = t[-1] - t[0]
    if n >= 4:
        ts = theilslopes(a, t)
        slope = float(ts.slope)
        delta = slope * span
        tau, p = kendalltau(t, a)
        p = float(p) if np.isfinite(p) else 1.0
        trend_ok = p < cfg.uncertainty.mk_alpha
    else:
        delta = float(a[-1] - a[0])
        slope = delta / span if span > 0 else 0.0
        p = math.nan
        d = np.sign(np.diff(a))
        trend_ok = bool(np.all(d == np.sign(delta))) if n == 3 else True
    sigma = float(np.hypot(s[0], s[-1]))
    snr = abs(delta) / sigma if sigma > 0 else math.inf
    sig = bool(trend_ok and snr > cfg.uncertainty.k_sigma)
    return {
        "n": n, "delta": round(delta, 5), "sigma": round(sigma, 5), "snr": round(snr, 2),
        "slope_per_yr": round(slope, 5), "mk_p": None if math.isnan(p) else round(p, 4),
        "rel_change": round(delta / a[0], 4) if a[0] > 0 else None,
        "direction": int(np.sign(delta)) if sig else 0, "significant": sig,
    }


# ----------------------------------------------------------------------------- lineage
def _overlaps(prev_lab, cur_lab, min_px):
    both = (prev_lab > 0) & (cur_lab > 0)
    if not both.any():
        return {}
    pairs, counts = np.unique(np.stack([prev_lab[both], cur_lab[both]], 1), axis=0, return_counts=True)
    out = {}
    for (p, c), n in zip(pairs, counts):
        if n >= min_px:
            out.setdefault(int(p), set()).add(int(c))
    return out


def detect_splits(years, comp_maps, comp_areas, cfg) -> list[dict]:
    """Split events along a sequence of valid epochs (component label maps, aligned grid).

    A split at epoch k: one significant component at k-1 overlaps >= 2 significant components
    at k. It is *confirmed* if the descendants stay >= 2 and never re-merge for
    `persistence_epochs` consecutive valid epochs (k included), and *pre-verified* if the parent
    region was one connected body in the epoch(s) before (rules out a transient snow bridge).
    """
    P = cfg.events.persistence_epochs
    mpx = cfg.geometry.overlap_min_px
    events = []
    for k in range(1, len(years)):
        ov = _overlaps(comp_maps[k - 1], comp_maps[k], mpx)
        for parent, children in ov.items():
            parent_area = comp_areas[k - 1][parent - 1]
            frag_min = max(cfg.events.min_fragment_km2, cfg.events.min_fragment_frac * parent_area)
            sig = sorted((c for c in children if comp_areas[k][c - 1] >= frag_min),
                         key=lambda c: -comp_areas[k][c - 1])
            if len(sig) < 2:
                continue                      # only marginal patches broke off: not fragmentation
            # --- forward: follow each part separately; separation holds while >= 2 parts survive
            #     and no later component overlaps two different parts (re-merge)
            parts, held = [{c} for c in sig], 1
            for j in range(k + 1, len(years)):
                if held >= P:
                    break
                nxt = _overlaps(comp_maps[j - 1], comp_maps[j], mpx)
                new_parts = [set().union(*(nxt.get(d, set()) for d in part)) for part in parts]
                new_parts = [q for q in new_parts if q]
                merged = any(new_parts[i] & new_parts[m] for i in range(len(new_parts))
                             for m in range(i + 1, len(new_parts)))
                if merged or len(new_parts) < 2:
                    break
                parts, held = new_parts, held + 1
            # --- backward: the parent region was one connected body before the split
            region = comp_maps[k - 1] == parent
            pre_ok, pre_checked = True, 0
            for j in range(k - 2, max(-1, k - 1 - (P - 1)) - 1, -1):
                ids, cnt = np.unique(comp_maps[j][region], return_counts=True)
                n_bodies = int(((ids > 0) & (cnt >= mpx)).sum())
                pre_checked += 1
                if n_bodies >= 2:
                    pre_ok = False
            events.append({
                "year": int(years[k]), "parent_year": int(years[k - 1]),
                "parent_area_km2": round(parent_area, 4),
                "children_areas_km2": [round(comp_areas[k][c - 1], 4) for c in sig],
                "held_epochs": held, "confirmed": held >= P,
                "pre_verified": bool(pre_ok and pre_checked >= P - 1),
            })
    return events


# ----------------------------------------------------------------------------- external
def glims_change(glims_df, y0, y1, cfg):
    """Area change from independent GLIMS outlines dated within [y0-1, y1+1] (5 % each, Paul 2013)."""
    if glims_df is None or len(glims_df) == 0:
        return None
    df = glims_df.assign(yr=glims_df["src_date"].str[:4].astype(int))
    df = df[(df.yr >= y0 - 1) & (df.yr <= y1 + 1)]
    if df.empty:
        return None
    area = df["area"].where(df["area"] > 0, df.get("db_area", df["area"]))
    per = df.assign(a=area).groupby(["src_date", "anlys_id"])["a"].sum().groupby("src_date").median().sort_index()
    per = per[per > 0]
    if len(per) < 2:
        return None
    t = [decimal_year(d) for d in per.index]
    if t[-1] - t[0] < 2:
        return None
    a = per.values
    s = 0.05 * a
    r = change_test([t[0], t[-1]], [a[0], a[-1]], [s[0], s[-1]], cfg)
    r["dates"] = [per.index[0][:10], per.index[-1][:10]]
    r["areas_km2"] = [round(float(a[0]), 4), round(float(a[-1]), 4)]
    return r


def glims_children_in(children_df, y0, y1):
    if children_df is None or len(children_df) == 0:
        return []
    # parent_id also encodes ordinary sub-units, so only children that FIRST appear inside the
    # window (no earlier outline) count as fragmentation evidence.
    first = children_df.groupby("glac_id")["src_date"].min().str[:4].astype(int)
    return sorted(first[(first >= y0) & (first <= y1 + 1)].index)


# ----------------------------------------------------------------------------- window label
def label_window(epochs, y0, y1, splits_ratio, splits_ndsi, cfg, glims_df=None, children_df=None) -> dict:
    """epochs: list of Measurement (one per year, whole sequence). Window = years y0..y1."""
    win = [e for e in epochs if y0 <= e.year <= y1]
    ok = [e for e in win if e.status == "ok" and np.isfinite(e.area_km2)]
    after = [e for e in epochs if e.year > y1 and e.status == "ok"]
    rec = {
        "start_year": y0, "end_year": y1, "length_years": y1 - y0 + 1,
        "n_epochs": len(win), "n_valid": len(ok), "valid_years": [e.year for e in ok],
        "excluded": {str(e.year): e.status for e in win if e.status != "ok"},
        "evidence": {}, "attributes": {}, "flags": [],
    }
    if len(ok) < 2:
        rec.update(primary_label="Insufficient", auto_tier="-", agree=[], against=[])
        return rec

    t = [decimal_year(e.date) for e in ok]
    ev = rec["evidence"]
    ev["area_ratio"] = change_test(t, [e.area_km2 for e in ok], [e.sigma_km2 for e in ok], cfg)
    ev["area_ndsi"] = change_test(t, [e.area_ndsi_km2 for e in ok], [e.sigma_ndsi_km2 for e in ok], cfg)
    z = [(ti, e.z_terminus_m) for ti, e in zip(t, ok) if np.isfinite(e.z_terminus_m)]
    if len(z) >= 2:
        zt, zz = zip(*z)
        dz_sigma = cfg.uncertainty.terminus_dz_m / cfg.uncertainty.k_sigma / math.sqrt(2)
        term = change_test(zt, zz, [dz_sigma] * len(zz), cfg)
        term["direction"] = -term["direction"]       # terminus rising = retreat (-1)
        ev["terminus"] = term
    ev["glims"] = glims_change(glims_df, y0, y1, cfg)

    # ------------------------------------------------------------- events
    mmu = cfg.geometry.mmu_km2
    P = cfg.events.persistence_epochs
    first_valid = ok[0].year
    in_win = lambda s: first_valid < s["year"] <= y1 and s["parent_year"] >= first_valid
    frag_r = [s for s in splits_ratio if in_win(s)]
    frag_n = [s for s in splits_ndsi if in_win(s)]
    # NB: split persistence is checked on the whole sequence (look-ahead); the split itself
    # must happen inside the window.
    ev["splits_ratio"] = frag_r
    ev["splits_ndsi"] = frag_n
    ev["glims_children"] = glims_children_in(children_df, y0, y1)
    frag_confirmed = [s for s in frag_r if s["confirmed"]]

    seq = ok + (after if cfg.events.allow_lookahead else [])
    disapp = None
    # "gone for good": clearly above MMU at the start, then below MMU in EVERY later valid epoch
    # (>= persistence_epochs of them), so flicker around the detection limit is not an event.
    if ok[0].area_km2 >= 2 * mmu:
        for i, e in enumerate(seq):
            if e.year > y1:
                break
            rest = seq[i:]
            if len(rest) >= P and all(x.area_km2 < mmu for x in rest):
                disapp = {"year": e.year, "confirmed_by": [x.year for x in rest]}
                break
    elif ok[0].area_km2 < 2 * mmu:
        rec["flags"].append("near_mmu_at_start")
    ev["disappearance"] = disapp

    # ------------------------------------------------------------- attributes
    ar, an, te, gl = ev["area_ratio"], ev["area_ndsi"], ev.get("terminus"), ev["glims"]
    # Independent evidence FAMILIES (the two spectral masks share snow/cloud errors, so they form
    # one family: the red/SWIR mask votes, the NDSI mask must merely be consistent with it).
    fam = {"area": ar, "terminus": te, "glims": gl}
    # NDSI mask must not contradict the primary mask: same sign; a same-sign near-miss (SNR >= 1)
    # only blocks tier A, an opposite-sign significant change counts against.
    area_consistent = an is None or an["direction"] == ar["direction"]
    area_opposed = an is not None and an["significant"] and an["direction"] == -ar["direction"] != 0
    dirs = [l["direction"] for l in fam.values() if l and l["significant"]]
    pos, neg = dirs.count(1), dirs.count(-1)
    change_dir = 0 if not dirs else (-1 if not pos else (1 if not neg else None))  # None = conflict
    attrs = rec["attributes"]
    attrs["Retreat"] = change_dir == -1
    attrs["Advance"] = change_dir == 1
    attrs["Fragmentation"] = bool(frag_confirmed)
    attrs["Disappearance"] = disapp is not None
    jumps = np.diff([e.area_km2 for e in ok])
    sig_pair = [np.hypot(a.sigma_km2, b.sigma_km2) for a, b in zip(ok[:-1], ok[1:])]
    attrs["possible_surge"] = bool(attrs["Advance"] and any(
        j > cfg.events.surge_jump_rel * a.area_km2 and j > cfg.uncertainty.k_sigma * s
        for j, a, s in zip(jumps, ok[:-1], sig_pair)))
    attrs["Stable"] = (len(ok) >= 3 and change_dir == 0 and not frag_r and disapp is None
                       and ar["snr"] <= cfg.uncertainty.k_sigma)
    attrs["conflicting_evidence"] = change_dir is None

    primary = next((c for c in PRECEDENCE if attrs.get(c)), None) or "Ambiguous"
    rec["primary_label"] = primary

    # ------------------------------------------------------------- agreement
    agree, against = [], []
    if primary in ("Retreat", "Advance", "Stable"):
        want = {"Retreat": -1, "Advance": 1, "Stable": 0}[primary]
        for name, line in fam.items():
            if not line:
                continue
            if line["direction"] == want:
                agree.append(name)
            elif line["significant"]:
                against.append(name)
    elif primary == "Fragmentation":
        agree.append("split_ratio")
        if any(abs(s["year"] - f["year"]) <= 1 for s in frag_n for f in frag_confirmed):
            agree.append("split_ndsi")
        if ev["glims_children"]:
            agree.append("glims_children")
        if any(f["pre_verified"] for f in frag_confirmed):
            agree.append("pre_split_connected")
        if ar["direction"] == 1:
            against.append("area_ratio")       # fragmentation while growing is suspicious
    elif primary == "Disappearance":
        agree.append("area_ratio")
        if all(e.area_ndsi_km2 < mmu for e in seq if e.year in disapp["confirmed_by"]):
            agree.append("area_ndsi")
    rec["agree"], rec["against"] = agree, against

    # ------------------------------------------------------------- tier
    valid_frac = len(ok) / max(1, len(win))
    strong_line = any(fam[n] and fam[n]["snr"] >= cfg.tiers.strong_snr for n in agree if n in fam)
    strength = {
        "Retreat": strong_line,
        "Advance": strong_line,
        "Stable": len(ok) >= 4,
        "Fragmentation": any(f["confirmed"] and f["pre_verified"] for f in frag_confirmed),
        "Disappearance": disapp is not None,
    }.get(primary, False)
    if "area" in agree and area_opposed:
        against.append("area_masks_opposed")
    if primary == "Ambiguous":
        tier = "C"
    elif (len(ok) >= 3 and valid_frac >= cfg.tiers.min_valid_frac and len(agree) >= 2
          and not against and strength and ("area" not in agree or area_consistent)):
        tier = "A"
    elif not against and (len(agree) >= 2 or (len(agree) == 1 and strength)):
        tier = "B"
    else:
        tier = "C"
    rec["auto_tier"] = tier

    # ------------------------------------------------------------- flags
    fl = rec["flags"]
    if len(ok) < 3:
        fl.append("short_window")
    if any(v == "seasonal_snow" for v in rec["excluded"].values()):
        fl.append("seasonal_snow_epochs_excluded")
    if np.nanmean([e.outline_coverage for e in ok]) < 0.6:
        fl.append("low_outline_coverage(debris_or_prior_retreat)")
    if np.nanmean([e.mask_iou for e in ok]) < 0.8:
        fl.append("ratio_ndsi_masks_disagree")
    if attrs["possible_surge"]:
        fl.append("possible_surge")
    if any(e.beyond_outline_km2 > 0.05 * max(e.area_km2, 1e-9) for e in ok):
        fl.append("ice_beyond_outline")
    if attrs["conflicting_evidence"]:
        fl.append("conflicting_evidence")
    # ---- tier caps for snow/debris-prone evidence
    if primary == "Advance" and not ({"terminus", "glims"} & set(agree)):
        fl.append("advance_area_only")       # residual snow mimics advance: needs independent support
        rec["auto_tier"] = "C"
    if rec["auto_tier"] == "A" and ({"ratio_ndsi_masks_disagree", "low_outline_coverage(debris_or_prior_retreat)"} & set(fl)):
        rec["auto_tier"] = "B"
        fl.append("tier_capped_by_mask_quality")
    if primary == "Fragmentation" and attrs["Retreat"]:
        fl.append("also_retreat")
    return rec


def sample_windows(years, n_random, rng, min_len=2):
    """Full sequence + n random sub-windows of random length (distinct)."""
    y0, y1 = min(years), max(years)
    wins = [(y0, y1)]
    tries = 0
    while len(wins) < n_random + 1 and tries < 200:
        tries += 1
        L = int(rng.integers(min_len, y1 - y0 + 1))
        s = int(rng.integers(y0, y1 - L + 2))
        w = (s, s + L - 1)
        if w not in wins:
            wins.append(w)
    return wins
