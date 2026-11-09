"""Test 3 -- temporal trajectories: does a glacier's embedding move smoothly, and does the
movement track how much the scene changed?

Per held-out glacier (>= 4 usable timestamps), in embedding space (cosine distance):

* smoothness: path length of the true chronological order divided by the mean path length
  of random orderings (< 1 = order matters, the trajectory is smoother than chance);
* drift vs time: Spearman(embedding distance, elapsed gap), over all pairs and over
  SAME-SEASON pairs only (constant illumination/snow regime, so any drift is year-on-year);
* drift vs change: Spearman(embedding distance, pixel-space change) over same-season
  pairs, and its partial form with the elapsed gap regressed out. Pixel change is the mean
  absolute difference of the normalised bands over pixels valid in both dates (32x32
  thumbnails). It is a crude, label-free proxy for "how different do the two scenes look";
  it is NOT a measurement of glacier retreat (snow, shadows and residual cloud all move it);
* seasonal vs annual: mean distance between seasons of one year relative to the distance
  between the same season in adjacent years (> 1 = the seasonal cycle dominates).

The untrained baseline is run through the identical analysis.
"""
from __future__ import annotations

import numpy as np

from .context import EvalContext
from .extract import usable
from .stats import cosine_dist_rows, l2n, mean_ci, partial_spearman, spearman
from ..utils.logging_utils import get_logger

log = get_logger("eval")

MIN_USABLE = 4
MIN_OVERLAP = 0.25       # fraction of thumbnail pixels valid in both dates
N_PERMS = 20


def cos_dist_matrix(z: np.ndarray) -> np.ndarray:
    zn = l2n(z)
    return 1.0 - zn @ zn.T


def pixel_dist_matrix(thumb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Mean |band difference| over pixels valid in both dates; NaN if overlap too small."""
    n = thumb.shape[0]
    D = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            both = valid[i] & valid[j]
            if both.mean() < MIN_OVERLAP:
                continue
            D[i, j] = D[j, i] = np.abs(thumb[i][:, both] - thumb[j][:, both]).mean()
    return D


def glacier_metrics(z, elapsed, seasons, years, thumb, valid, rng) -> dict | None:
    n = len(z)
    if n < MIN_USABLE:
        return None
    Dz = cos_dist_matrix(z)
    # smoothness vs random orderings
    true_path = float(sum(Dz[t, t + 1] for t in range(n - 1)))
    rand = []
    for _ in range(N_PERMS):
        p = rng.permutation(n)
        rand.append(sum(Dz[p[t], p[t + 1]] for t in range(n - 1)))
    path_ratio = true_path / max(float(np.mean(rand)), 1e-12)

    iu, ju = np.triu_indices(n, 1)
    gap = np.abs(elapsed[ju] - elapsed[iu])
    same_season = np.array([seasons[a] == seasons[b] for a, b in zip(iu, ju)])
    year_gap = np.abs(years[ju] - years[iu])
    dz = Dz[iu, ju]
    Dp = pixel_dist_matrix(thumb, valid)
    dp = Dp[iu, ju]

    ss = same_season & (year_gap >= 1)
    out = {
        "n_timestamps": n,
        "path_ratio": path_ratio,
        "spearman_gap_all": spearman(dz, gap),
        "spearman_gap_same_season": spearman(dz[ss], gap[ss]) if ss.sum() >= 4 else np.nan,
    }
    ok = ss & np.isfinite(dp)
    out["spearman_pixel_same_season"] = spearman(dz[ok], dp[ok]) if ok.sum() >= 4 else np.nan
    out["partial_pixel_given_gap"] = (partial_spearman(dz[ok], dp[ok], gap[ok])
                                      if ok.sum() >= 5 else np.nan)
    out["pixel_vs_gap_same_season"] = spearman(dp[ok], gap[ok]) if ok.sum() >= 4 else np.nan
    # seasonal cycle vs year-on-year drift
    adj = same_season & (year_gap == 1)
    cross = (~same_season) & (year_gap == 0)
    out["seasonal_over_annual"] = (float(dz[cross].mean() / max(dz[adj].mean(), 1e-12))
                                   if adj.any() and cross.any() else np.nan)
    out["_pairs"] = (dz[ok], dp[ok], gap[ok])
    return out


METRIC_KEYS = ["path_ratio", "spearman_gap_all", "spearman_gap_same_season",
               "spearman_pixel_same_season", "partial_pixel_given_gap",
               "seasonal_over_annual"]


def _figures(ctx, per_model, examples):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    figs = []
    # (1) example trajectories for the primary model
    if examples:
        fig, axes = plt.subplots(1, len(examples), figsize=(4.2 * len(examples), 3.8),
                                 squeeze=False)
        for ax, (gid, z, el, seasons) in zip(axes[0], examples):
            xy = PCA(n_components=2).fit_transform(z)
            ax.plot(xy[:, 0], xy[:, 1], "-", color="0.7", lw=0.8, zorder=1)
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=el, cmap="viridis", s=28, zorder=2)
            for k in (0, len(xy) - 1):
                ax.annotate("start" if k == 0 else "end", xy[k], fontsize=7,
                            xytext=(3, 3), textcoords="offset points")
            ax.set(title=gid, xticks=[], yticks=[])
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, label="years")
        fig.suptitle(f"Embedding trajectories ({ctx.primary}); PCA of timestamp embeddings")
        figs.append(ctx.store.save_fig(fig, "trajectories/fig_examples"))

    # (2) per-glacier distributions + pooled pixel-change scatter
    models = list(per_model)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for m in models:
        rows = per_model[m]
        for a, key, title in ((ax[0], "path_ratio", "path length: true / random order"),
                              (ax[1], "spearman_gap_same_season",
                               "Spearman(emb. distance, gap) same season")):
            v = np.array([r[key] for r in rows], dtype=float)
            v = v[np.isfinite(v)]
            if len(v):
                a.hist(v, bins=25, alpha=0.55, label=f"{m} (mean {v.mean():.2f})")
            a.set_title(title, fontsize=9)
    ax[0].axvline(1.0, color="k", ls="--", lw=0.8)
    ax[1].axvline(0.0, color="k", ls="--", lw=0.8)
    ax[0].legend(fontsize=7); ax[1].legend(fontsize=7)
    prim = per_model[models[0]]
    dz = np.concatenate([r["_pairs"][0] for r in prim if "_pairs" in r] or [np.zeros(0)])
    dp = np.concatenate([r["_pairs"][1] for r in prim if "_pairs" in r] or [np.zeros(0)])
    if len(dz) > 5:
        sel = np.random.default_rng(0).choice(len(dz), min(4000, len(dz)), replace=False)
        ax[2].scatter(dp[sel], dz[sel], s=3, alpha=0.3)
        ax[2].set(xlabel="pixel-space change (same-season pairs)",
                  ylabel="embedding cosine distance",
                  title=f"pooled, {models[0]}: rho={spearman(dz, dp):.2f}")
    figs.append(ctx.store.save_fig(fig, "trajectories/fig_summary"))
    return figs


def run(ctx: EvalContext) -> dict:
    log.info("[trajectory] smoothness, drift vs time, drift vs pixel change")
    cfg = ctx.cfg
    per_model: dict[str, list[dict]] = {m: [] for m in ctx.models if m in ctx.test.feats}
    examples = []
    for gid in ctx.test.gids:
        m = ctx.test.meta[gid]
        ok = usable(m, cfg)
        if ok.sum() < MIN_USABLE:
            continue
        el = np.asarray(m["elapsed"])[ok]
        seasons = np.asarray(m["seasons"])[ok]
        years = np.asarray(m["years"])[ok]
        thumb = m["thumb"].float().numpy()[ok]
        tv = m["thumb_valid"].numpy()[ok]
        for model in per_model:
            z = ctx.test.feats[model][gid]["timestamp_out"][ok]
            r = glacier_metrics(z, el, seasons, years, thumb, tv, ctx.rng(31))
            if r is None:
                continue
            r["glims_id"] = gid
            per_model[model].append(r)
            if model == ctx.primary:
                examples.append((gid, z, el, seasons, len(z)))
    if not per_model[ctx.primary]:
        ctx.store.add_section("Test 3 - Temporal trajectories",
                              "No glacier had enough usable timestamps.")
        return {"skipped": True}
    examples = [e[:4] for e in sorted(examples, key=lambda e: -e[4])[:3]]

    rows, headline = [], {}
    for model, rs in per_model.items():
        row = {"model": model, "n_glaciers": len(rs)}
        headline[model] = {"n_glaciers": len(rs)}
        for key in METRIC_KEYS:
            mean, lo, hi = mean_ci([r[key] for r in rs])
            row[key] = mean
            row[key + "_95CI"] = f"[{lo:.3f}, {hi:.3f}]"
            headline[model][key] = mean
        pv = [r["pixel_vs_gap_same_season"] for r in rs]
        headline[model]["pixel_vs_gap_same_season"] = mean_ci(pv)[0]
        rows.append(row)
    table = ctx.store.write_table("trajectories/summary", rows)
    per_g = [{"model": m, "glims_id": r["glims_id"],
              **{k: r[k] for k in ["n_timestamps"] + METRIC_KEYS + ["pixel_vs_gap_same_season"]}}
             for m, rs in per_model.items() for r in rs]
    ctx.store.write_table("trajectories/per_glacier", per_g)
    figs = _figures(ctx, per_model, examples)
    ctx.store.write_json("trajectories/metrics.json", headline)

    pg = headline[ctx.primary]
    md = [
        f"{len(per_model[ctx.primary])} held-out glaciers with >= {MIN_USABLE} usable "
        "timestamps; means over glaciers with bootstrap 95% CIs (resampling glaciers).", "",
        table, "",
        "- `path_ratio` < 1: the true chronological order is smoother than random order.",
        "- `spearman_gap_*`: does embedding distance grow with elapsed time (all pairs / "
        "same-season pairs only)?",
        "- `spearman_pixel_same_season` and `partial_pixel_given_gap`: does embedding "
        "distance track pixel-space change between same-season dates, and does it still do "
        "so once elapsed time is regressed out?",
        "- `seasonal_over_annual` > 1: seasons differ more than the same season does "
        "year-on-year, i.e. the seasonal cycle dominates the embedding.", "",
        f"Mean pixel-change-vs-gap (same season) is {pg['pixel_vs_gap_same_season']:.2f}: "
        "if this is near zero the pixel proxy mostly measures noise, and the "
        "embedding-vs-pixel correlations should be read accordingly.", "",
        "**Caveat.** Pixel change is a proxy, not a retreat measurement. A positive "
        "correlation means the embedding moves when the scene looks different; it does not "
        "show that it encodes glacier retreat specifically.", "",
    ] + [f"![]({f})" for f in figs]
    ctx.store.add_section("Test 3 - Temporal trajectories", "\n".join(md))
    ctx.store.record("trajectory", headline)
    log.info("[trajectory] %s: path_ratio %.3f, rho(gap, same season) %.3f, "
             "rho(pixel) %.3f", ctx.primary, pg["path_ratio"],
             pg["spearman_gap_same_season"], pg["spearman_pixel_same_season"])
    return headline
