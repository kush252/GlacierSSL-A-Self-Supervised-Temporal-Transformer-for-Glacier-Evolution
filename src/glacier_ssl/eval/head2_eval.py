"""Test 1 -- Head 2 (temporal distance) on held-out glaciers + the shortcut diagnostics.

Implements head2_archi.md sections 8, 10 and 11:

* MAE / RMSE (years, converted back from log1p space) / Spearman, overall and per gap range,
  with glacier-level bootstrap CIs (pairs from one glacier are not independent);
* calibration (predicted vs actual gap), error vs gap, predicted-vs-target distribution;
* Experiment G  -- pair-order invariance;
* Experiment F  -- metadata-only baselines (index difference, sequence length, season
  difference; never the true gap) fitted on TRAIN glaciers' manifests, no images needed;
* Experiment E  -- visual-feature-only control: a ridge probe on per-timestamp features
  taken BEFORE any temporal encoding is added, vs the same probe on the post-TET features,
  vs the same probes on a randomly-initialised backbone.

A strong Head-2 score is only evidence of *visual* temporal-distance learning if it beats
the metadata baselines and the visual-only probe does clearly better than random weights.
"""
from __future__ import annotations

import numpy as np
import torch

from ..data.manifest import GlacierSequence
from ..data.sampling import elapsed_years_from_dates
from .context import SEASON_ID, EvalContext
from .extract import FeatureSet, usable
from .metrics import distance_regression_metrics
from .stats import cluster_bootstrap_ratio, spearman
from ..utils.logging_utils import get_logger

log = get_logger("eval")

PAIR_CAP_EVAL = 400      # max pairs per held-out glacier (T=36 gives 630)
PAIR_CAP_FIT = 60        # max pairs per train glacier for the fitted controls


# --------------------------------------------------------------------------- pair tables
def records_from_featureset(fs: FeatureSet, cfg) -> list[dict]:
    recs = []
    for gid in fs.gids:
        m = fs.meta[gid]
        recs.append({"gid": gid, "elapsed": np.asarray(m["elapsed"], dtype=float),
                     "seasons": m["seasons"], "usable": usable(m, cfg)})
    return recs


def records_from_manifest(seqs: list[GlacierSequence], cap) -> list[dict]:
    recs = []
    for s in seqs:
        obs = s.observations[:cap] if cap else s.observations
        if len(obs) < 2:
            continue
        el = np.asarray(elapsed_years_from_dates([o.date_obj() for o in obs]))
        recs.append({"gid": s.glims_id, "elapsed": el, "seasons": [o.season for o in obs],
                     "usable": np.ones(len(obs), dtype=bool)})
    return recs


def build_pairs(records: list[dict], max_pairs: int, rng: np.random.Generator) -> dict:
    """All (i<j) pairs of usable timestamps with distinct dates, capped per glacier."""
    cols = {k: [] for k in ("g", "i", "j", "gap", "idx_diff", "T", "sdiff")}
    for g, r in enumerate(records):
        idx = np.flatnonzero(r["usable"])
        if len(idx) < 2:
            continue
        a, b = np.triu_indices(len(idx), 1)
        i, j = idx[a], idx[b]
        gap = np.abs(r["elapsed"][j] - r["elapsed"][i])
        keep = gap > 1e-9
        i, j, gap = i[keep], j[keep], gap[keep]
        if len(i) > max_pairs:
            sel = rng.choice(len(i), size=max_pairs, replace=False)
            i, j, gap = i[sel], j[sel], gap[sel]
        si = np.array([SEASON_ID.get(r["seasons"][k], 0) for k in i])
        sj = np.array([SEASON_ID.get(r["seasons"][k], 0) for k in j])
        n = len(i)
        cols["g"].append(np.full(n, g)); cols["i"].append(i); cols["j"].append(j)
        cols["gap"].append(gap); cols["idx_diff"].append(j - i)
        cols["T"].append(np.full(n, len(r["elapsed"]))); cols["sdiff"].append((sj - si) % 4)
    return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()}


def gather(feats: dict, gids: list[str], P: dict, key: str):
    """Embeddings of both pair members, aligned with the pair table."""
    n = len(P["g"])
    d = next(iter(feats.values()))[key].shape[1]
    zi = np.empty((n, d), np.float32)
    zj = np.empty((n, d), np.float32)
    for g in np.unique(P["g"]).astype(int):
        m = P["g"] == g
        arr = feats[gids[g]][key]
        zi[m] = arr[P["i"][m].astype(int)]
        zj[m] = arr[P["j"][m].astype(int)]
    return zi, zj


def metadata_matrix(P: dict) -> np.ndarray:
    onehot = np.eye(4)[P["sdiff"].astype(int)]
    return np.column_stack([P["idx_diff"], P["T"], onehot])


# --------------------------------------------------------------------------- scoring
def predict_head2(head2, zi: np.ndarray, zj: np.ndarray, device, bs: int = 4096):
    out = []
    with torch.no_grad():
        for s in range(0, len(zi), bs):
            a = torch.from_numpy(zi[s:s + bs]).to(device)
            b = torch.from_numpy(zj[s:s + bs]).to(device)
            out.append(head2(a, b).float().cpu())
    return torch.cat(out).numpy() if out else np.zeros(0)


def _edge_labels(edges) -> list[str]:
    e = [f"{x:g}" for x in edges]
    return [f"<{e[0]}"] + [f"{e[k]}-{e[k + 1]}" for k in range(len(e) - 1)] + [f">={e[-1]}"]


def score(pred: np.ndarray, P: dict, edges) -> dict:
    """Headline metrics + per-gap-bin rows + glacier-bootstrap CIs for one predictor."""
    gap = P["gap"]
    base = distance_regression_metrics(pred, gap, gap_edges=edges)
    err = np.abs(pred - gap)
    gids = np.unique(P["g"]).astype(int)
    num = np.array([err[P["g"] == g].sum() for g in gids])
    den = np.array([(P["g"] == g).sum() for g in gids], dtype=float)
    mae, lo, hi = cluster_bootstrap_ratio(num, den)
    sq = np.array([((pred - gap)[P["g"] == g] ** 2).sum() for g in gids])
    rmse2, rlo, rhi = cluster_bootstrap_ratio(sq, den)
    bins = np.digitize(gap, edges)
    labels = _edge_labels(edges)
    rows = []
    for b in range(len(edges) + 1):
        m = bins == b
        rows.append({"gap_range_yr": labels[b], "n_pairs": int(m.sum()),
                     "mae_yr": float(err[m].mean()) if m.any() else float("nan"),
                     "bias_yr": float((pred - gap)[m].mean()) if m.any() else float("nan")})
    return {"mae": mae, "mae_lo": lo, "mae_hi": hi, "rmse": float(np.sqrt(rmse2)),
            "rmse_lo": float(np.sqrt(rlo)) if np.isfinite(rlo) else float("nan"),
            "rmse_hi": float(np.sqrt(rhi)) if np.isfinite(rhi) else float("nan"),
            "spearman": base["spearman"], "bins": rows, "n_pairs": int(len(gap)),
            "n_glaciers": int(len(gids))}


def _row(name: str, features: str, s: dict) -> dict:
    return {"predictor": name, "features": features, "n_pairs": s["n_pairs"],
            "MAE_yr": s["mae"], "MAE_95CI": f"[{s['mae_lo']:.3f}, {s['mae_hi']:.3f}]",
            "RMSE_yr": s["rmse"], "Spearman": s["spearman"]}


# --------------------------------------------------------------------------- controls
def metadata_baselines(ctx: EvalContext, P_test: dict, edges) -> dict[str, dict]:
    """Experiment F: predict the gap from index difference / length / season difference."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression

    from .extract import select_sequences
    seqs = select_sequences(ctx.cfg, ctx.project_root, "train", None, ctx.seed)
    recs = records_from_manifest(seqs, ctx.max_timesteps)
    P_tr = build_pairs(recs, PAIR_CAP_FIT, ctx.rng(11))
    out = {}
    if len(P_tr["gap"]) == 0:
        return out
    Xtr, ytr, Xte = metadata_matrix(P_tr), P_tr["gap"], metadata_matrix(P_test)
    out["metadata: train-mean gap"] = score(np.full(len(Xte), ytr.mean()), P_test, edges)
    out["metadata: train-median gap"] = score(np.full(len(Xte), np.median(ytr)), P_test, edges)
    lin = LinearRegression().fit(Xtr, ytr)
    out["metadata: linear(index diff, length, season diff)"] = score(
        np.clip(lin.predict(Xte), 0, None), P_test, edges)
    gbm = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=ctx.seed)
    gbm.fit(Xtr, ytr)
    out["metadata: boosted trees(index diff, length, season diff)"] = score(
        np.clip(gbm.predict(Xte), 0, None), P_test, edges)
    log.info("metadata baselines fitted on %d train pairs (%d glaciers)", len(ytr),
             len(recs))
    return out


def _pair_features(zi: np.ndarray, zj: np.ndarray) -> np.ndarray:
    """concat(|u_i-u_j|, u_i*u_j) with u = per-vector standardised (LayerNorm, no affine),
    the same symmetric construction Head 2 uses."""
    def ln(z):
        return (z - z.mean(1, keepdims=True)) / np.maximum(z.std(1, keepdims=True), 1e-6)
    ui, uj = ln(zi), ln(zj)
    return np.concatenate([np.abs(ui - uj), ui * uj], axis=1)


def feature_probes(ctx: EvalContext, P_test: dict, edges) -> dict[str, dict]:
    """Experiment E: ridge on log1p(gap) from frozen per-timestamp features.

    ``visual_pool`` = ViT+HPE features pooled before any date/season encoding is added, so
    it cannot read the gap from the temporal embeddings -- only from appearance.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    if ctx.train is None:
        return {}
    P_tr = build_pairs(records_from_featureset(ctx.train, ctx.cfg), PAIR_CAP_FIT,
                       ctx.rng(13))
    if len(P_tr["gap"]) < 50:
        return {}
    out = {}
    for model in ctx.models:
        if model not in ctx.train.feats or model not in ctx.test.feats:
            continue
        for key, desc in (("timestamp_out", "post-TET (dates visible to TET)"),
                          ("visual_pool", "pre-TET appearance only")):
            ztr_i, ztr_j = gather(ctx.train.feats[model], ctx.train.gids, P_tr, key)
            zte_i, zte_j = gather(ctx.test.feats[model], ctx.test.gids, P_test, key)
            Xtr, Xte = _pair_features(ztr_i, ztr_j), _pair_features(zte_i, zte_j)
            sc = StandardScaler().fit(Xtr)
            ridge = RidgeCV(alphas=[1, 10, 100, 1e3, 1e4]).fit(
                sc.transform(Xtr), np.log1p(P_tr["gap"]))
            pred = np.expm1(np.clip(ridge.predict(sc.transform(Xte)), 0, None))
            out[f"ridge probe [{model}] {key}"] = (desc, score(pred, P_test, edges))
    return out


# --------------------------------------------------------------------------- figures
def _figures(ctx, P, pred, edges, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gap = P["gap"]
    figs = []
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    hb = ax[0].hexbin(gap, pred, gridsize=45, bins="log", mincnt=1, cmap="viridis")
    lim = max(gap.max(), pred.max()) * 1.02
    ax[0].plot([0, lim], [0, lim], "r--", lw=1, label="ideal")
    q = np.quantile(gap, np.linspace(0, 1, 13))
    qb = np.digitize(gap, q[1:-1])
    cx = [gap[qb == k].mean() for k in range(12) if (qb == k).any()]
    cy = [pred[qb == k].mean() for k in range(12) if (qb == k).any()]
    ax[0].plot(cx, cy, "o-", color="white", mec="k", ms=4, lw=1.2, label="binned mean")
    ax[0].set(xlabel="actual gap (years)", ylabel="predicted gap (years)",
              title="Calibration", xlim=(0, lim), ylim=(0, lim))
    ax[0].legend(loc="upper left", fontsize=8)
    fig.colorbar(hb, ax=ax[0], label="log10 pairs")
    labels = _edge_labels(edges)
    bins = np.digitize(gap, edges)
    data = [np.abs(pred - gap)[bins == b] for b in range(len(edges) + 1)]
    keep = [k for k, d in enumerate(data) if len(d)]
    ax[1].boxplot([data[k] for k in keep], tick_labels=[labels[k] for k in keep],
                  showfliers=False)
    ax[1].set(xlabel="actual gap range (years)", ylabel="|error| (years)",
              title="Error vs gap range")
    figs.append(ctx.store.save_fig(fig, f"head2/fig_calibration_error_{tag}"))

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    top = max(gap.max(), pred.max())
    b = np.linspace(0, top, 40)
    ax.hist(gap, bins=b, alpha=0.55, label="target gap", density=True)
    ax.hist(pred, bins=b, alpha=0.55, label="predicted gap", density=True)
    ax.set(xlabel="years", ylabel="density", title="Predicted vs target gap distribution")
    ax.legend()
    figs.append(ctx.store.save_fig(fig, f"head2/fig_gap_distribution_{tag}"))
    return figs


# --------------------------------------------------------------------------- entry point
def run(ctx: EvalContext) -> dict:
    log.info("[head2] held-out Head 2 evaluation + shortcut diagnostics")
    head2 = ctx.head2()
    if head2 is None:
        msg = "No Head 2 weights available (needs a checkpoint). Skipped."
        ctx.store.add_section("Test 1 - Head 2 (temporal distance)", msg)
        return {"skipped": True}
    if "student" not in ctx.test.feats:
        raise RuntimeError("Head 2 is trained on the student path; student features missing")

    edges = [float(e) for e in ctx.cfg.head2.gap_bins_years]
    recs = records_from_featureset(ctx.test, ctx.cfg)
    P = build_pairs(recs, PAIR_CAP_EVAL, ctx.rng(5))
    if len(P["gap"]) == 0:
        ctx.store.add_section("Test 1 - Head 2 (temporal distance)", "No usable pairs.")
        return {"skipped": True}
    zi, zj = gather(ctx.test.feats["student"], ctx.test.gids, P, "timestamp_out")
    pred = predict_head2(head2, zi, zj, ctx.device)
    main = score(pred, P, edges)

    # Experiment G: swapping the pair must not change the prediction.
    sel = ctx.rng(7).choice(len(pred), size=min(512, len(pred)), replace=False)
    swapped = predict_head2(head2, zj[sel], zi[sel], ctx.device)
    order_diff = float(np.abs(swapped - pred[sel]).max())

    controls = {}
    controls.update(metadata_baselines(ctx, P, edges))
    probes = feature_probes(ctx, P, edges)

    rows = [_row("Head 2 (trained, student)", "post-TET timestamp_out", main)]
    for name, s in controls.items():
        rows.append(_row(name, "metadata only (no images, no true gap)", s))
    for name, (desc, s) in probes.items():
        rows.append(_row(name, desc, s))
    table = ctx.store.write_table("head2/comparison", rows)
    bins_md = ctx.store.write_table("head2/mae_by_gap_range", main["bins"])
    figs = _figures(ctx, P, pred, edges, "head2")

    # per-glacier table (which glaciers are hard) -- useful for failure analysis
    err = np.abs(pred - P["gap"])
    per_g = [{"glims_id": recs[g]["gid"], "n_pairs": int((P["g"] == g).sum()),
              "mae_yr": float(err[P["g"] == g].mean())}
             for g in np.unique(P["g"]).astype(int)]
    ctx.store.write_table("head2/per_glacier", per_g)

    best_meta = min((s["mae"] for s in controls.values()), default=float("nan"))
    vis = {k: s["mae"] for k, (d, s) in probes.items()}
    headline = {
        "mae_years": main["mae"], "mae_ci": [main["mae_lo"], main["mae_hi"]],
        "rmse_years": main["rmse"], "spearman": main["spearman"],
        "n_pairs": main["n_pairs"], "n_glaciers": main["n_glaciers"],
        "mae_by_gap_range": {r["gap_range_yr"]: r["mae_yr"] for r in main["bins"]},
        "best_metadata_baseline_mae": best_meta,
        "probe_mae": vis,
        "pair_order_max_abs_diff": order_diff,
    }
    ctx.store.write_json("head2/metrics.json", {**headline, "all_predictors": rows})

    md = [
        f"Held-out glaciers: **{main['n_glaciers']}**, pairs: **{main['n_pairs']}**. "
        "Predictions use the **student** backbone + trained Head 2 (the path it was "
        "trained on). Errors in years; CIs resample glaciers.",
        "",
        f"**MAE {main['mae']:.3f} yr** (95% CI {main['mae_lo']:.3f}-{main['mae_hi']:.3f}), "
        f"RMSE {main['rmse']:.3f} yr, Spearman {main['spearman']:.3f}.",
        "",
        "### Per gap range", bins_md, "",
        "### Head 2 vs controls (head2_archi.md section 8)", table, "",
        f"Pair-order invariance (Experiment G): max |pred(i,j) - pred(j,i)| = "
        f"{order_diff:.2e} (should be ~0).", "",
        "**How to read this.** Head 2 only shows learned *visual* temporal distance if it "
        "beats the metadata-only rows (which see index difference / length / season, never "
        "images) AND the pre-TET appearance probe is clearly better than the same probe on "
        "random weights. If the metadata rows win, the pair construction is making the task "
        "trivially easy; if the post-TET probe is strong but the pre-TET one is weak, the "
        "score comes mainly from the explicit date encoding in the TET. Either outcome is "
        "reportable; neither supports a claim that Head 2 measures glacier change.", "",
    ] + [f"![]({f})" for f in figs]
    ctx.store.add_section("Test 1 - Head 2 (temporal distance)", "\n".join(md))
    ctx.store.record("head2", headline)
    log.info("[head2] MAE %.3f yr (CI %.3f-%.3f) | best metadata baseline %.3f | order "
             "diff %.1e", main["mae"], main["mae_lo"], main["mae_hi"], best_meta, order_diff)
    return headline
