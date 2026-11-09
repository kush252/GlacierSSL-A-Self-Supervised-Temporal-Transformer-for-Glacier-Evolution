"""Test 2 -- embedding analysis: what does the representation organise itself by?

For every held-out timestamp embedding (usable timestamps only) we ask, for the trained
model AND an untrained same-architecture baseline:

* health: variance, effective rank, mean pairwise cosine (collapse signals);
* how much variance is explained by glacier identity, season, region, year (eta-squared),
  and -- within a glacier, after removing its mean -- by season and by year;
* linear-probe accuracy for season / region / glacier size / latitude / year, with
  cross-validation GROUPED BY GLACIER so a glacier never sits in both train and test folds;
* PCA -> t-SNE maps coloured by each factor (UMAP too if ``umap-learn`` is installed).

The manifest carries no elevation, so glacier size (area) and latitude are the available
physical covariates; elevation would need a DEM join.
"""
from __future__ import annotations

import numpy as np
import torch

from .context import SEASON_ID, EvalContext
from .extract import usable
from .metrics import cosine_similarity_stats, embedding_variance
from .stats import effective_rank
from ..utils.logging_utils import get_logger

log = get_logger("eval")

PROBE_MAX_ROWS = 3000
TSNE_MAX_ROWS = 2500
MIN_REGION_GLACIERS = 8


def stack_embeddings(ctx: EvalContext, model: str, key: str = "timestamp_out") -> dict:
    """All usable test timestamps as one matrix with aligned metadata columns."""
    cols = {k: [] for k in ("X", "gid", "season", "year", "region", "area", "lat")}
    for gi, gid in enumerate(ctx.test.gids):
        m = ctx.test.meta[gid]
        ok = usable(m, ctx.cfg)
        n = int(ok.sum())
        if n == 0:
            continue
        cols["X"].append(ctx.test.feats[model][gid][key][ok])
        cols["gid"].append(np.full(n, gi))
        cols["season"].append(np.array([SEASON_ID.get(s, 0) for s in
                                        np.asarray(m["seasons"])[ok]]))
        cols["year"].append(np.asarray(m["years"])[ok])
        cols["region"].append(np.array([m["region"]] * n))
        cols["area"].append(np.full(n, np.nan if m["area_km2"] is None
                                    else float(m["area_km2"])))
        cols["lat"].append(np.full(n, np.nan if m["lat"] is None else float(m["lat"])))
    return {k: np.concatenate(v) for k, v in cols.items()}


def eta_squared(X: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of total variance explained by group means (between-group SS / total SS)."""
    X = np.asarray(X, dtype=np.float64)
    total = ((X - X.mean(0)) ** 2).sum()
    if total <= 0:
        return float("nan")
    mu = X.mean(0)
    between = 0.0
    for lab in np.unique(labels):
        m = labels == lab
        between += m.sum() * ((X[m].mean(0) - mu) ** 2).sum()
    return float(between / total)


def remove_group_mean(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = np.asarray(X, dtype=np.float64).copy()
    for g in np.unique(groups):
        m = groups == g
        out[m] -= out[m].mean(0)
    return out


def geographic_clusters(ctx, glacier: np.ndarray, gids: list[str], min_size: int):
    """Geographic grouping from glacier centroids (k-means on lat/lon).

    The manifest's ``region`` field holds glacier NAMES (one or two glaciers each), so it
    cannot serve as a region factor. Centroids can: the dataset spans the Alps, Caucasus,
    Tien Shan/Pamir, Scandinavia and the Himalaya. Clusters smaller than ``min_size``
    glaciers are dropped (label -1) rather than padded out.

    Returns (labels per timestamp, human-readable cluster names).
    """
    from sklearn.cluster import KMeans

    pts = np.array([[ctx.test.meta[g]["lat"] or np.nan,
                     ctx.test.meta[g]["lon"] or np.nan] for g in gids], dtype=float)
    ok = np.isfinite(pts).all(1)
    if ok.sum() < 2 * min_size:
        return np.full(len(glacier), -1), []
    k = int(np.clip(ok.sum() // (2 * min_size), 2, 6))
    km = KMeans(n_clusters=k, n_init=10, random_state=ctx.seed).fit(pts[ok])
    per_glacier = np.full(len(gids), -1)
    per_glacier[np.flatnonzero(ok)] = km.labels_
    counts = np.bincount(km.labels_, minlength=k)
    keep = [c for c in range(k) if counts[c] >= min_size]
    remap = {c: i for i, c in enumerate(keep)}
    names = [f"{km.cluster_centers_[c][0]:.0f}N,{km.cluster_centers_[c][1]:.0f}E "
             f"(n={counts[c]})" for c in keep]
    labels = np.array([remap.get(per_glacier[g], -1) for g in glacier])
    return labels, names


def tercile_labels(v: np.ndarray, glacier: np.ndarray) -> np.ndarray:
    """Terciles over GLACIERS (not timestamps), so long sequences don't dominate."""
    gl = {}
    for g, x in zip(glacier, v):
        gl.setdefault(g, x)
    vals = np.array([x for x in gl.values() if np.isfinite(x)])
    if len(vals) < 3:
        return np.full(len(v), -1)
    q = np.quantile(vals, [1 / 3, 2 / 3])
    out = np.digitize(v, q)
    out[~np.isfinite(v)] = -1
    return out


def probe(X, y, groups, seed: int, n_splits: int = 5) -> dict:
    """Logistic-regression probe, cross-validated with whole glaciers held out."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    keep = y >= 0
    X, y, groups = X[keep], y[keep], groups[keep]
    classes = np.unique(y)
    n_groups = len(np.unique(groups))
    if len(classes) < 2 or n_groups < 3:
        return {"acc": float("nan"), "bal_acc": float("nan"), "chance": float("nan"),
                "n_classes": int(len(classes)), "n": int(len(y))}
    rng = np.random.default_rng(seed)
    if len(y) > PROBE_MAX_ROWS:
        sel = rng.choice(len(y), PROBE_MAX_ROWS, replace=False)
        X, y, groups = X[sel], y[sel], groups[sel]
    pred = np.full(len(y), -1)
    for tr, te in GroupKFold(n_splits=min(n_splits, n_groups)).split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            pred[te] = y[tr][0]
            continue
        clf = make_pipeline(
            StandardScaler(),
            PCA(n_components=min(128, X.shape[1], len(tr) - 1), random_state=seed),
            LogisticRegression(max_iter=300, C=1.0))
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    majority = np.bincount(y).max() / len(y)
    return {"acc": float(accuracy_score(y, pred)),
            "bal_acc": float(balanced_accuracy_score(y, pred)),
            "chance": float(majority), "n_classes": int(len(classes)), "n": int(len(y))}


def health(X: np.ndarray, rng) -> dict:
    sub = X if len(X) <= 1500 else X[rng.choice(len(X), 1500, replace=False)]
    t = torch.from_numpy(sub.astype(np.float32))
    cs = cosine_similarity_stats(t)
    return {"variance": embedding_variance(torch.from_numpy(X.astype(np.float32))),
            "effective_rank": effective_rank(sub), "mean_pairwise_cos": cs["mean_cos"],
            "dim": int(X.shape[1])}


def _scatter_maps(ctx, tables: dict[str, dict], seed: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    rng = np.random.default_rng(seed)
    names = list(tables)
    n = len(next(iter(tables.values()))["gid"])
    sel = np.sort(rng.choice(n, min(TSNE_MAX_ROWS, n), replace=False))
    ref = tables[names[0]]
    # region -> colour index (top regions, rest grouped)
    regs, cnt = np.unique(ref["region"][sel], return_counts=True)
    top = list(regs[np.argsort(-cnt)][:8])
    reg_idx = np.array([top.index(r) if r in top else len(top) for r in ref["region"][sel]])
    factors = [
        ("season", ref["season"][sel], "tab10", ["DJF", "MAM", "JJA", "SON"]),
        ("region", reg_idx, "tab10", top + ["other"]),
        ("log10 area km2", np.log10(np.maximum(ref["area"][sel], 1e-3)), "viridis", None),
        ("latitude", ref["lat"][sel], "viridis", None),
        ("year", ref["year"][sel], "viridis", None),
    ]
    coords = {}
    for name, tab in tables.items():
        Z = tab["X"][sel]
        Z = PCA(n_components=min(50, Z.shape[1], len(Z) - 1), random_state=seed).fit_transform(Z)
        coords[name] = TSNE(n_components=2, perplexity=min(30, max(5, len(Z) // 10)),
                            init="pca", random_state=seed).fit_transform(Z)
    fig, axes = plt.subplots(len(names), len(factors), figsize=(3.6 * len(factors),
                                                               3.4 * len(names)),
                             squeeze=False)
    for r, name in enumerate(names):
        for c, (fname, vals, cmap, labels) in enumerate(factors):
            ax = axes[r][c]
            xy = coords[name]
            # categorical factors use a fixed 0..9 colour scale so the legend matches
            cat = {"vmin": 0, "vmax": 9} if labels is not None else {}
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=vals, cmap=cmap, s=4, alpha=0.8,
                            linewidths=0, **cat)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(fname, fontsize=10)
            if c == 0:
                ax.set_ylabel(name, fontsize=10)
            if labels is not None and r == 0:
                handles = [plt.Line2D([], [], marker="o", ls="", ms=4,
                                      color=plt.get_cmap(cmap)(k / 9), label=l)
                           for k, l in enumerate(labels)]
                ax.legend(handles=handles, fontsize=6, loc="best", framealpha=0.6)
            elif labels is None and r == len(names) - 1:
                fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle("t-SNE of timestamp embeddings (held-out glaciers)")
    return ctx.store.save_fig(fig, "embeddings/fig_tsne")


def run(ctx: EvalContext) -> dict:
    log.info("[embedding] health, variance decomposition, probes, t-SNE")
    rng = ctx.rng(21)
    tables = {m: stack_embeddings(ctx, m) for m in ctx.models if m in ctx.test.feats}
    ref = next(iter(tables.values()))
    if len(ref["gid"]) < 20:
        ctx.store.add_section("Test 2 - Embedding analysis", "Too few usable timestamps.")
        return {"skipped": True}

    y_season = ref["season"]
    y_geo, geo_names = geographic_clusters(ctx, ref["gid"], ctx.test.gids,
                                           MIN_REGION_GLACIERS)
    factors = {
        "season (4)": y_season,
        f"geographic cluster ({len(geo_names)})": y_geo,
        "glacier size (area terciles)": tercile_labels(ref["area"], ref["gid"]),
        "latitude (terciles)": tercile_labels(ref["lat"], ref["gid"]),
        "year": ref["year"] - ref["year"].min(),
    }

    health_rows, var_rows, probe_rows, headline = [], [], [], {}
    for model, tab in tables.items():
        X = tab["X"]
        h = health(X, rng)
        health_rows.append({"model": model, **h})
        Xw = remove_group_mean(X, tab["gid"])
        v = {"model": model,
             "glacier identity": eta_squared(X, tab["gid"]),
             "season": eta_squared(X, tab["season"]),
             "year": eta_squared(X, tab["year"]),
             "season | within glacier": eta_squared(Xw, tab["season"]),
             "year | within glacier": eta_squared(Xw, tab["year"])}
        var_rows.append(v)
        for fname, y in factors.items():
            p = probe(X, y, tab["gid"], ctx.seed)
            probe_rows.append({"model": model, "target": fname, "acc": p["acc"],
                               "balanced_acc": p["bal_acc"], "majority_chance": p["chance"],
                               "classes": p["n_classes"], "n": p["n"]})
        headline[model] = {
            "effective_rank": h["effective_rank"], "mean_pairwise_cos": h["mean_pairwise_cos"],
            "eta2_glacier": v["glacier identity"], "eta2_season": v["season"],
            "eta2_season_within_glacier": v["season | within glacier"],
            "eta2_year_within_glacier": v["year | within glacier"],
            "probe_acc": {r["target"]: r["acc"] for r in probe_rows if r["model"] == model},
        }
        log.info("[embedding] %s: erank %.1f, eta2(glacier) %.3f, eta2(season|glacier) %.3f",
                 model, h["effective_rank"], v["glacier identity"],
                 v["season | within glacier"])

    t_health = ctx.store.write_table("embeddings/health", health_rows)
    t_var = ctx.store.write_table("embeddings/variance_explained", var_rows)
    t_probe = ctx.store.write_table("embeddings/linear_probes", probe_rows)
    fig = _scatter_maps(ctx, tables, ctx.seed)
    ctx.store.write_json("embeddings/metrics.json", headline)

    md = [
        f"{len(ref['gid'])} usable held-out timestamps from "
        f"{len(np.unique(ref['gid']))} glaciers; `{ctx.primary}` backbone vs an untrained "
        f"{', '.join(f'`{b}`' for b in ctx.baselines) or 'no'} baseline(s) of the same "
        "architecture.", "",
        "### Embedding health (collapse check)", t_health, "",
        "### Share of variance explained by each factor (eta-squared)",
        "Rows marked `| within glacier` first subtract each glacier's mean embedding, so "
        "they show what the embedding encodes *about change over time* rather than which "
        "glacier it is.", t_var, "",
        "### Linear probes (5-fold CV, whole glaciers held out)", t_probe, "",
        "Geographic clusters are k-means groups of glacier centroids "
        + (", ".join(geo_names) if geo_names else "(too few glaciers to form any)")
        + ". The manifest's `region` field holds glacier names rather than regions, so it "
          "cannot be probed.", "",
        "A probe only supports 'the representation organises by X' when it clearly beats "
        "both `majority_chance` and the untrained baseline. Region/latitude/size accuracy "
        "can reflect glacier identity leaking through spectral appearance, since glaciers "
        "in one region look alike.", "", f"![]({fig})",
    ]
    ctx.store.add_section("Test 2 - Embedding analysis", "\n".join(md))
    ctx.store.record("embedding", headline)
    return headline
