"""Test 4 -- nearest-neighbour retrieval.

Every usable held-out timestamp is a query against all the others (cosine similarity).

* Task A, identity: relevant = other timestamps of the same glacier (P@1, P@5, mAP). The
  chance level is computed exactly from the glacier sizes, not assumed.
* Task B, similar state within a glacier: candidates are only that glacier's other
  timestamps. Does the nearest neighbour share the query's season, and what is its time
  offset? Also hit@1 for "same season, adjacent year" -- the closest thing to "the same
  glacier state" the unlabeled data supports.

Compared across: the trained model, an untrained same-architecture baseline, and raw
normalised pixels (32x32 thumbnails). Raw pixels are a strong baseline for identity and
seasonal retrieval, and the paper should say so rather than hide it.
"""
from __future__ import annotations

import numpy as np

from .context import SEASON_ID, EvalContext
from .extract import usable
from .stats import l2n
from ..utils.logging_utils import get_logger

log = get_logger("eval")

N_QUALITATIVE = 4
TOPK_SHOW = 5


def build_pool(ctx: EvalContext) -> dict:
    cols = {k: [] for k in ("gid", "season", "year", "elapsed", "thumb", "date", "gname",
                            "local")}
    for gi, gid in enumerate(ctx.test.gids):
        m = ctx.test.meta[gid]
        ok = np.flatnonzero(usable(m, ctx.cfg))
        if len(ok) == 0:
            continue
        cols["gid"].append(np.full(len(ok), gi))
        cols["local"].append(ok)
        cols["season"].append(np.array([SEASON_ID.get(m["seasons"][k], 0) for k in ok]))
        cols["year"].append(np.asarray(m["years"])[ok])
        cols["elapsed"].append(np.asarray(m["elapsed"])[ok])
        cols["thumb"].append(m["thumb"].float().numpy()[ok])
        cols["date"].append(np.array([m["dates"][k] for k in ok]))
        cols["gname"].append(np.array([gid] * len(ok)))
    return {k: np.concatenate(v) for k, v in cols.items()}


def embedding_matrix(ctx: EvalContext, pool: dict, model: str) -> np.ndarray:
    parts = []
    for gi, gid in enumerate(ctx.test.gids):
        sel = pool["gid"] == gi
        if sel.any():
            parts.append(ctx.test.feats[model][gid]["timestamp_out"][pool["local"][sel]])
    return np.concatenate(parts)


def pixel_matrix(pool: dict) -> np.ndarray:
    t = pool["thumb"]                                 # (N,6,s,s)
    flat = t.reshape(len(t), -1)
    return flat - flat.mean(0, keepdims=True)


def sim_matrix(X: np.ndarray) -> np.ndarray:
    Xn = l2n(X.astype(np.float32))
    S = Xn @ Xn.T
    np.fill_diagonal(S, -np.inf)
    return S


def identity_task(S: np.ndarray, g: np.ndarray) -> dict:
    N = len(g)
    p1, p5, ap, chance = [], [], [], []
    for q in range(N):
        rel = g == g[q]
        rel[q] = False
        nrel = int(rel.sum())
        if nrel == 0:
            continue
        order = np.argsort(-S[q])[: N - 1]
        hit = rel[order]
        cum = np.cumsum(hit)
        p1.append(float(hit[0]))
        p5.append(float(hit[:5].mean()))
        ap.append(float((cum / np.arange(1, len(hit) + 1) * hit).sum() / nrel))
        chance.append(nrel / (N - 1))
    return {"P@1": float(np.mean(p1)), "P@5": float(np.mean(p5)), "mAP": float(np.mean(ap)),
            "chance_P@1": float(np.mean(chance)), "n_queries": len(p1)}


def state_task(S: np.ndarray, pool: dict) -> dict:
    g, season, year, el = pool["gid"], pool["season"], pool["year"], pool["elapsed"]
    same_season, adj_hit, dts = [], [], []
    ch_season, ch_adj, ch_dt = [], [], []
    for gi in np.unique(g):
        idx = np.flatnonzero(g == gi)
        if len(idx) < 4:
            continue
        sub = S[np.ix_(idx, idx)]
        for a, q in enumerate(idx):
            cand = np.array([k for k in range(len(idx)) if k != a])
            nn = cand[np.argmax(sub[a, cand])]
            qi = idx[nn]
            same_season.append(float(season[qi] == season[q]))
            dts.append(abs(el[qi] - el[q]))
            ch_season.append(float(np.mean(season[idx[cand]] == season[q])))
            ch_dt.append(float(np.mean(np.abs(el[idx[cand]] - el[q]))))
            rel = (season[idx[cand]] == season[q]) & (np.abs(year[idx[cand]] - year[q]) == 1)
            if rel.any():
                adj_hit.append(float(rel[np.argmax(sub[a, cand])]))
                ch_adj.append(float(rel.mean()))
    return {"top1_same_season": float(np.mean(same_season)),
            "chance_same_season": float(np.mean(ch_season)),
            "hit@1_same_season_adjacent_year": float(np.mean(adj_hit)) if adj_hit else np.nan,
            "chance_adjacent_year": float(np.mean(ch_adj)) if ch_adj else np.nan,
            "median_top1_time_offset_yr": float(np.median(dts)),
            "mean_candidate_time_offset_yr": float(np.mean(ch_dt)),
            "n_queries": len(same_season)}


def _rgb(thumb: np.ndarray, band_order: list[str]) -> np.ndarray:
    idx = [band_order.index(b) for b in ("B04", "B03", "B02") if b in band_order]
    if len(idx) != 3:
        idx = [0, 1, 2]
    img = thumb[idx].transpose(1, 2, 0).astype(np.float32)
    lo, hi = np.percentile(img, [2, 98])
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)


def _qualitative(ctx, pool, S, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = ctx.rng(41)
    queries = rng.choice(len(S), size=min(N_QUALITATIVE, len(S)), replace=False)
    bands = list(ctx.cfg.data.band_order)
    fig, axes = plt.subplots(len(queries), TOPK_SHOW + 1,
                             figsize=(1.9 * (TOPK_SHOW + 1), 2.1 * len(queries)),
                             squeeze=False)
    for r, q in enumerate(queries):
        order = np.argsort(-S[q])[:TOPK_SHOW]
        for c, k in enumerate([q] + list(order)):
            ax = axes[r][c]
            ax.imshow(_rgb(pool["thumb"][k], bands), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            same = pool["gid"][k] == pool["gid"][q]
            if c == 0:
                ax.set_title(f"QUERY\n{pool['gname'][k][-9:]}\n{pool['date'][k]}", fontsize=6)
            else:
                ax.set_title(f"{'same' if same else 'other'}  s={S[q, k]:.2f}\n"
                             f"{pool['gname'][k][-9:]}\n{pool['date'][k]}", fontsize=6)
                for sp in ax.spines.values():
                    sp.set_edgecolor("tab:green" if same else "tab:red")
                    sp.set_linewidth(2)
    fig.suptitle(f"Nearest neighbours ({name}); green = same glacier, red = different",
                 fontsize=9)
    return ctx.store.save_fig(fig, "retrieval/fig_neighbours")


def run(ctx: EvalContext) -> dict:
    log.info("[retrieval] identity + within-glacier state retrieval")
    pool = build_pool(ctx)
    N = len(pool["gid"])
    if N < 10 or len(np.unique(pool["gid"])) < 2:
        ctx.store.add_section("Test 4 - Retrieval", "Too few usable timestamps.")
        return {"skipped": True}

    mats = {m: embedding_matrix(ctx, pool, m) for m in ctx.models if m in ctx.test.feats}
    mats["raw pixels (32x32)"] = pixel_matrix(pool)
    id_rows, st_rows, headline, sims = [], [], {}, {}
    for name, X in mats.items():
        S = sim_matrix(X)
        sims[name] = S
        a = identity_task(S, pool["gid"])
        b = state_task(S, pool)
        id_rows.append({"representation": name, **a})
        st_rows.append({"representation": name, **b})
        headline[name] = {"identity": a, "state": b}
        log.info("[retrieval] %-20s P@1 %.3f (chance %.3f) mAP %.3f | same-season@1 %.3f "
                 "(chance %.3f)", name, a["P@1"], a["chance_P@1"], a["mAP"],
                 b["top1_same_season"], b["chance_same_season"])
    t_id = ctx.store.write_table("retrieval/identity", id_rows)
    t_st = ctx.store.write_table("retrieval/state_within_glacier", st_rows)
    fig = _qualitative(ctx, pool, sims[ctx.primary], ctx.primary)
    ctx.store.write_json("retrieval/metrics.json", headline)

    md = [
        f"{N} usable held-out timestamps from {len(np.unique(pool['gid']))} glaciers; "
        "every timestamp is a query against all others.", "",
        "### Task A - glacier-identity retrieval (all test timestamps as the pool)", t_id, "",
        "### Task B - similar state within a glacier (candidates = same glacier only)",
        t_st, "",
        "Read each row against its chance column. Raw pixels are a legitimate baseline: if "
        "they match or beat the trained model on a task, that task does not demonstrate "
        "the benefit of pretraining. Identity retrieval measures appearance, so it is not "
        "evidence about temporal reasoning; Task B is the more relevant one.", "",
        f"![]({fig})",
    ]
    ctx.store.add_section("Test 4 - Retrieval", "\n".join(md))
    ctx.store.record("retrieval", headline)
    return headline
