"""Test 5 -- robustness to missing timestamps, short sequences and cloud cover.

This is the test that uses the variable-length design directly. ViT+HPE act on each
timestamp independently, so the cached pre-TET regional tokens let us drop or truncate
timestamps and re-run ONLY the temporal transformer -- exactly what a shorter / gappier
input would have produced, at a fraction of the cost.

A. Missing observations: drop a random fraction of each glacier's timestamps and measure
   (i) how far the surviving timestamps' embeddings move relative to the full-sequence
   context (cosine distance), (ii) how far the sequence-level embedding moves, and
   (iii) Head 2's error on the surviving pairs.
B. Sequence length: keep only the first / last L timestamps (L = 2 ... 24); same
   measurements, plus sequence-level glacier re-identification between two NON-overlapping
   windows of the same glacier (chance = 1 / number of glaciers).
C. Cloud cover, observed: Head 2 error and embedding outlier distance binned by the
   manifest's cloud percentage.
D. Cloud cover, simulated: random 8x8-grid blocks zeroed per timestamp (zero is exactly how
   masked pixels reach the model), then the FULL backbone re-run on a subset of glaciers.
   (``--occlusion-glaciers 0`` skips this part; it is the only part that re-reads images.)

Drift is context sensitivity, not error: a representation that changes when context is
removed is not necessarily wrong. Head 2 error and re-identification are the task-facing
numbers.
"""
from __future__ import annotations

import numpy as np
import torch

from ..train.accel import amp_settings
from .context import EvalContext
from .extract import (build_backbones, build_loader, extract_all, run_tet,
                      select_sequences, usable)
from .head2_eval import build_pairs, gather, predict_head2
from .stats import cosine_dist_rows, l2n, mean_ci
from ..utils.logging_utils import get_logger

log = get_logger("eval")

DROP_FRACS = [0.1, 0.25, 0.5, 0.75]
DROP_REPS = 3
LENGTHS = [2, 3, 4, 6, 8, 12, 16, 24]
CLOUD_EDGES = [5.0, 20.0, 50.0]
CLOUD_LABELS = ["<5%", "5-20%", "20-50%", ">=50%"]
OCC_FRACS = [0.1, 0.25, 0.5]


def _tet(bb, tokens, meta, sel, device, amp):
    """TET on a subset of cached tokens; elapsed time re-measured from the subset's first
    date, exactly as a shorter input sequence would have been presented."""
    sel = np.asarray(sel)
    el = np.asarray(meta["elapsed"])[sel]
    out = run_tet(bb, tokens[torch.as_tensor(sel)].unsqueeze(0).float(),
                  [meta["seasons"][k] for k in sel], el - el[0], device, amp)
    return (out["timestamp_out"][0].float().cpu().numpy(),
            out["sequence_out"][0].float().cpu().numpy())


def _head2_mae(head2, z: np.ndarray, el: np.ndarray, ok: np.ndarray, device) -> float:
    idx = np.flatnonzero(ok)
    if head2 is None or len(idx) < 2:
        return float("nan")
    a, b = np.triu_indices(len(idx), 1)
    i, j = idx[a], idx[b]
    gap = np.abs(el[j] - el[i])
    keep = gap > 1e-9
    if not keep.any():
        return float("nan")
    pred = predict_head2(head2, z[i[keep]], z[j[keep]], device)
    return float(np.abs(pred - gap[keep]).mean())


def _agg(per_glacier: list[dict], keys: list[str]) -> dict:
    out = {"n_glaciers": len(per_glacier)}
    for k in keys:
        mean, lo, hi = mean_ci([g[k] for g in per_glacier if k in g])
        out[k] = mean
        out[k + "_lo"], out[k + "_hi"] = lo, hi
    return out


def _fmt_rows(rows, keys):
    out = []
    for r in rows:
        row = {c: r[c] for c in r if c in ("condition", "n_glaciers", "n_timestamps")}
        for k in keys:
            if k in r:
                row[k] = r[k]
                row[k + "_95CI"] = f"[{r[k + '_lo']:.3f}, {r[k + '_hi']:.3f}]"
        out.append(row)
    return out


# --------------------------------------------------------------------------- A + B
def missing_and_length(ctx, head2, runs, amp):
    cfg, dev = ctx.cfg, ctx.device
    prim = ctx.primary
    bbs = {n: ctx.backbone(n) for n in runs}
    drop = {p: [] for p in DROP_FRACS}
    length = {L: [] for L in LENGTHS}
    seq_pairs = {L: ([], []) for L in LENGTHS}         # (window A, window B) seq embeddings
    seq_gids = {L: [] for L in LENGTHS}
    full = {"drift": [], "mae": []}
    rng = ctx.rng(51)

    for gid in ctx.test.gids:
        m = ctx.test.meta[gid]
        n = len(m["dates"])
        if n < 4 or ctx.test.feats[prim][gid]["tokens"] is None:
            continue
        ok = usable(m, cfg)
        el = np.asarray(m["elapsed"])
        F = {name: ctx.test.feats[name][gid] for name in runs}
        zf = F[prim]["timestamp_out"]
        if "student" in F:
            full["mae"].append({"mae": _head2_mae(head2, F["student"]["timestamp_out"],
                                                  el, ok, dev)})

        def measure(sel):
            res = {}
            for name in runs:
                z, s = _tet(bbs[name], F[name]["tokens"], m, sel, dev, amp)
                if name == prim:
                    res["drift"] = float(cosine_dist_rows(z, zf[sel]).mean())
                    res["seq_drift"] = float(cosine_dist_rows(s[None],
                                                              F[name]["sequence_out"][None])[0])
                    res["seq"] = s
                if name == "student" and head2 is not None:
                    res["mae"] = _head2_mae(head2, z, el[sel], ok[sel], dev)
            return res

        for p in DROP_FRACS:
            reps = []
            for _ in range(DROP_REPS):
                k = max(2, int(round((1 - p) * n)))
                sel = np.sort(rng.choice(n, size=k, replace=False))
                r = measure(sel)
                r.pop("seq")
                reps.append(r)
            drop[p].append({key: float(np.nanmean([r[key] for r in reps if key in r]))
                            if any(key in r for r in reps) else np.nan
                            for key in ("drift", "seq_drift", "mae")})
        for L in LENGTHS:
            if L > n:
                continue
            wins = {"first": np.arange(0, L), "last": np.arange(n - L, n)}
            rs = {w: measure(sel) for w, sel in wins.items()}
            length[L].append({key: float(np.nanmean([rs[w][key] for w in rs if key in rs[w]]))
                              if any(key in rs[w] for w in rs) else np.nan
                              for key in ("drift", "seq_drift", "mae")})
            if n >= 2 * L:                                  # non-overlapping windows
                seq_pairs[L][0].append(rs["first"]["seq"])
                seq_pairs[L][1].append(rs["last"]["seq"])
                seq_gids[L].append(gid)

    keys = ["drift", "seq_drift", "mae"]
    drop_rows = [{"condition": "dropped 0% (full)", **_agg(full["mae"], ["mae"]),
                  "drift": 0.0, "seq_drift": 0.0, "drift_lo": 0.0, "drift_hi": 0.0,
                  "seq_drift_lo": 0.0, "seq_drift_hi": 0.0}]
    drop_rows += [{"condition": f"dropped {int(p * 100)}%", **_agg(v, keys)}
                  for p, v in drop.items() if v]
    len_rows, ident_rows = [], []
    for L, v in length.items():
        if not v:
            continue
        len_rows.append({"condition": f"L={L}", **_agg(v, keys)})
        A, B = seq_pairs[L]
        if len(A) >= 5:
            sim = l2n(np.stack(A)) @ l2n(np.stack(B)).T
            acc = float((sim.argmax(1) == np.arange(len(A))).mean())
            ident_rows.append({"window_length": L, "n_glaciers": len(A),
                               "top1_reidentification": acc, "chance": 1.0 / len(A)})
    return drop_rows, len_rows, ident_rows


# --------------------------------------------------------------------------- C
def cloud_analysis(ctx, head2):
    cfg = ctx.cfg
    rows = []
    # outlier distance of each timestamp to its own glacier's centroid, by cloud bin
    bins_x, vals = [], []
    for gid in ctx.test.gids:
        m = ctx.test.meta[gid]
        z = ctx.test.feats[ctx.primary][gid]["timestamp_out"]
        c = np.asarray(m["cloud_pct"], dtype=float)
        cen = l2n(z.mean(0, keepdims=True))
        d = cosine_dist_rows(z, np.repeat(cen, len(z), 0))
        ok = usable(m, cfg) & np.isfinite(c)
        bins_x.append(np.digitize(c[ok], CLOUD_EDGES))
        vals.append(d[ok])
    bx, vv = np.concatenate(bins_x), np.concatenate(vals)

    mae_by_bin = {}
    if head2 is not None and "student" in ctx.test.feats:
        recs = []
        for gid in ctx.test.gids:
            m = ctx.test.meta[gid]
            recs.append({"gid": gid, "elapsed": np.asarray(m["elapsed"], dtype=float),
                         "seasons": m["seasons"], "usable": usable(m, cfg)})
        P = build_pairs(recs, 300, ctx.rng(61))
        if len(P["gap"]):
            zi, zj = gather(ctx.test.feats["student"], ctx.test.gids, P, "timestamp_out")
            err = np.abs(predict_head2(head2, zi, zj, ctx.device) - P["gap"])
            cmax = np.array([max(ctx.test.meta[ctx.test.gids[g]]["cloud_pct"][a],
                                 ctx.test.meta[ctx.test.gids[g]]["cloud_pct"][b])
                             for g, a, b in zip(P["g"].astype(int), P["i"].astype(int),
                                                P["j"].astype(int))], dtype=float)
            pb = np.digitize(cmax, CLOUD_EDGES)
            for k in range(len(CLOUD_LABELS)):
                sel = (pb == k) & np.isfinite(cmax)
                mae_by_bin[k] = (float(err[sel].mean()) if sel.any() else np.nan,
                                 int(sel.sum()))
    for k, lab in enumerate(CLOUD_LABELS):
        sel = bx == k
        rows.append({"cloud_pct_bin": lab, "n_timestamps": int(sel.sum()),
                     "mean_dist_to_glacier_centroid": float(vv[sel].mean()) if sel.any()
                     else np.nan,
                     "head2_MAE_yr_(pair max cloud)": mae_by_bin.get(k, (np.nan, 0))[0],
                     "n_pairs": mae_by_bin.get(k, (np.nan, 0))[1]})
    return rows


# --------------------------------------------------------------------------- D
def occlusion_analysis(ctx, head2, amp_cfg_runs):
    n_gl = int(ctx.extra.get("occlusion_glaciers", 0))
    if n_gl <= 0:
        return []
    rng = ctx.rng(71)
    gids = sorted(rng.choice(ctx.test.gids, size=min(n_gl, len(ctx.test.gids)),
                             replace=False).tolist())
    all_seqs = {s.glims_id: s for s in select_sequences(ctx.cfg, ctx.project_root,
                                                        ctx.test.split, None, ctx.seed)}
    seqs = [all_seqs[g] for g in gids if g in all_seqs]
    names = list(amp_cfg_runs)
    bbs = build_backbones(ctx.cfg, ctx.bundle, names, ctx.device)
    rows = []
    for f in OCC_FRACS:
        log.info("[robustness] simulated occlusion %.0f%% on %d glaciers", f * 100, len(seqs))
        loader = build_loader(ctx.cfg, ctx.project_root, seqs, ctx.max_timesteps,
                              occlusion=f, occlusion_seed=ctx.seed)
        meta, feats = extract_all(bbs, loader, ctx.device, ctx.cfg, keep_tokens=False)
        per_g = []
        for gid in gids:
            if gid not in meta:
                continue
            m = meta[gid]
            z_clean = ctx.test.feats[ctx.primary][gid]["timestamp_out"]
            ok = usable(m, ctx.cfg)
            r = {"drift": float(cosine_dist_rows(feats[ctx.primary][gid]["timestamp_out"],
                                                 z_clean).mean()),
                 "usable_fraction": float(ok.mean())}
            if "student" in feats and head2 is not None:
                r["mae"] = _head2_mae(head2, feats["student"][gid]["timestamp_out"],
                                      np.asarray(m["elapsed"]), ok, ctx.device)
            per_g.append(r)
        rows.append({"condition": f"{int(f * 100)}% of each image occluded",
                     **_agg(per_g, ["drift", "usable_fraction", "mae"])})
    return rows


# --------------------------------------------------------------------------- figures
def _figures(ctx, drop_rows, len_rows, ident_rows, cloud_rows, occ_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def series(rows, key, xs):
        y = np.array([r.get(key, np.nan) for r in rows], dtype=float)
        lo = np.array([r.get(key + "_lo", np.nan) for r in rows], dtype=float)
        hi = np.array([r.get(key + "_hi", np.nan) for r in rows], dtype=float)
        return xs, y, lo, hi

    figs = []
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.5), constrained_layout=True)
    xs = [0, 10, 25, 50, 75][:len(drop_rows)]
    for a, key, title in ((ax[0][0], "drift", "embedding drift vs context removed"),
                          (ax[0][1], "mae", "Head 2 MAE (yr) vs timestamps dropped")):
        x, y, lo, hi = series(drop_rows, key, xs)
        if np.isfinite(y).any():
            a.plot(x, y, "o-")
            a.fill_between(x, lo, hi, alpha=0.2)
        a.set(xlabel="% of timestamps dropped", title=title)
    xl = [int(r["condition"][2:]) for r in len_rows]
    for a, key, title in ((ax[1][0], "mae", "Head 2 MAE (yr) vs sequence length"),):
        x, y, lo, hi = series(len_rows, key, xl)
        if np.isfinite(y).any():
            a.plot(x, y, "o-")
            a.fill_between(x, lo, hi, alpha=0.2)
        a.set(xlabel="window length L (timestamps)", title=title, xscale="log")
    if ident_rows:
        a = ax[1][1]
        a.plot([r["window_length"] for r in ident_rows],
               [r["top1_reidentification"] for r in ident_rows], "o-", label="top-1")
        a.plot([r["window_length"] for r in ident_rows], [r["chance"] for r in ident_rows],
               "k--", label="chance")
        a.set(xlabel="window length L", title="sequence-level glacier re-identification",
              xscale="log")
        a.legend()
    figs.append(ctx.store.save_fig(fig, "robustness/fig_missing_and_length"))

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    ax[0].bar(range(len(cloud_rows)), [r["head2_MAE_yr_(pair max cloud)"] for r in cloud_rows])
    ax[0].set_xticks(range(len(cloud_rows)), [r["cloud_pct_bin"] for r in cloud_rows])
    ax[0].set(xlabel="cloud cover (pair maximum)", ylabel="Head 2 MAE (yr)",
              title="observed cloud cover")
    if occ_rows:
        xs = [10, 25, 50][:len(occ_rows)]
        ax[1].plot(xs, [r["drift"] for r in occ_rows], "o-", label="embedding drift")
        if any(np.isfinite(r.get("mae", np.nan)) for r in occ_rows):
            ax2 = ax[1].twinx()
            ax2.plot(xs, [r.get("mae", np.nan) for r in occ_rows], "s--", color="tab:red",
                     label="Head 2 MAE")
            ax2.set_ylabel("Head 2 MAE (yr)")
        ax[1].set(xlabel="% of image occluded", ylabel="drift vs clean",
                  title="simulated cloud occlusion")
    figs.append(ctx.store.save_fig(fig, "robustness/fig_clouds"))
    return figs


def run(ctx: EvalContext) -> dict:
    log.info("[robustness] missing timestamps, sequence length, cloud cover")
    if ctx.bundle is None:
        ctx.store.add_section("Test 5 - Robustness", "Needs a checkpoint. Skipped.")
        return {"skipped": True}
    head2 = ctx.head2()
    runs = [ctx.primary] + (["student"] if head2 is not None and ctx.primary != "student"
                            else [])
    if any(ctx.test.feats.get(n, {}) == {} for n in runs):
        raise RuntimeError("robustness needs features (with tokens) for: %s" % runs)
    amp = amp_settings(ctx.cfg.get("train.amp", False),
                       ctx.cfg.get("train.amp_dtype", "float16"), ctx.device)

    drop_rows, len_rows, ident_rows = missing_and_length(ctx, head2, runs, amp)
    cloud_rows = cloud_analysis(ctx, head2)
    occ_rows = occlusion_analysis(ctx, head2, runs)

    keys = ["drift", "seq_drift", "mae"]
    t_drop = ctx.store.write_table("robustness/missing_timestamps",
                                   _fmt_rows(drop_rows, keys))
    t_len = ctx.store.write_table("robustness/sequence_length", _fmt_rows(len_rows, keys))
    t_id = ctx.store.write_table("robustness/sequence_reidentification", ident_rows)
    t_cloud = ctx.store.write_table("robustness/cloud_observed", cloud_rows)
    t_occ = (ctx.store.write_table("robustness/cloud_simulated",
                                   _fmt_rows(occ_rows, ["drift", "usable_fraction", "mae"]))
             if occ_rows else "_(simulated occlusion not run; pass --occlusion-glaciers N)_")
    figs = _figures(ctx, drop_rows, len_rows, ident_rows, cloud_rows, occ_rows)

    headline = {
        "missing_timestamps": {r["condition"]: {k: r.get(k) for k in keys}
                               for r in drop_rows},
        "sequence_length": {r["condition"]: {k: r.get(k) for k in keys} for r in len_rows},
        "reidentification": {str(r["window_length"]): r["top1_reidentification"]
                             for r in ident_rows},
        "cloud_observed": {r["cloud_pct_bin"]: r["head2_MAE_yr_(pair max cloud)"]
                           for r in cloud_rows},
        "cloud_simulated": {r["condition"]: {"drift": r.get("drift"), "mae": r.get("mae")}
                            for r in occ_rows},
    }
    ctx.store.write_json("robustness/metrics.json", headline)

    md = [
        f"Embedding drift uses the `{ctx.primary}` backbone; Head 2 error uses the "
        "student path it was trained on. Means over glaciers with bootstrap 95% CIs.", "",
        "### A. Missing observations (random drop, 3 repeats per glacier)", t_drop, "",
        "### B. Short sequences (first / last L timestamps)", t_len, "",
        "Sequence-level re-identification between two non-overlapping windows of one "
        "glacier:", t_id, "",
        "### C. Observed cloud cover", t_cloud, "",
        "### D. Simulated cloud occlusion", t_occ, "",
        "**How to read this.** Drift measures how much context matters, not whether the "
        "output is wrong; the task-facing numbers are Head 2 MAE and re-identification. A "
        "flat Head 2 MAE as timestamps are removed supports the variable-length claim; a "
        "sharp rise at small L or high drop rates shows where the model needs context. The "
        "observed-cloud bins confound cloud with season (winter scenes are cloudier), so "
        "prefer the simulated occlusion for a controlled statement.", "",
    ] + [f"![]({f})" for f in figs]
    ctx.store.add_section("Test 5 - Robustness", "\n".join(md))
    ctx.store.record("robustness", headline)
    return headline
