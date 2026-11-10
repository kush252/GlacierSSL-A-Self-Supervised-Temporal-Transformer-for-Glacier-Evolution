"""Build ground truth + labels for a few glaciers and random-length windows.

  python run_labeling.py --n-glaciers 6 --windows 3 --seed 7
  python run_labeling.py --glims-id G007772E46495N G041523E43281N
  python run_labeling.py --merge-reviews output/run_x/review_alice.csv output/run_x/review_bob.csv --run output/run_x

Outputs (output/<run>/):
  labels.jsonl              one record per window: label, attributes, evidence lines, tier, figure
  epochs.csv                the measurement table every label is derived from (one row / glacier-year)
  candidates.csv            every scene evaluated + why it was or was not used
  glims_validation.csv      S2 measurement vs independent GLIMS outlines of the same year
  <glims_id>/ground_truth.json, masks.npz, fig_<y0>_<y1>.png/.pdf
  class_gallery.png/.pdf    what each class looks like (best example per class)
  review_blind.html         reviewer sheet (auto label hidden); review_with_auto.html
  summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from glacier_gt import CLASSES  # noqa: E402
from glacier_gt.catalog import glims_analyses, load_manifest  # noqa: E402
from glacier_gt.config import load_config  # noqa: E402
from glacier_gt.figures import class_gallery, window_figure  # noqa: E402
from glacier_gt.labels import sample_windows  # noqa: E402
from glacier_gt.pipeline import build_glacier  # noqa: E402
from glacier_gt.review import build_review_sheet, merge_reviews  # noqa: E402

TIER_RANK = {"A": 0, "B": 1, "C": 2, "-": 3}


def pick_glaciers(cfg, n, rng, split=None, require_glims=False):
    man = load_manifest(cfg)
    ids = sorted(g for g, v in man.items() if split is None or v.get("split") == split)
    rng.shuffle(ids)
    out = []
    for g in ids:
        if require_glims:
            df = glims_analyses(cfg, g)
            if df is None or not (df["src_date"].str[:4].astype(int) >= cfg.scenes.years[0]).any():
                continue
        out.append(g)
        if len(out) >= n:
            break
    return out


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-glaciers", type=int, default=6)
    ap.add_argument("--glims-id", nargs="*")
    ap.add_argument("--windows", type=int, default=3, help="random sub-windows per glacier (+ full sequence)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--split", choices=["train", "test"], default=None)
    ap.add_argument("--require-glims", action="store_true", help="only glaciers with GLIMS outlines in the period")
    ap.add_argument("--run", default=None, help="run directory name/path")
    ap.add_argument("--merge-reviews", nargs="*", help="reviewer CSVs to merge into final ground truth")
    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.merge_reviews:
        run = Path(args.run)
        rep = merge_reviews(run / "labels.jsonl", [Path(p) for p in args.merge_reviews], run / "ground_truth_final.jsonl")
        (run / "review_report.json").write_text(json.dumps(rep, indent=1))
        print(json.dumps(rep, indent=1))
        return

    rng = np.random.default_rng(args.seed)
    run = Path(args.run) if args.run else cfg.paths.output / f"run_{datetime.now():%Y%m%d_%H%M%S}_seed{args.seed}"
    run.mkdir(parents=True, exist_ok=True)
    gids = args.glims_id or pick_glaciers(cfg, args.n_glaciers, rng, args.split, args.require_glims)
    print(f"run -> {run}\nglaciers: {gids}")

    records, epoch_rows, cand_rows, val_rows, contexts = [], [], [], [], {}
    for gid in gids:
        t0 = time.time()
        print(f"[{gid}]")
        try:
            gt = build_glacier(gid, cfg)
        except Exception as e:  # keep going; report the failure
            print(f"  FAILED: {e}")
            continue
        gdir = run / gid
        saved = gt.save(gdir)
        val_rows += saved["glims_validation"]
        for e in gt.epochs:
            epoch_rows.append({"glims_id": gid, "region": gt.info.region, "split": gt.info.split,
                               **{k: v for k, v in asdict(e).items() if k not in ("component_areas", "path")}})
        for m in gt.measurements:
            cand_rows.append({"glims_id": gid, **{k: v for k, v in asdict(m).items()
                                                  if k not in ("component_areas", "path")}})
        for (y0, y1) in sample_windows(gt.years, args.windows, rng):
            rec = gt.label(y0, y1)
            rec["figure"] = window_figure(gt.info, gt.epochs, gt.masks, gt.regions, rec, gdir / f"fig_{y0}_{y1}")
            records.append(rec)
            contexts[rec["window_id"]] = gt
            print(f"  {y0}-{y1}: {rec['primary_label']:<13} tier {rec['auto_tier']}  "
                  f"valid {rec['n_valid']}/{rec['n_epochs']}  agree={rec['agree']} flags={rec['flags']}")
        print(f"  done in {time.time() - t0:.0f}s")

    with open(run / "labels.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, default=float) + "\n")
    write_csv(run / "epochs.csv", epoch_rows)
    write_csv(run / "candidates.csv", cand_rows)
    write_csv(run / "glims_validation.csv", val_rows)

    # best example per class for the gallery: highest tier, then most valid epochs
    examples = []
    for c in CLASSES:
        cands = [r for r in records if r["primary_label"] == c and r["n_valid"] >= 2]
        if not cands:
            continue
        best = min(cands, key=lambda r: (TIER_RANK[r["auto_tier"]], -r["n_valid"]))
        gt = contexts[best["window_id"]]
        examples.append({"label": c, "rec": best, "epochs": gt.epochs, "masks": gt.masks, "regions": gt.regions})
    class_gallery(examples, run / "class_gallery")

    if records:
        build_review_sheet(records, run, blind=True)
        build_review_sheet(records, run, blind=False)
    summary = {
        "glaciers": gids, "n_windows": len(records),
        "labels": {k: sum(r["primary_label"] == k for r in records) for k in CLASSES + ["Ambiguous", "Insufficient"]},
        "tiers": {k: sum(r["auto_tier"] == k for r in records) for k in "ABC-"},
        "epoch_status": {s: sum(r["status"].split(":")[0] == s for r in epoch_rows)
                         for s in sorted({r["status"].split(":")[0] for r in epoch_rows})},
        "glims_validation": {"n": len(val_rows),
                             "median_iou": float(np.median([v["iou"] for v in val_rows if v["iou"] is not None]))
                             if val_rows else None},
        "config": cfg.raw,
    }
    (run / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=1))


if __name__ == "__main__":
    main()
