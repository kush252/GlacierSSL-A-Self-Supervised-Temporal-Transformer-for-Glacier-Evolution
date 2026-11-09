"""Post-training evaluation suite (label-free tests) -- run after SSL pretraining.

    python src/scripts/evaluate.py --ckpt <runs/.../ckpt_latest.pt> --device cuda

Tests (all on HELD-OUT glaciers; see glacier_ssl/eval/*.py for what each measures):

    head2        Head 2 error in years + the shortcut diagnostics from head2_archi.md
    embedding    do the features cluster by season / region / size / latitude / year?
    trajectory   does a glacier's embedding move smoothly and track scene change?
    retrieval    nearest-neighbour identity and same-state retrieval
    robustness   missing timestamps, short sequences, observed + simulated cloud cover

Everything is written to ONE folder per run under ``src/artifacts/eval/`` (or
``/kaggle/working/eval_results`` on Kaggle): REPORT.md, summary.json, run_manifest.json,
per-test tables (.csv + .md) and figures (.png + .pdf). A cross-run ``results_index.jsonl``
accumulates headline numbers so checkpoints and ablations can be compared later.

The checkpoint's own config defines the architecture; ``--config`` is only consulted for
machine-specific keys (data root, workers) so a Kaggle config works unchanged.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from glacier_ssl.config import load_config
from glacier_ssl.eval import (context, embedding_eval, extract, head2_eval,
                              retrieval_eval, robustness_eval, trajectory_eval)
from glacier_ssl.eval.store import ResultsStore
from glacier_ssl.train.accel import resolve_device
from glacier_ssl.utils.logging_utils import get_logger
from glacier_ssl.utils.run_metadata import _git_state
from glacier_ssl.utils.seed import seed_everything

TESTS = {
    "head2": head2_eval.run,
    "embedding": embedding_eval.run,
    "trajectory": trajectory_eval.run,
    "retrieval": retrieval_eval.run,
    "robustness": robustness_eval.run,
}
TITLES = {"head2": "Test 1 - Head 2 (temporal distance)",
          "embedding": "Test 2 - Embedding analysis",
          "trajectory": "Test 3 - Temporal trajectories",
          "retrieval": "Test 4 - Retrieval",
          "robustness": "Test 5 - Robustness"}


def default_out_root(project_root: Path) -> Path:
    kaggle = Path("/kaggle/working")
    if kaggle.exists():
        return kaggle / "eval_results"
    return project_root / "src" / "artifacts" / "eval"


def zip_run(run_dir: Path) -> Path:
    """Zip a run folder WITHOUT the (large) feature cache, for easy download."""
    out = run_dir.with_suffix(".zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(run_dir.rglob("*")):
            if f.is_file() and "cache" not in f.relative_to(run_dir).parts:
                zf.write(f, Path(run_dir.name) / f.relative_to(run_dir))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="training checkpoint (ckpt_latest.pt)")
    ap.add_argument("--config", default=None,
                    help="optional YAML for machine-specific keys (data root, workers)")
    ap.add_argument("--split", default="test", help="held-out split to evaluate")
    ap.add_argument("--tests", default="all",
                    help="comma list of: " + ",".join(TESTS) + " (default: all)")
    ap.add_argument("--backbone", default="student", choices=["student", "teacher"],
                    help="weights the label-free tests probe. Head 2 was trained on the "
                         "student path, so student keeps every test on the same weights; "
                         "Head 2 always uses student regardless.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--data-root", default=None,
                    help="root the manifest's relative band paths resolve against. The "
                         "checkpoint stores the machine it trained on (e.g. a Kaggle "
                         "mount), so pass 'project' (or a path) to evaluate elsewhere.")
    ap.add_argument("--threads", type=int, default=None,
                    help="torch CPU threads (default: torch's own choice)")
    ap.add_argument("--workers", type=int, default=0,
                    help="dataloader workers for image reading (default 0 = in-process; "
                         ">0 can fail on spawn platforms because each worker inherits a "
                         "parent holding the checkpoint)")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--name", default=None, help="optional tag appended to the run folder")
    ap.add_argument("--limit-glaciers", type=int, default=None,
                    help="evaluate a seeded random subset (debugging / quick looks)")
    ap.add_argument("--max-timesteps", type=int, default=None,
                    help="cap timestamps per glacier (default: all, up to 36)")
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the untrained same-architecture baseline")
    ap.add_argument("--probe-train-glaciers", type=int, default=120,
                    help="train glaciers used to fit the Head-2 control probes (0 = skip)")
    ap.add_argument("--occlusion-glaciers", type=int, default=24,
                    help="glaciers for the simulated-cloud test (0 = skip)")
    ap.add_argument("--reuse-cache", default=None,
                    help="cache/ folder of an earlier run to skip re-extraction")
    ap.add_argument("--zip", action="store_true", help="also zip the results (no cache)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    wanted = list(TESTS) if args.tests == "all" else [t.strip() for t in args.tests.split(",")]
    bad = [t for t in wanted if t not in TESTS]
    if bad:
        ap.error(f"unknown test(s): {bad}")

    project_root = Path(__file__).resolve().parents[2]
    out_root = Path(args.out_root) if args.out_root else default_out_root(project_root)
    device = resolve_device(args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    log = get_logger("eval")

    # ---- checkpoint + config -----------------------------------------------------
    ckpt_path = Path(args.ckpt)
    log.info("loading %s", ckpt_path)
    bundle = extract.load_bundle(ckpt_path)
    cfg = bundle["cfg"]
    if args.config:                                   # machine-specific keys only
        local = load_config(args.config)
        for key in ("data.data_root", "train.num_workers", "train.amp", "train.amp_dtype"):
            val = local.get(key, None)
            if val is not None:
                cfg.set(key, val)
    if args.data_root:
        cfg.set("data.data_root", None if args.data_root == "project" else args.data_root)
    cfg.set("eval.workers", args.workers)
    # Evaluation never needs gradient checkpointing; keep the trained architecture intact.

    ckpt_run = ckpt_path.resolve().parent.name or ckpt_path.stem
    run_name = (f"{ckpt_run}__step{bundle['step']}__{time.strftime('%Y%m%d_%H%M%S')}"
                + (f"__{args.name}" if args.name else ""))

    models = [args.backbone]
    if "head2" in wanted or "robustness" in wanted:
        if "student" not in models:
            models.append("student")
    if not args.no_baseline:
        models.append("random")
    # Each unused backbone state dict is ~0.7 GB for this architecture; drop them now so
    # they are not carried alongside the instantiated models.
    for unused in {"student", "teacher"} - set(models):
        bundle.pop(unused, None)
    keep_tokens = {m for m in models if m != "random"} if "robustness" in wanted else False

    cache_dirs = [Path(args.reuse_cache)] if args.reuse_cache else []
    store = ResultsStore(out_root, run_name, manifest={
        "checkpoint": {"path": str(ckpt_path), "step": bundle["step"],
                       "epoch": bundle["epoch"],
                       "size_mb": round(ckpt_path.stat().st_size / 1e6, 1),
                       "modified": time.strftime("%Y-%m-%dT%H:%M:%S",
                                                 time.localtime(ckpt_path.stat().st_mtime))},
        "args": vars(args), "tests": wanted, "models": models, "git": _git_state(),
        "torch": torch.__version__, "device": str(device),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "config": cfg.to_dict(),
    })
    fh = logging.FileHandler(store.dir / "eval.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    log.addHandler(fh)
    log.info("results -> %s", store.dir)

    # ---- features (one expensive pass shared by every test) ----------------------
    cap = args.max_timesteps
    tag = f"{args.split}_n{args.limit_glaciers or 'all'}_t{cap or 'all'}"
    test_fs = extract.ensure_features(
        cache_dirs + [store.cache_dir], store.cache_dir, tag, models, bundle, cfg,
        project_root, device, args.split, args.limit_glaciers, cap, keep_tokens, args.seed)
    train_fs = None
    if "head2" in wanted and args.probe_train_glaciers > 0:
        n = args.probe_train_glaciers
        train_fs = extract.ensure_features(
            cache_dirs + [store.cache_dir], store.cache_dir, f"train_n{n}_t{cap or 'all'}",
            [m for m in models], bundle, cfg, project_root, device, "train", n, cap,
            False, args.seed)
    store._manifest["n_glaciers"] = len(test_fs.meta)
    store._manifest["n_timestamps"] = test_fs.n_timestamps()
    store.write_json("run_manifest.json", store._manifest)
    log.info("evaluating %d glaciers / %d timestamps (%s split)", len(test_fs.meta),
             test_fs.n_timestamps(), args.split)

    ctx = context.EvalContext(
        cfg=cfg, store=store, device=device, project_root=project_root,
        primary=args.backbone, test=test_fs, train=train_fs, bundle=bundle,
        seed=args.seed, baseline=None if args.no_baseline else "random",
        max_timesteps=cap, extra={"occlusion_glaciers": args.occlusion_glaciers})

    # ---- tests (one failing test must not lose the others) -----------------------
    failures = {}
    for name in wanted:
        t0 = time.time()
        try:
            TESTS[name](ctx)
            log.info("[%s] finished in %.1fs", name, time.time() - t0)
        except Exception as exc:                       # noqa: BLE001
            failures[name] = repr(exc)
            log.error("[%s] FAILED: %s\n%s", name, exc, traceback.format_exc())
            store.add_section(TITLES[name], f"**FAILED**: `{exc!r}` -- see `eval.log`.")
            store.record(name, {"failed": repr(exc)})

    report = store.finalize()
    zipped = zip_run(store.dir) if args.zip else None
    log.info("done. report: %s", report)
    if zipped:
        log.info("zip (without cache): %s", zipped)
    if failures:
        log.error("tests that failed: %s", failures)
        sys.exit(1)


if __name__ == "__main__":
    main()
