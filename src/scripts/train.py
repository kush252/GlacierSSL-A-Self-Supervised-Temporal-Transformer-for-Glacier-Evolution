"""Phase 12/14 — SSL training entrypoint over the real glacier dataset.

Boots config + logging + reproducibility (Phase 1), loads the manifest + train split,
builds the window dataset/loader (Phase 2-4), and runs the combined three-head loop
(Phase 12). Keep windows/image size small via config for CPU runs.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from glacier_ssl.config import load_config
from glacier_ssl.data.dataset import GlacierWindowDataset, collate_windows
from glacier_ssl.data.manifest import load_manifest, split_glaciers
from glacier_ssl.data.normalization import load_norm_stats
from glacier_ssl.data.preprocessing import Preprocessor
from glacier_ssl.data.sampling import WindowSampler
from glacier_ssl.train.trainer import SSLTrainer
from glacier_ssl.utils.logging_utils import get_logger
from glacier_ssl.utils.run_metadata import write_run_metadata
from glacier_ssl.utils.seed import seed_everything, worker_init_fn


def build_loader(cfg, project_root, split="train", limit=None):
    seqs = load_manifest(project_root / str(cfg.data.manifest_path))
    seqs = split_glaciers(seqs, split)
    if limit:
        seqs = dict(list(seqs.items())[:limit])
    norm = load_norm_stats(project_root / str(cfg.data.norm_stats_path))
    pre = Preprocessor(cfg, norm_stats=norm, project_root=project_root)
    sampler = WindowSampler(cfg)
    ds = GlacierWindowDataset(seqs, cfg, pre, sampler, epoch_seed=0)
    loader = DataLoader(ds, batch_size=int(cfg.train.batch_size), shuffle=True,
                        num_workers=int(cfg.train.num_workers),
                        collate_fn=collate_windows, worker_init_fn=worker_init_fn)
    return loader, len(seqs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit-glaciers", type=int, default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.set("train.epochs", args.epochs)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    log = get_logger("train", cfg.run.log_level)
    seed = seed_everything(int(cfg.run.seed))
    project_root = Path(__file__).resolve().parents[2]
    run_dir = project_root / str(cfg.run.output_root) / time.strftime("%Y%m%d_%H%M%S")
    write_run_metadata(run_dir, cfg.to_dict(), seed)
    log.info("Run dir: %s | device=%s | seed=%d", run_dir, device, seed)

    loader, n = build_loader(cfg, project_root, "train", args.limit_glaciers)
    log.info("Train glaciers: %d | steps/epoch=%d", n, len(loader))

    trainer = SSLTrainer(cfg, output_dir=run_dir, device=device)
    trainer.fit(loader, epochs=int(cfg.train.epochs))
    log.info("Training complete.")


if __name__ == "__main__":
    main()
