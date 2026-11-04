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
from glacier_ssl.train.checkpoint import load_checkpoint
from glacier_ssl.train.trainer import SSLTrainer
from glacier_ssl.utils.logging_utils import get_logger
from glacier_ssl.utils.run_metadata import write_run_metadata
from glacier_ssl.utils.seed import seed_everything, worker_init_fn


def build_loader(cfg, project_root, split="train", limit=None, pin_memory=False):
    seqs = load_manifest(project_root / str(cfg.data.manifest_path))
    seqs = split_glaciers(seqs, split)
    if limit:
        seqs = dict(list(seqs.items())[:limit])
    norm = load_norm_stats(project_root / str(cfg.data.norm_stats_path))
    pre = Preprocessor(cfg, norm_stats=norm, project_root=project_root)
    sampler = WindowSampler(cfg)
    ds = GlacierWindowDataset(seqs, cfg, pre, sampler, epoch_seed=0)
    nw = int(cfg.train.num_workers)
    loader = DataLoader(ds, batch_size=int(cfg.train.batch_size), shuffle=True,
                        num_workers=nw, collate_fn=collate_windows,
                        worker_init_fn=worker_init_fn, pin_memory=pin_memory,
                        persistent_workers=nw > 0)
    return loader, len(seqs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit-glaciers", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", default=None,
                    help="path to a ckpt_latest.pt to continue from (session caps)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.set("train.epochs", args.epochs)

    log = get_logger("train", cfg.run.log_level)
    # Report the hardware so Kaggle dual-GPU runs are easy to confirm at a glance.
    if torch.cuda.is_available():
        gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        log.info("CUDA available: %d GPU(s): %s", len(gpus), gpus)
    else:
        log.info("CUDA not available -> running on CPU.")
    is_cuda = torch.cuda.is_available() and args.device != "cpu"
    pin = (str(cfg.get("train.pin_memory", "auto")) == "auto" and is_cuda) or \
          cfg.get("train.pin_memory", False) is True

    seed = seed_everything(int(cfg.run.seed))
    project_root = Path(__file__).resolve().parents[2]
    run_dir = project_root / str(cfg.run.output_root) / time.strftime("%Y%m%d_%H%M%S")
    write_run_metadata(run_dir, cfg.to_dict(), seed)
    log.info("Run dir: %s | requested device=%s | seed=%d", run_dir, args.device, seed)

    loader, n = build_loader(cfg, project_root, "train", args.limit_glaciers, pin_memory=pin)
    log.info("Train glaciers: %d | steps/epoch=%d", n, len(loader))

    trainer = SSLTrainer(cfg, output_dir=run_dir, device=args.device)

    start_epoch = start_step = 0
    if args.resume:
        heads = {"head1": trainer.head1, "head2": trainer.head2, "head3": trainer.head3}
        start_step = load_checkpoint(args.resume, trainer.student, trainer.teacher,
                                     heads, trainer.optimizer,
                                     map_location=trainer.device)
        # steps advance once per batch, so the completed-epoch count follows from the
        # loader length rather than needing its own field in the payload.
        start_epoch = start_step // max(1, len(loader))
        log.info("Resumed from %s at step %d -> starting epoch %d/%d",
                 args.resume, start_step, start_epoch, int(cfg.train.epochs))

    trainer.fit(loader, epochs=int(cfg.train.epochs),
                start_epoch=start_epoch, start_step=start_step)
    log.info("Training complete.")


if __name__ == "__main__":
    main()
