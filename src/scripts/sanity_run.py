"""Phase 14 — small end-to-end sanity run on REAL data.

Loads a handful of real glaciers, builds the pipeline with a reduced image size / window
length (so a 12-block ViT at 512px is not required on CPU), and runs a few combined steps,
asserting finite losses and backbone gradient flow. Proves the system trains end to end;
it makes NO scientific-validity claim (build plan Phase 14).
"""
from __future__ import annotations

import argparse
import sys
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
from glacier_ssl.utils.seed import seed_everything


# Reduced-but-consistent config: image 128 -> ViT grid 8 -> HPE 8->4->2 = 4 regions.
SANITY_OVERRIDES = {
    "data": {"image_size": 128},
    "vit": {"image_size": 128, "patch_size": 16, "depth": 2, "num_heads": 6},
    "hpe": {"input_grid": 8},
    "tet": {"depth": 2, "num_regions": 4},
    "sampling": {"global_crop_size": 128, "local_crop_size": 128,
                 "window": {"min_length": 2, "max_length": 5, "short_max": 3,
                            "medium_max": 4,
                            "length_bucket_freq": {"short": 0.4, "medium": 0.3, "long": 0.3}}},
    "head1": {"num_prototypes": 1024},
    "head2": {"pairs_per_glacier": 3},
    "head3": {"intervals_per_glacier": 4},
    "train": {"batch_size": 2, "num_workers": 0, "log_every": 1},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glaciers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4)
    args = ap.parse_args()

    log = get_logger("sanity")
    seed_everything(42)
    cfg = load_config(overrides=SANITY_OVERRIDES)
    root = Path(__file__).resolve().parents[2]

    seqs = split_glaciers(load_manifest(root / str(cfg.data.manifest_path)), "train")
    seqs = dict(list(seqs.items())[: args.glaciers])
    norm = load_norm_stats(root / str(cfg.data.norm_stats_path))
    pre = Preprocessor(cfg, norm_stats=norm, project_root=root)
    sampler = WindowSampler(cfg)
    ds = GlacierWindowDataset(seqs, cfg, pre, sampler, epoch_seed=0)
    loader = DataLoader(ds, batch_size=int(cfg.train.batch_size), shuffle=True,
                        num_workers=0, collate_fn=collate_windows)

    out_dir = root / str(cfg.run.output_root) / "sanity"
    trainer = SSLTrainer(cfg, output_dir=out_dir, device="cpu")

    p0 = next(trainer.student.vit.parameters()).clone()
    step = 0
    done = False
    for batch in loader:
        if done:
            break
        logs = trainer.train_step(batch, step=step, total_steps=args.steps)
        log.info("step %d | total %.4f dino %.4f dist %.4f change %.4f | gnorm %.3f | "
                 "pairs %d intervals %d Ts %s",
                 step, logs["loss_total"], logs["loss_dino"], logs["loss_distance"],
                 logs["loss_change"], logs["backbone_grad_norm"],
                 logs.get("n_pairs", 0), logs.get("n_intervals", 0), batch["T"])
        assert all(torch.isfinite(torch.tensor(logs[k]))
                   for k in ("loss_total", "loss_dino", "loss_distance", "loss_change"))
        step += 1
        if step >= args.steps:
            done = True

    assert not torch.allclose(p0, next(trainer.student.vit.parameters())), \
        "backbone did not update"
    trainer.metrics.close()
    log.info("SANITY OK: %d real-data steps, finite losses, backbone updated. "
             "No scientific-validity claim is made.", step)


if __name__ == "__main__":
    main()
