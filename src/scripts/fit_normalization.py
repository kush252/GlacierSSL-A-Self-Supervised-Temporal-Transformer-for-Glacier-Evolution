"""Fit and persist train-only normalization statistics (Phase 3)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glacier_ssl.config import load_config
from glacier_ssl.data.manifest import load_manifest, split_glaciers
from glacier_ssl.data.normalization import NormalizationFitter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-observations", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(__file__).resolve().parents[2]
    seqs = load_manifest(root / str(cfg.data.manifest_path))
    train = split_glaciers(seqs, "train")
    fitter = NormalizationFitter(cfg)
    stats = fitter.fit(train, max_observations=args.max_observations, seed=args.seed)
    fitter.save(stats)


if __name__ == "__main__":
    main()
