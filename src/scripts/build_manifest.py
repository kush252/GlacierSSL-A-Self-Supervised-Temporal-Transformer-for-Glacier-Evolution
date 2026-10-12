"""Build and persist the dataset manifest (Phase 2)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glacier_ssl.config import load_config
from glacier_ssl.data.manifest import ManifestBuilder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--min-observations", type=int, default=2)
    ap.add_argument("--no-disk-check", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    builder = ManifestBuilder(cfg)
    seqs = builder.build(require_on_disk=not args.no_disk_check,
                         min_observations=args.min_observations)
    builder.save(seqs)


if __name__ == "__main__":
    main()
