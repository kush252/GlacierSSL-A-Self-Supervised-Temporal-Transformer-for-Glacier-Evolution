"""Shared test helpers: a small, internally-consistent config for fast CPU tests.

The real default config keeps the spec dims (embed_dim 768, 512x512, 64 regions). For unit
tests we shrink to a self-consistent miniature (embed_dim 64, image 128 -> grid 8 -> two
HPE merges -> 2x2 = 4 regions) so shape logic is exercised without 12-block/512px cost.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glacier_ssl.config import load_config


def small_config(**extra):
    overrides = {
        "data": {"image_size": 128},
        "vit": {"image_size": 128, "patch_size": 16, "embed_dim": 64, "depth": 2,
                "num_heads": 4, "drop_path": 0.0, "dropout": 0.0},
        "hpe": {"embed_dim": 64, "input_grid": 8, "num_merge_stages": 2},
        "tet": {"embed_dim": 64, "depth": 2, "num_heads": 4, "num_regions": 4,
                "mlp_dropout": 0.0, "attn_dropout": 0.0},
        "sampling": {"global_crop_size": 128, "local_crop_size": 128},
        "head1": {"hidden_dim": 128, "bottleneck_dim": 64, "num_prototypes": 256},
        "head2": {"mlp_hidden": [64, 32]},
        "head3": {"encoder_hidden": [128], "encoder_out": 64, "proj_dim": 32,
                  "proj_hidden": 64},
    }
    for k, v in extra.items():
        overrides.setdefault(k, {})
        if isinstance(v, dict):
            overrides[k].update(v)
        else:
            overrides[k] = v
    return load_config(overrides=overrides)
