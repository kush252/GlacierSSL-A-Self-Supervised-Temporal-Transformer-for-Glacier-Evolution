"""Shared state handed to every evaluation test."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..config import Config
from ..heads.head2_distance import Head2Distance
from .extract import FeatureSet, build_backbones
from .store import ResultsStore

SEASON_ID = {"DJF": 0, "MAM": 1, "JJA": 2, "SON": 3}


@dataclass
class EvalContext:
    cfg: Config
    store: ResultsStore
    device: torch.device
    project_root: Path
    primary: str                       # model the label-free tests probe ("student"/"teacher")
    test: FeatureSet                   # held-out glaciers (features for every model)
    train: Optional[FeatureSet] = None  # small TRAIN subset, only for fitting control probes
    bundle: Optional[dict] = None
    seed: int = 0
    # Same-architecture reference models scored by every label-free test:
    #   "random"     -- untrained everything (is the model better than random?)
    #   "pretrained" -- pretrained ViT, untrained HPE/TET (does glacier SSL add anything
    #                   beyond the off-the-shelf spatial encoder it started from?)
    baselines: list = field(default_factory=lambda: ["random"])
    max_timesteps: Optional[int] = None
    extra: dict = field(default_factory=dict)

    # -- helpers -----------------------------------------------------------------
    @property
    def models(self) -> list[str]:
        """Models compared in the label-free tests: primary first, then the baselines."""
        return [self.primary] + list(self.baselines)

    def rng(self, salt: int = 0) -> np.random.Generator:
        return np.random.default_rng(self.seed * 7919 + salt)

    def head2(self) -> Optional[Head2Distance]:
        """Trained Head 2 (needs the checkpoint); None for checkpoint-free runs."""
        if self.bundle is None or self.bundle.get("head2") is None:
            return None
        h = Head2Distance(self.cfg)
        h.load_state_dict(self.bundle["head2"])
        return h.to(self.device).eval()

    def backbone(self, name: str):
        return build_backbones(self.cfg, self.bundle, [name], self.device)[name]
