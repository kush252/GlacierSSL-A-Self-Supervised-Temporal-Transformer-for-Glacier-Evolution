"""Global seeding for reproducibility (Phase 1, Cross-cutting rule #8)."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> int:
    """Seed Python, NumPy, and Torch RNGs. Returns the seed used."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def worker_init_fn(worker_id: int) -> None:
    """Deterministic per-worker seeding for DataLoader."""
    base = torch.initial_seed() % (2 ** 31)
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)
