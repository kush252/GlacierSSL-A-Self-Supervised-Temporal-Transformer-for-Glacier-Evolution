from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
    return obj


def load_config(path: str | Path | None = None) -> SimpleNamespace:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = _ns(raw)
    for key in vars(cfg.paths):
        p = Path(getattr(cfg.paths, key))
        setattr(cfg.paths, key, p if p.is_absolute() else (ROOT / p).resolve())
    cfg.raw = raw
    return cfg
