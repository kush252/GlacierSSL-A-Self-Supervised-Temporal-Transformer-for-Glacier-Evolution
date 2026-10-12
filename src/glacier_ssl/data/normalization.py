"""Phase 3 — band-wise normalization statistics, fit on the TRAIN split only.

Uses a streaming (Welford-style, count/sum/sumsq) estimator over valid pixels across a
sample of train observations, so stats are leakage-free and reused for val/test/inference
(VIT_aarchi.md §2, build plan Phase 3). Stats are stored in reflectance units.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import Config
from ..utils.logging_utils import get_logger
from .manifest import GlacierSequence
from .preprocessing import Preprocessor


class NormalizationFitter:
    def __init__(self, config: Config, project_root: Optional[Path] = None):
        self.cfg = config
        self.bands = list(config.data.band_order)
        self.log = get_logger("normalization")
        # A preprocessor with no norm stats: we need raw reflectance + validity.
        self.pre = Preprocessor(config, norm_stats=None, project_root=project_root)

    def fit(self, train_seqs: dict[str, GlacierSequence],
            max_observations: int = 400, max_glaciers: Optional[int] = None,
            seed: int = 0) -> dict:
        rng = random.Random(seed)
        # Gather candidate observations from TRAIN glaciers only.
        gids = list(train_seqs.keys())
        rng.shuffle(gids)
        if max_glaciers:
            gids = gids[:max_glaciers]
        obs_pool = []
        for gid in gids:
            obs_pool.extend(train_seqs[gid].observations)
        rng.shuffle(obs_pool)
        obs_pool = obs_pool[:max_observations]
        self.log.info("Fitting normalization on %d train observations", len(obs_pool))

        n = np.zeros(len(self.bands), dtype=np.float64)
        s = np.zeros(len(self.bands), dtype=np.float64)
        ss = np.zeros(len(self.bands), dtype=np.float64)

        used = 0
        for obs in obs_pool:
            try:
                raw, scl = self.pre.load_raw(obs)
            except Exception as exc:
                self.log.warning("skip %s: %s", obs.glims_id, exc)
                continue
            valid = self.pre.validity_mask(raw, scl)
            refl = self.pre.to_reflectance(raw)
            if valid.sum() == 0:
                continue
            for i in range(len(self.bands)):
                vals = refl[i][valid]
                n[i] += vals.size
                s[i] += vals.sum(dtype=np.float64)
                ss[i] += np.square(vals, dtype=np.float64).sum(dtype=np.float64)
            used += 1

        stats = {}
        for i, band in enumerate(self.bands):
            if n[i] > 0:
                mean = s[i] / n[i]
                var = max(ss[i] / n[i] - mean * mean, 0.0)
                std = float(np.sqrt(var))
            else:
                mean, std = 0.0, 1.0
            stats[band] = {"mean": float(mean), "std": std, "count": int(n[i])}
        self.log.info("Used %d observations; stats: %s", used,
                      {b: (round(v["mean"], 4), round(v["std"], 4)) for b, v in stats.items()})
        return stats

    def save(self, stats: dict, path: Optional[str] = None) -> Path:
        out = self.pre._resolve(path or str(self.cfg.data.norm_stats_path))
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({"bands": self.bands, "units": "reflectance",
                       "scale": self.cfg.data.reflectance_scale, "stats": stats},
                      fh, indent=2)
        self.log.info("Wrote normalization stats: %s", out)
        return out


def load_norm_stats(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload["stats"]
