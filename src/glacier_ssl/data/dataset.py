"""Phase 4 — torch Dataset + variable-length collate (the shared front end).

``GlacierWindowDataset`` turns one glacier into a sampled window and builds, before the
backbone, all tensors each head consumes:

* ``base``        : (T,6,H,W) normalized + per-timestamp validity — Head 2 / canonical seq.
* ``dino_views``  : list of crop views (T,6,out,out) + per-timestamp crop validity — Head 1.
* ``intervals``   : directed (i→j) interval, two augmented views, endpoint images — Head 3.
* ``distance_pairs`` : (i,j) index pairs into the base window — Head 2.

Every tensor is reproducible from the stored ``WindowSample`` spec (dates, indices, crop
coords, aug params, glacier id, seed). ``collate_windows`` pads variable ``T`` to the batch
max and emits timestamp masks so padding is inert downstream (build plan Phase 8).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import Config
from .augmentation import apply_view
from .manifest import GlacierSequence
from .preprocessing import Preprocessor
from .sampling import WindowSampler, WindowSample


class GlacierWindowDataset(Dataset):
    def __init__(self, sequences: dict[str, GlacierSequence], config: Config,
                 preprocessor: Preprocessor, sampler: WindowSampler,
                 epoch_seed: int = 0, build_dino: bool = True,
                 build_intervals: bool = True, build_pairs: bool = True):
        self.cfg = config
        self.seqs = list(sequences.values())
        self.pre = preprocessor
        self.sampler = sampler
        self.epoch_seed = epoch_seed
        self.build_dino = build_dino
        self.build_intervals = build_intervals
        self.build_pairs = build_pairs

    def set_epoch(self, epoch_seed: int) -> None:
        self.epoch_seed = epoch_seed

    def __len__(self) -> int:
        return len(self.seqs)

    # -- loading -----------------------------------------------------------------
    def _load_window(self, sample: WindowSample, seq: GlacierSequence):
        imgs, masks = [], []
        for idx in sample.obs_indices:
            obs = seq.observations[idx]
            norm, valid = self.pre.process(obs)          # (6,H,W), (H,W)
            imgs.append(torch.from_numpy(norm))
            masks.append(torch.from_numpy(valid))
        window = torch.stack(imgs, dim=0)                # (T,6,H,W)
        valid = torch.stack(masks, dim=0)                # (T,H,W)
        return window, valid

    def __getitem__(self, idx: int) -> dict:
        seq = self.seqs[idx]
        seed = self.epoch_seed * 1_000_003 + idx
        sample = self.sampler.sample(seq, seed=seed)
        rng = np.random.default_rng(seed + 7)

        window, valid = self._load_window(sample, seq)   # (T,6,H,W),(T,H,W)

        # base view (identity crop) -> timestamp validity for the canonical sequence.
        base_img, base_valid = apply_view(window, valid, sample.base_view, self.cfg)
        out: dict = {
            "glims_id": sample.glims_id,
            "split": sample.split,
            "T": len(sample.obs_indices),
            "dates": sample.dates,
            "seasons": sample.seasons,
            "elapsed_years": torch.tensor(sample.elapsed_years, dtype=torch.float32),
            "span_years": float(sample.span_years),
            "observation_ids": sample.observation_ids,
            "base_img": base_img,                        # (T,6,out,out)
            "base_valid": base_valid,                    # (T,)
            "sample_spec": sample,
        }

        if self.build_dino:
            views = []
            for vs in sample.dino_views:
                vimg, vvalid = apply_view(window, valid, vs, self.cfg)
                views.append({"img": vimg, "crop_valid": vvalid,
                              "view_type": vs.view_type})
            out["dino_views"] = views

        if self.build_pairs:
            sample.distance_pairs = self.sampler.sample_distance_pairs(
                sample, int(self.cfg.head2.pairs_per_glacier), rng)
            out["distance_pairs"] = sample.distance_pairs

        if self.build_intervals:
            sample.change_intervals = self.sampler.sample_change_intervals(
                sample, int(self.cfg.head3.intervals_per_glacier), rng)
            intervals = []
            for iv in sample.change_intervals:
                i, j = iv["i"], iv["j"]
                endpoints = window[[i, j]]               # (2,6,H,W)
                ep_valid = valid[[i, j]]                 # (2,H,W)
                a_img, a_valid = apply_view(endpoints, ep_valid, iv["view_a"], self.cfg)
                b_img, b_valid = apply_view(endpoints, ep_valid, iv["view_b"], self.cfg)
                intervals.append({
                    "i": i, "j": j, "gap_years": float(iv["gap_years"]),
                    "season_i": sample.seasons[i], "season_j": sample.seasons[j],
                    "view_a_img": a_img, "view_a_valid": a_valid,   # (2,6,out,out),(2,)
                    "view_b_img": b_img, "view_b_valid": b_valid,
                })
            out["intervals"] = intervals

        return out


def _pad_sequence_batch(imgs: list[torch.Tensor], valids: list[torch.Tensor]):
    """Pad a list of (T_i,C,H,W) + (T_i,) to (B,T_max,C,H,W) + masks (B,T_max)."""
    B = len(imgs)
    T_max = max(x.shape[0] for x in imgs)
    C, H, W = imgs[0].shape[1:]
    out = torch.zeros(B, T_max, C, H, W, dtype=imgs[0].dtype)
    ts_mask = torch.zeros(B, T_max, dtype=torch.bool)      # real observation present
    crop_valid = torch.zeros(B, T_max, dtype=torch.bool)   # valid AND real
    for b, (im, va) in enumerate(zip(imgs, valids)):
        t = im.shape[0]
        out[b, :t] = im
        ts_mask[b, :t] = True
        crop_valid[b, :t] = va
    return out, ts_mask, crop_valid


def collate_windows(batch: list[dict]) -> dict:
    """Variable-length collate. Pads base + each DINO view slot; keeps Head 2 pairs and
    Head 3 intervals as flat, batch-indexed lists of tensors."""
    B = len(batch)
    base_imgs = [s["base_img"] for s in batch]
    base_valids = [s["base_valid"] for s in batch]
    base, ts_mask, base_valid = _pad_sequence_batch(base_imgs, base_valids)

    elapsed = torch.zeros(B, base.shape[1], dtype=torch.float32)
    span = torch.zeros(B, dtype=torch.float32)
    for b, s in enumerate(batch):
        elapsed[b, : s["T"]] = s["elapsed_years"]
        span[b] = s["span_years"]

    out = {
        "base": base, "timestamp_mask": ts_mask, "base_valid": base_valid,
        "elapsed_years": elapsed, "span_years": span,
        "seasons": [s["seasons"] for s in batch],
        "glims_id": [s["glims_id"] for s in batch],
        "T": [s["T"] for s in batch],
    }

    # DINO views: every sample builds the same number of view slots (config-driven).
    if "dino_views" in batch[0]:
        n_views = len(batch[0]["dino_views"])
        view_batches = []
        for v in range(n_views):
            imgs = [s["dino_views"][v]["img"] for s in batch]
            valids = [s["dino_views"][v]["crop_valid"] for s in batch]
            vb, vmask, vvalid = _pad_sequence_batch(imgs, valids)
            view_batches.append({
                "img": vb, "timestamp_mask": vmask, "crop_valid": vvalid,
                "view_type": batch[0]["dino_views"][v]["view_type"],
            })
        out["dino_views"] = view_batches

    # Head 2 pairs: (batch_index, i, j)
    if "distance_pairs" in batch[0]:
        pairs = []
        for b, s in enumerate(batch):
            for (i, j) in s["distance_pairs"]:
                pairs.append((b, i, j))
        out["distance_pairs"] = pairs

    # Head 3 intervals: flat list, each carries its own endpoint view tensors.
    if "intervals" in batch[0]:
        intervals = []
        for b, s in enumerate(batch):
            for iv in s["intervals"]:
                intervals.append({"batch_index": b, **iv})
        out["intervals"] = intervals

    return out
