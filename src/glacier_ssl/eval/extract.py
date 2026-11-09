"""Feature extraction for the post-training evaluation suite.

Runs the trained backbone ONCE over every held-out glacier's full dated sequence (identity
crop, no augmentation -- the same "base view" Head 2 trained on) and caches everything the
tests need, so the five tests share one expensive pass:

* ``timestamp_out``  (T,D)   TET timestamp embedding (what Heads 1 and 2 consume)
* ``sequence_out``   (D,)    final context-token state
* ``visual_pool``    (T,D)   per-timestamp ViT+HPE features pooled BEFORE any temporal
                             encoding is added -- the "visual-feature-only" control
                             (head2_archi.md, Experiment E)
* ``tokens``         (T,R,D) the pre-TET regional tokens (fp16). ViT+HPE is per-timestamp
                             and independent, so robustness tests can drop / truncate
                             timestamps and re-run only the (cheap) TET, exactly.

Models: ``student`` / ``teacher`` come from the checkpoint; ``random`` is the identical
architecture with fresh weights, the "is it better than random" baseline; ``pretrained``
loads only the pretrained ViT (e.g. Prithvi) into fresh HPE/TET, the "does glacier SSL add
anything beyond the off-the-shelf encoder" baseline.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..backbone.backbone import GlacierBackbone
from ..backbone.tet import seasons_to_ids
from ..backbone.vit import load_pretrained
from ..config import Config
from ..data.augmentation import apply_view
from ..data.manifest import GlacierSequence, load_manifest, split_glaciers
from ..data.normalization import load_norm_stats
from ..data.preprocessing import Preprocessor
from ..data.sampling import (CropGeometry, RadiometricAug, SpatialAug, ViewSpec,
                             elapsed_years_from_dates)
from ..train.accel import amp_settings, autocast_ctx
from ..utils.logging_utils import get_logger

log = get_logger("eval")

THUMB = 32          # side of the stored per-timestamp thumbnail (pixel-change proxy, figures)
OCC_GRID = 8        # synthetic occlusion works on an 8x8 grid of square blocks


# --------------------------------------------------------------------------- checkpoint
def load_bundle(path: str | Path) -> dict:
    """Load only what evaluation needs from a training checkpoint (drops optimizer)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    bundle = {
        "cfg": Config(ck["config"]),
        "step": ck.get("step"),
        "epoch": (ck.get("extra") or {}).get("epoch"),
        "student": ck["student_backbone"],
        "teacher": ck["teacher_backbone"],
        "head2": (ck.get("heads") or {}).get("head2"),
    }
    del ck
    return bundle


def build_backbones(cfg: Config, bundle: Optional[dict], names: list[str],
                    device: torch.device) -> dict[str, GlacierBackbone]:
    out = {}
    for name in names:
        if name in ("random", "pretrained"):
            torch.manual_seed(int(cfg.run.seed))      # fixed, so the baseline is reproducible
            bb = GlacierBackbone(cfg)
            if name == "pretrained":
                path = cfg.get("vit.pretrained_ckpt", None)
                if not path or not Path(str(path)).exists():
                    raise FileNotFoundError(
                        f"'pretrained' baseline needs vit.pretrained_ckpt (got {path!r}); "
                        "pass --pretrained-ckpt")
                load_pretrained(bb.vit, str(path), cfg)
        else:
            if bundle is None:
                raise ValueError(f"model '{name}' needs a checkpoint")
            bb = GlacierBackbone(cfg)
            bb.load_state_dict(bundle[name])
        out[name] = bb.to(device).eval()
        for p in out[name].parameters():
            p.requires_grad_(False)
    return out


# --------------------------------------------------------------------------- sequences
def select_sequences(cfg: Config, project_root: Path, split: str,
                     limit: Optional[int] = None, seed: int = 0) -> list[GlacierSequence]:
    seqs = split_glaciers(load_manifest(project_root / str(cfg.data.manifest_path)), split)
    ids = sorted(seqs)
    if limit and limit < len(ids):
        rng = np.random.default_rng(seed)
        ids = sorted(rng.choice(ids, size=limit, replace=False).tolist())
    return [seqs[i] for i in ids]


def base_view(src: int) -> ViewSpec:
    """Identity crop of the full scene -- exactly the view Head 2 trains on."""
    return ViewSpec(geometry=CropGeometry(0, 0, src, src, src, "base"),
                    spatial=SpatialAug(), radiometric=RadiometricAug(), view_type="base")


def occlude(img: torch.Tensor, valid: torch.Tensor, fraction: float, seed: int):
    """Zero a random ``fraction`` of an 8x8 block grid, independently per timestamp.

    Zero is precisely how invalid (cloudy / nodata) pixels reach the model in training
    (``apply_view`` multiplies by the validity mask), so this simulates real cloud cover.
    """
    T, _, S, _ = img.shape
    block = max(1, S // OCC_GRID)
    cells = OCC_GRID * OCC_GRID
    k = int(round(fraction * cells))
    rng = np.random.default_rng(seed)
    occ = torch.zeros(T, S, S, dtype=torch.bool)
    for t in range(T):
        for c in rng.choice(cells, size=k, replace=False):
            r, col = divmod(int(c), OCC_GRID)
            occ[t, r * block:(r + 1) * block, col * block:(col + 1) * block] = True
    return img * (~occ).unsqueeze(1).float(), valid & ~occ


class GlacierEvalDataset(Dataset):
    """One item = one glacier's full chronological sequence (identity crop)."""

    def __init__(self, seqs: list[GlacierSequence], cfg: Config, pre: Preprocessor,
                 max_timesteps: Optional[int] = None, occlusion: float = 0.0,
                 occlusion_seed: int = 0):
        self.seqs, self.cfg, self.pre = seqs, cfg, pre
        self.cap = max_timesteps
        self.occlusion = float(occlusion)
        self.occ_seed = occlusion_seed
        self.view = base_view(int(cfg.data.image_size))

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx: int) -> dict:
        seq = self.seqs[idx]
        obs = seq.observations[: self.cap] if self.cap else seq.observations
        imgs, masks = [], []
        for o in obs:
            norm, valid = self.pre.process(o)
            imgs.append(torch.from_numpy(norm))
            masks.append(torch.from_numpy(valid))
        window = torch.stack(imgs)                       # (T,6,H,W)
        valid = torch.stack(masks)                       # (T,H,W)
        img, _ = apply_view(window, valid, self.view, self.cfg)
        if self.occlusion > 0:
            img, valid = occlude(img, valid, self.occlusion, self.occ_seed + idx)
        return make_item(seq, obs, img, valid)


def make_item(seq, obs, img: torch.Tensor, valid: torch.Tensor) -> dict:
    """Assemble the per-glacier record (also used by the unit tests with synthetic data)."""
    dates = [o.date_obj() for o in obs]
    vf = valid.float()
    pooled_v = F.adaptive_avg_pool2d(vf.unsqueeze(1), THUMB)                 # (T,1,t,t)
    pooled_i = F.adaptive_avg_pool2d(img * vf.unsqueeze(1), THUMB)           # (T,6,t,t)
    thumb = pooled_i / pooled_v.clamp(min=1e-6)
    cloud = np.array([np.nan if o.cloud_pct is None else float(o.cloud_pct) for o in obs])
    return {
        "meta": {
            "glims_id": seq.glims_id, "region": seq.region, "split": seq.split,
            "area_km2": seq.area_km2, "lon": seq.centroid[0], "lat": seq.centroid[1],
            "dates": [d.isoformat() for d in dates],
            "seasons": [o.season for o in obs],
            "years": np.array([o.year for o in obs]),
            "observation_ids": [o.observation_id for o in obs],
            "cloud_pct": cloud,
            "valid_frac": vf.mean(dim=(1, 2)).numpy(),
            "elapsed": np.array(elapsed_years_from_dates(dates), dtype=np.float32),
            "thumb": thumb.half(), "thumb_valid": (pooled_v.squeeze(1) > 0.5),
        },
        "imgs": img,
    }


# --------------------------------------------------------------------------- forward
def head2_key(cfg: Config) -> str:
    """Feature Head 2 must be scored on: the path it was TRAINED on. Checkpoints trained
    with the temporal firewall read a time-blind TET pass (older configs lack the key)."""
    return "timestamp_out_blind" if bool(cfg.get("head2.time_blind", False))         else "timestamp_out"


def run_tet(bb: GlacierBackbone, regional: torch.Tensor, seasons: list[str],
            elapsed: np.ndarray, device: torch.device, amp: tuple,
            time_blind: bool = False) -> dict:
    """TET over (1,T,R,D) regional tokens for ONE sequence. Dates drive all temporal
    quantities (elapsed measured from the first timestamp in THIS sequence); with
    ``time_blind`` the explicit temporal metadata is withheld (see TET.forward)."""
    T = regional.shape[1]
    season_ids = seasons_to_ids([seasons], T, device)
    el = torch.as_tensor(np.asarray(elapsed), dtype=torch.float32, device=device)[None]
    span = el[:, -1] - el[:, 0]
    tmask = torch.ones(1, T, dtype=torch.bool, device=device)
    enabled, dtype = amp
    with torch.no_grad(), autocast_ctx(device.type, dtype, enabled):
        return bb.tet(regional.to(device), season_ids, el, span, tmask,
                      time_blind=time_blind)


def encode_tokens(bb: GlacierBackbone, imgs: torch.Tensor, device: torch.device,
                  amp: tuple, chunk: int) -> torch.Tensor:
    """(1,T,6,H,W) -> (1,T,R,D). Timestamps are independent, so chunking is exact."""
    enabled, dtype = amp
    parts = []
    with torch.no_grad(), autocast_ctx(device.type, dtype, enabled):
        for s in range(0, imgs.shape[1], chunk):
            parts.append(bb.encode_regional(imgs[:, s:s + chunk].to(device)).float())
    return torch.cat(parts, dim=1)


def features_from_tokens(bb, tokens, seasons, elapsed, device, amp, keep_tokens,
                         blind: bool = False):
    out = run_tet(bb, tokens, seasons, elapsed, device, amp)
    rec = {
        "timestamp_out": out["timestamp_out"][0].float().cpu().numpy(),
        "sequence_out": out["sequence_out"][0].float().cpu().numpy(),
        "visual_pool": tokens[0].float().mean(dim=1).cpu().numpy(),
        "tokens": tokens[0].half().cpu() if keep_tokens else None,   # bool for this model
    }
    if blind:
        # the time-blind pass the firewalled Head 2 trained on (TET only; ViT is reused)
        ob = run_tet(bb, tokens, seasons, elapsed, device, amp, time_blind=True)
        rec["timestamp_out_blind"] = ob["timestamp_out"][0].float().cpu().numpy()
    return rec


def extract_all(backbones: dict[str, GlacierBackbone], items, device: torch.device,
                cfg: Config, keep_tokens=True, chunk: int = 12,
                n_total: Optional[int] = None):
    """Single pass over the glaciers; every model sees the same pixels.

    Returns ``(meta, feats)`` with ``meta[gid]`` and ``feats[model][gid]``.
    """
    amp = amp_settings(cfg.get("train.amp", False), cfg.get("train.amp_dtype", "float16"),
                       device)
    blind = bool(cfg.get("head2.time_blind", False))
    meta: dict[str, dict] = {}
    feats: dict[str, dict] = {n: {} for n in backbones}
    t0 = time.time()
    for k, item in enumerate(items):
        m = item["meta"]
        gid = m["glims_id"]
        meta[gid] = m
        imgs = item["imgs"].unsqueeze(0)
        for name, bb in backbones.items():
            tokens = encode_tokens(bb, imgs, device, amp, chunk)
            if isinstance(keep_tokens, (set, list, tuple)):
                keep = name in keep_tokens
            else:
                keep = bool(keep_tokens)
            feats[name][gid] = features_from_tokens(
                bb, tokens, m["seasons"], m["elapsed"], device, amp, keep, blind)
        if (k + 1) % 5 == 0 or k + 1 == n_total:
            rate = (time.time() - t0) / (k + 1)
            log.info("extracted %d/%s glaciers (%.1fs each, ~%.0f min left)", k + 1,
                     n_total if n_total else "?", rate,
                     (n_total - k - 1) * rate / 60 if n_total else float("nan"))
    return meta, feats


# --------------------------------------------------------------------------- caching
@dataclass
class FeatureSet:
    """Everything the tests read for one split."""
    split: str
    meta: dict
    feats: dict = field(default_factory=dict)       # model -> gid -> record

    @property
    def gids(self) -> list[str]:
        return sorted(self.meta)

    def n_timestamps(self) -> int:
        return int(sum(len(m["dates"]) for m in self.meta.values()))


def usable(m: dict, cfg: Config) -> np.ndarray:
    """Timestamps with enough valid pixels (data.min_valid_fraction), as in training."""
    return np.asarray(m["valid_frac"]) >= float(cfg.data.min_valid_fraction)


def build_loader(cfg: Config, project_root: Path, seqs, max_timesteps=None,
                 occlusion: float = 0.0, occlusion_seed: int = 0) -> DataLoader:
    norm = load_norm_stats(project_root / str(cfg.data.norm_stats_path))
    pre = Preprocessor(cfg, norm_stats=norm, project_root=project_root)
    ds = GlacierEvalDataset(seqs, cfg, pre, max_timesteps, occlusion, occlusion_seed)
    # Evaluation is compute-bound (the backbone costs several times the image read), and
    # worker processes inherit a parent already holding the checkpoint, which is what makes
    # them die on spawn platforms. Default to loading in-process; opt in with eval.workers.
    nw = int(cfg.get("eval.workers", 0))
    return DataLoader(ds, batch_size=None, shuffle=False, num_workers=nw,
                      persistent_workers=False)


def ensure_features(cache_dirs: list[Path], write_dir: Path, tag: str,
                    models: list[str], bundle, cfg: Config, project_root: Path,
                    device: torch.device, split: str, limit: Optional[int],
                    max_timesteps: Optional[int], keep_tokens, seed: int
                    ) -> FeatureSet:
    """Load cached features, extracting (once) whatever is missing."""
    def find(fname):
        for d in cache_dirs:
            p = Path(d) / fname
            if p.exists():
                return p
        return None

    meta_name = f"meta_{tag}.pt"
    meta_path = find(meta_name)
    fs = FeatureSet(split=split, meta=torch.load(meta_path, weights_only=False)
                    if meta_path else {})
    missing = []
    for name in models:
        p = find(f"feats_{name}_{tag}.pt")
        if p is not None and meta_path is not None:
            fs.feats[name] = torch.load(p, weights_only=False)
            log.info("cache hit: %s", p)
        else:
            missing.append(name)
    if not missing:
        return fs

    seqs = select_sequences(cfg, project_root, split, limit, seed)
    log.info("extracting %s for %d glaciers (%s split) ...", missing, len(seqs), split)
    backbones = build_backbones(cfg, bundle, missing, device)
    loader = build_loader(cfg, project_root, seqs, max_timesteps)
    meta, feats = extract_all(backbones, loader, device, cfg, keep_tokens,
                              n_total=len(seqs))
    write_dir.mkdir(parents=True, exist_ok=True)
    torch.save(meta, write_dir / meta_name)
    fs.meta = meta
    for name in missing:
        torch.save(feats[name], write_dir / f"feats_{name}_{tag}.pt")
        fs.feats[name] = feats[name]
    del backbones
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return fs
