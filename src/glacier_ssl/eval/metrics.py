"""Phase 14 — evaluation metrics and training-health diagnostics.

Head 2 regression metrics are reported in YEARS (converted back from log1p space). Head 3
and DINO diagnostics focus on training health (collapse/occupancy) since there are no
manual labels (head2 §10, head3 §13). Evaluation is on held-out glaciers only.
"""
from __future__ import annotations

import numpy as np
import torch


# ----------------------------------------------------------- Head 2 regression
def distance_regression_metrics(pred_years, target_years, gap_edges=(0.5, 1, 2, 4)):
    pred = np.asarray(pred_years, dtype=float)
    tgt = np.asarray(target_years, dtype=float)
    err = np.abs(pred - tgt)
    out = {
        "mae_years": float(err.mean()),
        "rmse_years": float(np.sqrt(((pred - tgt) ** 2).mean())),
        "spearman": _spearman(pred, tgt),
    }
    # per-gap-range MAE (short/medium/long) — a model can look good on frequent short gaps.
    bins = np.digitize(tgt, gap_edges)
    for b in np.unique(bins):
        m = bins == b
        out[f"mae_gapbin_{b}"] = float(err[m].mean()) if m.any() else float("nan")
    return out


def _spearman(a, b):
    if len(a) < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


# ----------------------------------------------------------- collapse diagnostics
def embedding_variance(embeddings: torch.Tensor) -> float:
    """Low per-dimension variance across samples is a collapse signal."""
    return float(embeddings.var(dim=0).mean())


def cosine_similarity_stats(embeddings: torch.Tensor) -> dict:
    z = torch.nn.functional.normalize(embeddings, dim=-1)
    sim = z @ z.t()
    n = sim.shape[0]
    off = sim[~torch.eye(n, dtype=torch.bool, device=z.device)]
    return {"mean_cos": float(off.mean()), "max_cos": float(off.max())}


def prototype_occupancy(assignments: torch.Tensor, num_prototypes: int) -> dict:
    """Occupancy + entropy of cluster/prototype assignments (head3 §13)."""
    counts = torch.bincount(assignments.flatten(), minlength=num_prototypes).float()
    probs = counts / counts.sum().clamp(min=1)
    entropy = -(probs * (probs + 1e-12).log()).sum()
    return {"occupied": int((counts > 0).sum()), "total": num_prototypes,
            "entropy": float(entropy), "max_entropy": float(np.log(num_prototypes))}
