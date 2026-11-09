"""Small statistics helpers shared by the evaluation tests (no torch dependency)."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def spearman(a, b) -> float:
    """Spearman rho with average ranks for ties (gaps/ids are heavily tied)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def partial_spearman(a, b, control) -> float:
    """Spearman(a, b) after regressing the rank of ``control`` out of both ranks."""
    a, b, c = (np.asarray(x, dtype=float) for x in (a, b, control))
    if len(a) < 4 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)
    if np.ptp(rc) == 0:
        return float(np.corrcoef(ra, rb)[0, 1])
    X = np.stack([np.ones_like(rc), rc], axis=1)
    res_a = ra - X @ np.linalg.lstsq(X, ra, rcond=None)[0]
    res_b = rb - X @ np.linalg.lstsq(X, rb, rcond=None)[0]
    if res_a.std() == 0 or res_b.std() == 0:
        return float("nan")
    return float(np.corrcoef(res_a, res_b)[0, 1])


def cluster_bootstrap_ratio(num, den, n_boot: int = 1000, seed: int = 0,
                            alpha: float = 0.05) -> tuple[float, float, float]:
    """Point estimate and CI of sum(num)/sum(den), resampling whole groups (glaciers).

    ``num``/``den`` are per-group sums (e.g. per-glacier absolute-error sum and pair
    count). Resampling at the glacier level is the honest unit: pairs drawn from one
    glacier are not independent.
    """
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    point = float(num.sum() / max(den.sum(), 1e-12))
    if len(num) < 2:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    vals = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def mean_ci(x, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(x) == 1:
        return float(x[0]), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    m = x[idx].mean(1)
    lo, hi = np.quantile(m, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def effective_rank(x: np.ndarray) -> float:
    """RankMe-style effective rank: exp(entropy of the normalised singular values) of the
    UNCENTERED embedding matrix (Garrido et al. 2023).

    ~1 means every embedding points the same way (collapse, however small the noise on top
    of it); ~min(N, D) means the embeddings use all their dimensions. Centering is
    deliberately NOT applied: a collapsed model emits one vector plus tiny noise, and
    centering would turn that noise into an isotropic cloud that reads as full rank.
    """
    s = np.linalg.svd(np.asarray(x, dtype=np.float64), compute_uv=False)
    if s.sum() <= 0:
        return 1.0
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def cosine_dist_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise 1 - cos(a_i, b_i)."""
    return 1.0 - (l2n(a) * l2n(b)).sum(-1)


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
