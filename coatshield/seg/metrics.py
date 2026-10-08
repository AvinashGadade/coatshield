"""Segmentation metrics: boundary error, Dice per class, thinnest separable film."""

from __future__ import annotations

import numpy as np


def boundary_error(found: np.ndarray, truth: np.ndarray, valid: np.ndarray,
                   px_um: float | None = None) -> dict[str, float]:
    """Mean absolute error of a surface over the valid columns, in pixels (and um)."""
    ok = valid & np.isfinite(truth)
    if not ok.any():
        return {"mae_px": float("nan"), "mae_um": float("nan"), "n": 0}
    mae = float(np.mean(np.abs(found[ok] - truth[ok])))
    return {"mae_px": mae, "mae_um": mae * px_um if px_um else float("nan"), "n": int(ok.sum())}


def dice_per_class(pred: np.ndarray, truth: np.ndarray, n_classes: int,
                   weight: np.ndarray | None = None) -> np.ndarray:
    """Dice coefficient of each class between two label maps (NaN for an absent class)."""
    w = np.ones(pred.shape, bool) if weight is None else weight.astype(bool)
    out = np.full(n_classes, np.nan)
    for k in range(n_classes):
        p, t = (pred == k) & w, (truth == k) & w
        size = p.sum() + t.sum()
        if size:
            out[k] = 2.0 * (p & t).sum() / size
    return out


def surfaces_separate(found_outer, found_inner, true_outer, true_inner, valid,
                      min_px: float, tolerance: float) -> bool:
    """True when two surfaces are found and their gap matches the true optical thickness.

    The found gap (median over valid columns) must be at least min_px and within
    max(min_px, tolerance x true gap) of the true gap.
    """
    ok = valid & np.isfinite(true_outer) & np.isfinite(true_inner)
    if not ok.any():
        return False
    gap = float(np.median((found_inner - found_outer)[ok]))
    true_gap = float(np.median((true_inner - true_outer)[ok]))
    return bool(gap >= min_px and abs(gap - true_gap) <= max(min_px, tolerance * true_gap))


def thinnest_separable(thickness_um: np.ndarray, separated: np.ndarray, rate: float = 0.9,
                       n_bins: int = 12) -> float:
    """Thinnest film (um) from which the two surfaces separate in at least `rate` of scans.

    Scans are binned by true thickness on a log scale; the answer is the lower edge of
    the lowest bin from which every bin clears the rate.
    """
    t = np.asarray(thickness_um, dtype=float)
    edges = np.exp(np.linspace(np.log(t.min()), np.log(t.max()) + 1e-9, n_bins + 1))
    which = np.clip(np.digitize(t, edges) - 1, 0, n_bins - 1)
    ok = np.array([separated[which == b].mean() >= rate if (which == b).any() else True
                   for b in range(n_bins)])
    failing = np.flatnonzero(~ok)
    if failing.size == 0:
        return float(edges[0])
    if failing[-1] == n_bins - 1:
        return float("nan")
    return float(edges[failing[-1] + 1])
