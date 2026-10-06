"""Deterministic quantiles and the raw (naive in-line) estimator."""

from __future__ import annotations

import numpy as np

QUANTILES = (0.1, 0.5, 0.9)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q) -> np.ndarray:
    """Quantiles of a weighted sample, by linear interpolation of the weighted CDF.

    Each point sits at the midpoint of its own weight, so equal weights reproduce the
    usual sample quantile closely and the result never depends on random tie-breaking.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values, kind="stable")
    v, w = values[order], weights[order]
    cdf = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(q, cdf, v)


def summarise(values: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    """mean, d10, d50, d90 and CV of a (weighted) thickness sample."""
    if weights is None:
        weights = np.ones_like(values, dtype=float)
    total = weights.sum()
    mean = float(np.sum(weights * values) / total)
    var = float(np.sum(weights * (values - mean) ** 2) / total)
    d10, d50, d90 = weighted_quantile(values, weights, QUANTILES)
    return {
        "mean": mean,
        "d10": float(d10),
        "d50": float(d50),
        "d90": float(d90),
        "cv": float(np.sqrt(var) / mean) if mean > 0 else float("nan"),
    }


def raw_estimate(thickness_um: np.ndarray) -> dict[str, float]:
    """Unweighted summary of the window: what a naive in-line system reports."""
    return summarise(np.asarray(thickness_um, dtype=float))
