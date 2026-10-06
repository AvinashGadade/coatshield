"""Illustrative dissolution projection: coating thickness -> release time t63.

A straight line through the published pellet data (configs/default.yaml, dissolution).
It is an illustration of what a thickness distribution could mean for release, not a
dissolution model, and it is only supported inside the published thickness range.
"""

from __future__ import annotations

import numpy as np

from coatshield.config import Config

LABEL = "Illustrative only: linear fit to published pellet data, not a dissolution model"


def fit_line(cfg: Config) -> tuple[float, float]:
    """(slope in min per um, intercept in min) of t63 against thickness."""
    slope, intercept = np.polyfit(cfg.dissolution.thickness_um, cfg.dissolution.t63_min, 1)
    return float(slope), float(intercept)


def supported_range(cfg: Config) -> tuple[float, float]:
    return min(cfg.dissolution.thickness_um), max(cfg.dissolution.thickness_um)


def t63_minutes(thickness_um, cfg: Config) -> np.ndarray:
    slope, intercept = fit_line(cfg)
    return slope * np.asarray(thickness_um, dtype=float) + intercept


def in_supported_range(thickness_um, cfg: Config) -> np.ndarray:
    lo, hi = supported_range(cfg)
    t = np.asarray(thickness_um, dtype=float)
    return (t >= lo) & (t <= hi)


def project(thickness_edges_um: np.ndarray, counts: np.ndarray, cfg: Config) -> dict:
    """Release-time distribution for a thickness histogram, with the share outside the range."""
    centres = 0.5 * (thickness_edges_um[:-1] + thickness_edges_um[1:])
    share = counts / counts.sum()
    inside = in_supported_range(centres, cfg)
    return {
        "t63_min": t63_minutes(centres, cfg),
        "share": share,
        "inside": inside,
        "share_outside_range": float(share[~inside].sum()),
        "label": LABEL,
    }
