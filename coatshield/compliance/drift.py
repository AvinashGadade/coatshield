"""Drift monitor: is the instrument still seeing what it saw when the model was validated?

Each scan gives a feature vector. It is compared with a clean baseline by Hotelling's T2
(99% control limit); the undecided rate and the reference reflector's level are tracked
with moving averages. The state is OK, Warning or Alarm. A second check flags products
whose parameters fall outside the ranges the boundary finder was trained on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import f as f_dist

from coatshield.config import Config

FEATURES = ("mean_db", "noise_floor_db", "surface_snr_db", "reflector_db", "core_contrast",
            "segmentation_entropy", "gate_confidence")
OK, WARNING, ALARM = "OK", "Warning", "Alarm"
_RANK = {OK: 0, WARNING: 1, ALARM: 2}


def scan_features(db: np.ndarray, noise_floor: float, reflector_db: float,
                  prob: np.ndarray | None, core_mask: np.ndarray | None,
                  gate_confidence: float) -> np.ndarray:
    """Feature vector of one scan, in the order of FEATURES.

    db: [A-scan, depth] dB above the noise floor; prob: class probabilities
    [3, depth, A-scan] or None; core_mask: boolean [A-scan, depth] of the core region.
    """
    contrast = 0.0
    if core_mask is not None and core_mask.any():
        linear = 10.0 ** (db[core_mask] / 10.0)
        contrast = float(linear.std() / linear.mean())
    entropy = 0.0
    if prob is not None:
        entropy = float(-(prob * np.log(prob + 1e-9)).sum(axis=0).mean())
    return np.array([float(db.mean()), 10.0 * np.log10(noise_floor), float(db.max()),
                     reflector_db, contrast, entropy, gate_confidence])


@dataclass(frozen=True)
class Baseline:
    mean: np.ndarray
    cov_inv: np.ndarray
    t2_limit: float
    reflector_db: float
    n: int


def fit_baseline(features: np.ndarray, cfg: Config) -> Baseline:
    """Baseline from clean scans [n, features]; the T2 limit is for a new single scan."""
    x = np.asarray(features, dtype=float)
    n, p = x.shape
    if n < max(cfg.drift.min_baseline_scans, p + 2):
        raise ValueError(f"need at least {cfg.drift.min_baseline_scans} clean scans, got {n}")
    cov = np.cov(x, rowvar=False)
    cov += np.eye(p) * 1e-9 * np.trace(cov) / p  # keep it invertible if a feature is constant
    limit = (p * (n + 1) * (n - 1) / (n * (n - p))
             * f_dist.ppf(cfg.drift.t2_confidence, p, n - p))
    return Baseline(x.mean(axis=0), np.linalg.inv(cov), float(limit),
                    float(x[:, FEATURES.index("reflector_db")].mean()), n)


def hotelling_t2(features: np.ndarray, baseline: Baseline) -> np.ndarray:
    d = np.atleast_2d(features) - baseline.mean
    return np.einsum("ni,ij,nj->n", d, baseline.cov_inv, d)


def ewma(values: np.ndarray, lam: float, start: float = 0.0) -> np.ndarray:
    out = np.empty(len(values))
    level = start
    for i, v in enumerate(values):
        level = lam * v + (1.0 - lam) * level
        out[i] = level
    return out


def monitor(features: np.ndarray, undecided: np.ndarray, baseline: Baseline,
            cfg: Config) -> dict[str, np.ndarray]:
    """Drift state after each scan of a sequence.

    features: [n, len(FEATURES)]; undecided: [n] booleans. Returns T2, the three moving
    averages and the state per scan.
    """
    dc = cfg.drift
    t2 = hotelling_t2(features, baseline)
    exceed = ewma((t2 > baseline.t2_limit).astype(float), dc.ewma_lambda,
                  1.0 - dc.t2_confidence)
    undecided_rate = ewma(np.asarray(undecided, dtype=float), dc.ewma_lambda)
    reflector = np.asarray(features)[:, FEATURES.index("reflector_db")]
    drop = baseline.reflector_db - ewma(reflector, dc.ewma_lambda, baseline.reflector_db)

    def level(value, warn, alarm):
        return np.where(value >= alarm, 2, np.where(value >= warn, 1, 0))

    rank = np.maximum.reduce([level(exceed, dc.exceed_warning, dc.exceed_alarm),
                              level(undecided_rate, dc.undecided_warning, dc.undecided_alarm),
                              level(drop, dc.reflector_warning_db, dc.reflector_alarm_db)])
    names = np.array([OK, WARNING, ALARM])
    return {"t2": t2, "t2_limit": np.full(len(t2), baseline.t2_limit), "exceed_rate": exceed,
            "undecided_rate": undecided_rate, "reflector_drop_db": drop, "state": names[rank]}


def worst(states) -> str:
    return max(states, key=_RANK.__getitem__) if len(states) else OK


def out_of_training_range(cfg: Config, **product) -> list[str]:
    """Product parameters outside the ranges the boundary finder was trained on.

    Accepts any of thickness_um, n_coat, n_core, radius_um, speed_m_s, snr_db.
    """
    ranges = cfg.oct_dataset.model_dump()
    problems = []
    for name, value in product.items():
        lo, hi = ranges[name]
        if not lo <= value <= hi:
            problems.append(f"{name} = {value:g} is outside the training range {lo:g} to {hi:g}")
    return problems
