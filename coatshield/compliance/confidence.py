"""Confidence of one thickness reading, and the "undecided" state.

score = gate confidence x segmentation confidence x fit confidence. Below the threshold
(chain.confidence_min, tuned on validation data) the pellet is undecided: it is left out
of the batch estimate and counted.
"""

from __future__ import annotations

import numpy as np

from coatshield.config import Config


def segmentation_confidence(margin_outer: np.ndarray, margin_inner: np.ndarray,
                            used: np.ndarray) -> float:
    """Mean probability margin along both graph-search paths, over the A-scans used."""
    if not used.any():
        return 0.0
    return float(np.mean(np.minimum(margin_outer[used], margin_inner[used])))


def fit_confidence(fit_residual_um: float, intra_cv: float, cfg: Config) -> float:
    """High when the outer surface is a clean circle and thickness agrees across A-scans."""
    cc = cfg.chain
    return float(np.exp(-((fit_residual_um / cc.fit_residual_scale_um) ** 2))
                 * np.exp(-((intra_cv / cc.intra_cv_scale) ** 2)))


def fuse(gate_confidence: float, seg_confidence: float, fit_conf: float) -> float:
    return float(gate_confidence * seg_confidence * fit_conf)


def is_decided(score: float, cfg: Config) -> bool:
    return score >= cfg.chain.confidence_min


def tune_threshold(scores: np.ndarray, abs_error_um: np.ndarray, cfg: Config) -> dict:
    """Lowest threshold at which at most max_error_share of accepted pellets are off by
    more than max_error_um; returns it with the share of pellets it leaves undecided."""
    cc = cfg.chain
    order = np.argsort(scores)[::-1]  # most confident first
    bad = (abs_error_um[order] > cc.max_error_um).astype(float)
    share_bad = np.cumsum(bad) / np.arange(1, bad.size + 1)
    ok = np.flatnonzero(share_bad <= cc.max_error_share)
    if ok.size == 0:
        return {"confidence_min": 1.0, "undecided_share": 1.0, "accepted_bad_share": 0.0}
    keep = int(ok[-1]) + 1  # accept the `keep` most confident pellets
    return {"confidence_min": float(scores[order][keep - 1]),
            "undecided_share": float(1.0 - keep / scores.size),
            "accepted_bad_share": float(share_bad[keep - 1])}
