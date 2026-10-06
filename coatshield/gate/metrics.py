"""Gate metrics shared by the classical baseline and the CNN."""

from __future__ import annotations

import numpy as np

from coatshield.gate.silhouettes import CLASSES, SINGLE, TOUCHING, TWIN


def confusion_matrix(labels: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """[true class, predicted class] counts."""
    n = len(CLASSES)
    return np.bincount(labels * n + predicted, minlength=n * n).reshape(n, n)


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, n_bins: int) -> float:
    """Mean gap between confidence and accuracy over equal-width confidence bins."""
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        sel = bins == b
        if sel.any():
            total += sel.mean() * abs(confidence[sel].mean() - correct[sel].mean())
    return float(total)


def gate_summary(labels: np.ndarray, predicted: np.ndarray, confidence: np.ndarray,
                 single_confidence_min: float, n_bins: int) -> dict:
    """The numbers the gate is judged on.

    twin_recall: twins not passed to OCT as a confident single.
    false_twin_rate: single pellets called twin or touching.
    twin_leak: twins passed as a confident single (the complement of recall).
    """
    passes = (predicted == SINGLE) & (confidence >= single_confidence_min)
    twins, singles = labels == TWIN, labels == SINGLE
    return {
        "accuracy": float((predicted == labels).mean()),
        "twin_recall": float(1.0 - passes[twins].mean()),
        "twin_leak": float(passes[twins].mean()),
        "false_twin_rate": float(np.isin(predicted[singles], (TWIN, TOUCHING)).mean()),
        "single_pass_rate": float(passes[singles].mean()),
        "non_single_leak": float(passes[~singles].mean()),
        "ece": expected_calibration_error(confidence, predicted == labels, n_bins),
    }
