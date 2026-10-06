"""Speckle statistics for comparing synthetic scans with real OCT (the Zenodo skin scans).

Only dimensionless statistics are compared, because pixel sizes, resolution and the
dB-per-grey-level scale differ between instruments: the shape of the log-intensity
distribution inside a scattering region (after removing the local mean), and the
speckle grain size along each axis measured in pixels and as a ratio.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.stats import ks_2samp, kurtosis, skew

_DETREND_PX = 15  # box used to remove the slow change of mean brightness with depth
_MAX_LAG = 12


def _residual(log_image: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """Log intensity minus its local mean inside the region of interest."""
    weight = uniform_filter(roi.astype(float), _DETREND_PX, mode="constant")
    local = uniform_filter(np.where(roi, log_image, 0.0), _DETREND_PX, mode="constant")
    return log_image - local / np.maximum(weight, 1e-9)


def _grain_px(residual: np.ndarray, roi: np.ndarray, axis: int) -> float:
    """Speckle grain size: full width at half maximum of the autocorrelation, in pixels."""
    r = np.where(roi, residual, 0.0)
    corr = []
    for lag in range(_MAX_LAG + 1):
        a = np.take(r, range(0, r.shape[axis] - lag), axis=axis)
        b = np.take(r, range(lag, r.shape[axis]), axis=axis)
        both = (np.take(roi, range(0, r.shape[axis] - lag), axis=axis)
                & np.take(roi, range(lag, r.shape[axis]), axis=axis))
        corr.append(float((a * b)[both].mean()) if both.any() else 0.0)
    corr = np.array(corr) / corr[0]
    below = np.flatnonzero(corr < 0.5)
    if below.size == 0:
        return float(2 * _MAX_LAG)
    i = below[0]
    return float(2.0 * (i - 1 + (corr[i - 1] - 0.5) / (corr[i - 1] - corr[i])))


def speckle_stats(log_image: np.ndarray, roi: np.ndarray) -> dict:
    """Statistics of a scattering region. log_image: any linear function of log intensity,
    array [depth, lateral]; roi: boolean mask of the region."""
    log_image = np.asarray(log_image, dtype=float)
    res = _residual(log_image, roi)
    values = res[roi]
    std = values.std()
    axial, lateral = _grain_px(res, roi, 0), _grain_px(res, roi, 1)
    return {
        "n_pixels": int(roi.sum()),
        "log_std": float(std),
        "skewness": float(skew(values)),
        "excess_kurtosis": float(kurtosis(values)),
        "grain_axial_px": axial,
        "grain_lateral_px": lateral,
        "grain_ratio": lateral / axial,
        "standardised": (values - values.mean()) / std,
    }


def linear_contrast(db: np.ndarray, roi: np.ndarray) -> float:
    """Speckle contrast (std / mean of linear intensity) of a region with known dB scale.

    Fully developed speckle gives 1; averaging A-scans lowers it.
    """
    res = _residual(np.asarray(db, dtype=float), roi)[roi]
    linear = 10.0 ** (res / 10.0)
    return float(linear.std() / linear.mean())


def bright_band_roi(image: np.ndarray, share: float = 0.6) -> np.ndarray:
    """Region of a real scan with strong scattering: rows whose mean is near the brightest."""
    rows = image.mean(axis=1)
    keep = rows > share * rows.max()
    return np.repeat(keep[:, None], image.shape[1], axis=1)


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance between two standardised log-intensity samples."""
    return float(ks_2samp(a, b).statistic)
