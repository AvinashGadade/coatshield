"""Three independent ways to find the coating's refractive index.

A: reflectance. The air-coating reflection, calibrated with the on-window reference
   reflector, inverted through the Fresnel equation; and a ratio variant that needs no
   absolute calibration but a known core index.
B: camera-OCT fusion. Mean optical thickness from OCT over mean physical thickness
   from the growth of the camera's mean diameter.
C: at-line anchor. Optical thickness against microscopy cross-sections of a few pellets.
The index belongs to the coating formulation, so estimates are pooled across pellets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from coatshield.config import Config
from coatshield.oct import physics
from coatshield.oct.preprocess import peak_db, sensitivity_db  # noqa: F401

N_SEARCH = (1.05, 2.5)  # range in which a coating index is sought


def _mean_db(levels_db: np.ndarray) -> float:
    """Average of dB levels in the power domain."""
    return float(10.0 * np.log10(np.mean(10.0 ** (np.asarray(levels_db) / 10.0))))


def index_from_reflectance(surface_db: float, reflector_db: float, surface_depth_um: float,
                           cfg: Config, calibration_error: float = 0.0) -> float:
    """Method A. Fresnel inversion of the surface reflection at the apex.

    Both returns cross the window twice, so fouling cancels in their ratio; the
    instrument's calibrated depth roll-off is divided out. calibration_error is a relative
    error on the reference reflector's assumed amplitude (for the sensitivity study).
    """
    oc = cfg.oct
    rolloff_db = sensitivity_db(oc, oc.reflector_um) - sensitivity_db(oc, surface_depth_um)
    r = (np.sqrt(oc.reflector_reflectance) * (1.0 + calibration_error)
         * 10.0 ** ((surface_db - reflector_db + rolloff_db) / 20.0))
    return float((1.0 + r) / (1.0 - r))


def index_from_ratio(inner_db: float, surface_db: float, cfg: Config) -> float:
    """Method A, ratio variant: coating-core peak against air-coating peak, known core index.

    Needs no absolute calibration. Attenuation in the coating is neglected, and the
    coating index is sought on one side of the core index (solve.core_branch).
    """
    n_core = cfg.core.n
    ratio = 10.0 ** ((inner_db - surface_db) / 20.0)

    def mismatch(n: float) -> float:
        r_s = abs(physics.fresnel_amplitude(1.0, n))
        return abs(physics.fresnel_amplitude(n, n_core)) * (1.0 - r_s**2) / r_s - ratio

    lo, hi = (N_SEARCH[0], n_core) if cfg.solve.core_branch == "below" else (n_core, N_SEARCH[1])
    if mismatch(lo) * mismatch(hi) > 0:
        return float("nan")
    return float(brentq(mismatch, lo, hi))


def index_from_fusion(mean_optical_um: float, mean_coated_diameter_um: float,
                      mean_core_diameter_um: float) -> float:
    """Method B. Mean optical thickness over the mean physical thickness seen by the camera.

    Half the growth of the mean diameter is the mean physical thickness; both diameters
    come from the same camera, so its scale error cancels, and both means should come
    from the debiased estimator so the window's size bias does not leak in.
    """
    physical = 0.5 * (mean_coated_diameter_um - mean_core_diameter_um)
    return float(mean_optical_um / physical) if physical > 0 else float("nan")


def index_from_anchor(optical_um: np.ndarray, microscopy_um: np.ndarray) -> float:
    """Method C. One-time anchor: OCT optical thickness against microscopy cross-sections."""
    return float(np.sum(optical_um) / np.sum(microscopy_um))


@dataclass(frozen=True)
class PooledIndex:
    n: float
    standard_error: float
    count: int


def pool_index(values: np.ndarray) -> PooledIndex:
    """Pool per-pellet estimates of one time window: median, with a robust standard error.

    Precision improves with the square root of the number of pellets.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return PooledIndex(float("nan"), float("nan"), 0)
    spread = 1.4826 * np.median(np.abs(v - np.median(v)))
    # 1.2533 = sqrt(pi / 2): standard error of a median relative to that of a mean.
    return PooledIndex(float(np.median(v)), float(1.2533 * spread / np.sqrt(v.size)), int(v.size))


def assumed_index_error(n_true: float, n_assumed: float) -> float:
    """Relative thickness error from assuming an index: thickness reads n_true / n_assumed."""
    return n_true / n_assumed - 1.0
