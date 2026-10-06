"""Per-pellet solver: surfaces -> radius, refractive index estimates, thickness."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coatshield.config import Config
from coatshield.solve.ellipse import Circle, fit_circle
from coatshield.solve.index import (
    _mean_db,
    index_from_ratio,
    index_from_reflectance,
    peak_db,
)
from coatshield.solve.refraction import incidence_angle, thickness_along_normal, thickness_vertical


@dataclass(frozen=True)
class PelletSurfaces:
    """Geometry of one scan in physical units, from the two surfaces found in the image."""

    x_um: np.ndarray  # lateral position of the A-scans used
    theta: np.ndarray  # incidence angle at those A-scans (rad)
    optical_um: np.ndarray  # optical path between the two surfaces
    circle: Circle  # fit to the outer surface (depth in air)
    used: np.ndarray  # mask over the scan's columns: valid and below the angle limit
    n_reflectance: float  # method A at the apex (NaN when it cannot be read)
    n_ratio: float  # method A, ratio variant


@dataclass(frozen=True)
class PelletThickness:
    thickness_um: float  # mean over the valid A-scans, along the surface normal
    uncertainty_um: float  # standard error across A-scans plus the share from the index
    intra_cv: float  # coefficient of variation of thickness across the A-scans
    radius_um: float
    fit_residual_um: float  # rms distance of the outer surface from the fitted circle
    n_ascans: int
    optical_um: float  # mean optical thickness at the apex, for the fusion method


def measure_surfaces(db: np.ndarray, outer_px: np.ndarray, inner_px: np.ndarray,
                     valid: np.ndarray, x_um: np.ndarray, depth_px_um: float,
                     crop_start_px: int, reflector_db: float, cfg: Config,
                     calibration_error: float = 0.0) -> PelletSurfaces | None:
    """Fit the outer surface and read the index-bearing peak levels of one scan.

    db: [A-scan, depth] in dB above the noise floor; outer_px, inner_px: surface rows
    per A-scan (from the boundary finder); valid: columns with usable signal.
    """
    sc = cfg.solve
    ok = valid & np.isfinite(outer_px) & np.isfinite(inner_px)
    if ok.sum() < sc.min_ascans:
        return None
    z_outer = (outer_px + crop_start_px) * depth_px_um  # depth in air: true geometric depth
    circle = fit_circle(x_um[ok], z_outer[ok])
    theta_all = incidence_angle(x_um, circle)
    used = ok & (np.abs(theta_all) <= np.radians(sc.max_angle_deg))
    if used.sum() < sc.min_ascans:
        return None

    apex = used & (np.abs(theta_all) <= np.radians(sc.apex_angle_deg))
    n_reflectance = n_ratio = float("nan")
    if apex.any():
        surface = _mean_db(peak_db(db[apex], outer_px[apex], sc.search_px))
        inner = _mean_db(peak_db(db[apex], inner_px[apex], sc.search_px))
        depth = float(np.mean(z_outer[apex]))
        n_reflectance = index_from_reflectance(surface, reflector_db, depth, cfg,
                                               calibration_error)
        n_ratio = index_from_ratio(inner, surface, cfg)
    return PelletSurfaces(
        x_um=x_um[used],
        theta=theta_all[used],
        optical_um=(inner_px[used] - outer_px[used]) * depth_px_um,
        circle=circle,
        used=used,
        n_reflectance=n_reflectance,
        n_ratio=n_ratio,
    )


def pellet_thickness(surfaces: PelletSurfaces, n: float, n_standard_error: float = 0.0,
                     refraction: bool = True) -> PelletThickness:
    """Thickness of one pellet with the pooled index n."""
    if refraction:
        t = thickness_along_normal(surfaces.optical_um, surfaces.theta, surfaces.circle.radius, n)
    else:
        t = thickness_vertical(surfaces.optical_um, n)
    mean = float(t.mean())
    spread = float(t.std(ddof=1))
    from_ascans = spread / np.sqrt(t.size)
    from_index = mean * n_standard_error / n  # thickness scales as 1 / n
    near_apex = np.abs(surfaces.theta) <= np.abs(surfaces.theta).min() + np.radians(2.0)
    return PelletThickness(
        thickness_um=mean,
        uncertainty_um=float(np.hypot(from_ascans, from_index)),
        intra_cv=spread / mean if mean > 0 else float("nan"),
        radius_um=surfaces.circle.radius,
        fit_residual_um=surfaces.circle.residual_rms,
        n_ascans=int(t.size),
        optical_um=float(surfaces.optical_um[near_apex].mean()),
    )
