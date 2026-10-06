"""From optical path to physical thickness along the surface normal, with Snell refraction."""

from __future__ import annotations

import numpy as np

from coatshield.solve.ellipse import Circle


def incidence_angle(x_um: np.ndarray, circle: Circle) -> np.ndarray:
    """Angle between the vertical beam and the surface normal at each A-scan (rad)."""
    return np.arcsin(np.clip((np.asarray(x_um) - circle.xc) / circle.radius, -1.0, 1.0))


def thickness_along_normal(optical_um: np.ndarray, theta: np.ndarray, radius_um: float,
                           n: float) -> np.ndarray:
    """Coating thickness along the normal of a coated sphere.

    optical_um is the optical-path difference between the two surfaces along the beam.
    The beam refracts at the surface (Snell), runs a length optical / n inside the
    coating, and meets the core, a concentric sphere; the thickness is the difference
    of the two radii.
    """
    sin_r = np.sin(theta) / n
    cos_r = np.sqrt(1.0 - sin_r**2)
    path = np.asarray(optical_um) / n
    core_radius = np.sqrt((radius_um * cos_r - path) ** 2 + (radius_um * sin_r) ** 2)
    return radius_um - core_radius


def thickness_vertical(optical_um: np.ndarray, n: float) -> np.ndarray:
    """Thickness without the refraction correction: optical path over index."""
    return np.asarray(optical_um) / n
