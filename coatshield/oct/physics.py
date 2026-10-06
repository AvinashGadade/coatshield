"""Spectral-domain OCT physics for a coated sphere crossing a fixed beam.

Depths are optical path lengths in air, in micrometres, from the zero-delay plane.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from coatshield.config import OctCfg

UM = 1e-6


@dataclass(frozen=True)
class Spectrometer:
    """Pixel grid of the spectrometer and the even-wavenumber grid the FFT needs."""

    wavelength_um: np.ndarray  # [P] evenly spaced in wavelength, ascending
    k: np.ndarray  # [P] wavenumber 2 pi / lambda in rad/um (descending)
    k_lin: np.ndarray  # [P] evenly spaced wavenumbers, ascending, same range
    k0: float  # wavenumber at the centre wavelength
    source: np.ndarray  # [P] source power spectrum at the pixels (peak 1)
    n_fft: int
    depth_px_um: float  # optical depth of one FFT bin
    z_max_um: float  # depth at which the fringe period equals two pixels (mid-band)

    @property
    def dk(self) -> float:
        return float(self.k_lin[1] - self.k_lin[0])

    @property
    def half_range(self) -> float:
        return 0.5 * float(self.k_lin[-1] - self.k_lin[0])

    def band_coordinate(self, k: np.ndarray) -> np.ndarray:
        """Wavenumber mapped to [-1, 1] across the band (for the dispersion polynomial)."""
        return (k - 0.5 * (self.k_lin[0] + self.k_lin[-1])) / self.half_range


@lru_cache(maxsize=8)
def _spectrometer(center_nm, fwhm_nm, span_nm, n_pixels, zero_pad) -> Spectrometer:
    lam = np.linspace(center_nm - span_nm / 2, center_nm + span_nm / 2, n_pixels) / 1000.0
    k = 2.0 * np.pi / lam
    k_lin = np.linspace(k[-1], k[0], n_pixels)
    source = np.exp(-4.0 * np.log(2.0) * ((lam * 1000.0 - center_nm) / fwhm_nm) ** 2)
    n_fft = zero_pad * n_pixels
    dk = k_lin[1] - k_lin[0]
    d_lam = lam[1] - lam[0]
    return Spectrometer(
        wavelength_um=lam,
        k=k,
        k_lin=k_lin,
        k0=2.0 * np.pi / (center_nm / 1000.0),
        source=source,
        n_fft=n_fft,
        depth_px_um=float(np.pi / (n_fft * dk)),
        z_max_um=float((center_nm / 1000.0) ** 2 / (4.0 * d_lam)),
    )


def spectrometer(oc: OctCfg) -> Spectrometer:
    return _spectrometer(oc.center_nm, oc.fwhm_nm, oc.span_nm, oc.n_pixels, oc.zero_pad)


def axial_resolution_um(center_nm: float, fwhm_nm: float) -> float:
    """FWHM of the axial point-spread function in air for a Gaussian source."""
    return 0.44 * (center_nm / 1000.0) ** 2 / (fwhm_nm / 1000.0)


def fresnel_amplitude(n1: float, n2: float) -> float:
    """Amplitude reflection coefficient at normal incidence."""
    return (n1 - n2) / (n1 + n2)


@dataclass(frozen=True)
class SphereGeometry:
    """A coated sphere under a vertical beam, at lateral offsets x from its apex."""

    inside: np.ndarray  # beam hits the pellet and the refracted ray reaches the core
    sag_um: np.ndarray  # surface depth below the apex
    theta: np.ndarray  # incidence angle on the outer surface (rad)
    theta_core: np.ndarray  # incidence angle of the refracted ray on the core (rad)
    path_um: np.ndarray  # geometric length of the refracted ray inside the coating


def sphere_geometry(x_um: np.ndarray, radius_um: float, thickness_um: float,
                    n_coat: float) -> SphereGeometry:
    r_core = radius_um - thickness_um
    sin_t = np.clip(x_um / radius_um, -1.0, 1.0)
    sin_r = sin_t / n_coat  # Snell: refracted angle inside the coating
    reach = (np.abs(x_um) < radius_um) & (radius_um * np.abs(sin_r) < r_core)
    safe_r = np.where(reach, sin_r, 0.0)
    cos_r = np.sqrt(1.0 - safe_r**2)
    path = radius_um * cos_r - np.sqrt(np.maximum(r_core**2 - (radius_um * safe_r) ** 2, 0.0))
    sag = radius_um - np.sqrt(np.maximum(radius_um**2 - x_um**2, 0.0))
    return SphereGeometry(
        inside=reach,
        sag_um=np.where(reach, sag, np.nan),
        theta=np.arcsin(sin_t),
        theta_core=np.arcsin(np.clip(radius_um * safe_r / r_core, -1.0, 1.0)),
        path_um=np.where(reach, path, np.nan),
    )


def angular_falloff(theta: np.ndarray, falloff_deg: float) -> np.ndarray:
    """Share of an interface's reflection that returns to the fibre at incidence theta."""
    return np.exp(-((np.degrees(theta) / falloff_deg) ** 2))


def motion_washout(theta: np.ndarray, speed_m_s: float, oc: OctCfg, k0: float) -> np.ndarray:
    """Fringe washout: the surface under the beam moves axially during the exposure."""
    exposure_s = oc.exposure_duty / oc.ascan_rate_hz
    dz_um = speed_m_s * np.abs(np.tan(theta)) * exposure_s / UM
    return np.abs(np.sinc(k0 * dz_um / np.pi))


def sensitivity_rolloff(z_um: np.ndarray, z_max_um: float) -> np.ndarray:
    """Finite pixel width: sensitivity falls with depth."""
    return np.sinc(0.5 * z_um / z_max_um)


def dispersion_phase(spec: Spectrometer, k: np.ndarray, coeffs) -> np.ndarray:
    """Phase mismatch between the arms: quadratic and cubic in band coordinate."""
    u = spec.band_coordinate(k)
    return coeffs[0] * u**2 + coeffs[1] * u**3
