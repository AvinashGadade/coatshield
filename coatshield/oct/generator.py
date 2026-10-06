"""Synthetic spectral-domain OCT scans of coated pellets, with labels from the parameters.

Each raw A-scan is a spectral interferogram on the spectrometer's wavelength pixels:
window faces, an on-window reference reflector, the air-coating and coating-core
interfaces (Fresnel amplitudes, Snell refraction, angular fall-off) and coherent
sub-resolution scatterers (Zaitsev et al., arXiv 1406.3448), with pigment, motion
washout, beam-spot blur, window fouling, dispersion mismatch and detector noise.
Labels are computed from the simulation parameters, never from the image.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from coatshield.config import Config, OctCfg
from coatshield.oct import physics
from coatshield.oct.preprocess import chain_gains
from coatshield.seeds import rng

ABOVE, COATING, CORE = 0, 1, 2  # classes of the per-pixel mask
_SYNTH_RANGE = 2  # scatterers are synthesised on a wavenumber grid this many bands wide


@dataclass(frozen=True)
class ScanParams:
    thickness_um: float
    n_coat: float
    n_core: float
    radius_um: float  # outer radius of the coated pellet
    speed_m_s: float
    snr_db: float  # surface SNR at normal incidence through a clean window
    fouling: float = 0.0  # 0 (clean window) to 1
    pigment: float = 0.0  # 0 (clear film) to 1 (opaque)
    standoff_um: float = 250.0  # optical depth of the pellet apex
    seed: int = 0  # index of this scan's random stream

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class RawScan:
    spectra: np.ndarray  # [A-scan, pixel] raw detector counts (arbitrary units)
    background: np.ndarray  # [pixel] reference-arm spectrum recorded with the sample blocked
    x_um: np.ndarray  # lateral position of each A-scan relative to the pellet apex
    params: ScanParams
    noise_sigma: float


def default_params(cfg: Config, **overrides) -> ScanParams:
    """A scan of the configured product at the default SNR, mid standoff."""
    base = dict(
        thickness_um=cfg.batch.target_mean_um,
        n_coat=cfg.coating.n,
        n_core=cfg.core.n,
        radius_um=cfg.pellet.core_median_um / 2.0 + cfg.batch.target_mean_um,
        speed_m_s=float(np.mean(cfg.oct_dataset.speed_m_s)),
        snr_db=cfg.oct.snr_db,
        standoff_um=0.5 * sum(cfg.oct.standoff_um),
    )
    base.update(overrides)
    return ScanParams(**base)


def scan_axis(p: ScanParams, oc: OctCfg) -> tuple[np.ndarray, float]:
    """Lateral positions of the simulated A-scans and the noise scale that goes with them.

    A-scans are v / rate apart. A slow pellet gives more A-scans than max_raw_ascans;
    it is then simulated on a coarser grid with the noise reduced as averaging would.
    """
    native_um = p.speed_m_s / oc.ascan_rate_hz * 1e6
    n_native = oc.field_um / native_um
    n = int(min(np.ceil(n_native), oc.max_raw_ascans))
    x = (np.arange(n) - (n - 1) / 2.0) * (oc.field_um / n)
    return x, float(np.sqrt(min(1.0, n / n_native)))


def interface_amplitudes(p: ScanParams, x_um: np.ndarray, oc: OctCfg):
    """Depths (optical path) and amplitudes of the two pellet interfaces at each position."""
    geo = physics.sphere_geometry(x_um, p.radius_um, p.thickness_um, p.n_coat)
    spec = physics.spectrometer(oc)
    through_window = 1.0 - (1.0 - oc.fouling_transmission) * p.fouling  # one-way amplitude
    common = (through_window**2 * physics.motion_washout(geo.theta, p.speed_m_s, oc, spec.k0))
    coat_atten = (oc.coating_atten_per_mm + p.pigment * oc.pigment_atten_per_mm) / 1000.0
    r_surface = physics.fresnel_amplitude(1.0, p.n_coat)
    r_inner = physics.fresnel_amplitude(p.n_coat, p.n_core)
    path = np.nan_to_num(geo.path_um)
    z_surface = p.standoff_um + np.nan_to_num(geo.sag_um)
    z_inner = z_surface + p.n_coat * path
    a_surface = r_surface * physics.angular_falloff(geo.theta, oc.falloff_deg) * common
    a_inner = (r_inner * (1.0 - r_surface**2) * np.exp(-2.0 * coat_atten * path)
               * physics.angular_falloff(geo.theta_core, oc.falloff_deg) * common)
    inside = geo.inside
    return {
        "geo": geo,
        "inside": inside,
        "z_surface": z_surface,
        "z_inner": z_inner,
        "a_surface": np.where(inside, a_surface, 0.0),
        "a_inner": np.where(inside, a_inner, 0.0),
        "common": np.where(inside, common, 0.0),
        "through_window": through_window,
        "coat_atten": coat_atten,
    }


def _scatterers(p: ScanParams, x_um: np.ndarray, faces: dict, oc: OctCfg,
                gen: np.random.Generator):
    """Random sub-resolution scatterers: (column, optical depth, complex amplitude)."""
    n_col = x_um.size
    dx = oc.field_um / n_col
    geo = faces["geo"]
    path = np.nan_to_num(geo.path_um)

    def draw(density: float, depth_um: float):
        n = gen.poisson(density * oc.field_um * depth_um)
        col = gen.integers(0, n_col, size=n)
        depth = gen.random(n) * depth_um
        phasor = (gen.standard_normal(n) + 1j * gen.standard_normal(n)) / np.sqrt(2.0)
        return col, depth, phasor

    cols, zs, amps = [], [], []
    core_atten = oc.core_atten_per_mm / 1000.0
    # Core: dense scatterers below the coating, attenuated with depth.
    col, depth, phasor = draw(oc.core_scatter_per_um2, oc.scatter_depth_um)
    keep = faces["inside"][col] & (depth >= path[col])
    col, depth, phasor = col[keep], depth[keep], phasor[keep]
    in_core = depth - path[col]
    cols.append(col)
    zs.append(faces["z_surface"][col] + p.n_coat * path[col] + p.n_core * in_core)
    amps.append(phasor * oc.core_scatter_amp * faces["common"][col]
                * np.exp(-2.0 * (faces["coat_atten"] * path[col] + core_atten * in_core)))
    # Coating: sparse scatterers, plus pigment.
    for density, amp in ((oc.coating_scatter_per_um2, oc.coating_scatter_amp),
                         (p.pigment * oc.pigment_scatter_per_um2, oc.pigment_scatter_amp)):
        if density <= 0 or not faces["inside"].any():
            continue
        col, depth, phasor = draw(density, float(path.max()))
        keep = faces["inside"][col] & (depth < path[col])
        col, depth, phasor = col[keep], depth[keep], phasor[keep]
        cols.append(col)
        zs.append(faces["z_surface"][col] + p.n_coat * depth)
        amps.append(phasor * amp * faces["common"][col]
                    * np.exp(-2.0 * faces["coat_atten"] * depth))
    # Fouling: a haze layer on the window.
    if p.fouling > 0:
        col, depth, phasor = draw(p.fouling * oc.haze_scatter_per_um2, oc.haze_thickness_um)
        cols.append(col)
        zs.append(oc.window_face_um + depth)
        amps.append(phasor * oc.haze_scatter_amp * p.fouling)
    return np.concatenate(cols), np.concatenate(zs), np.concatenate(amps), dx


def _scatter_field(col, z_um, amp, n_col: int, dx_um: float, z_ref_um: float,
                   spec: physics.Spectrometer, oc: OctCfg) -> np.ndarray:
    """Sum of amp * exp(2ikz) over scatterers, per A-scan, at the spectrometer pixels.

    Scatterers are binned in depth (carrying their sub-bin phase), blurred across
    A-scans by the beam spot, and transformed to wavenumber with one FFT per A-scan.
    """
    n_syn = _SYNTH_RANGE * spec.n_fft // oc.zero_pad * 2
    dk = _SYNTH_RANGE * 2.0 * spec.half_range / n_syn
    k_start = spec.k_lin[0] - 0.5 * (_SYNTH_RANGE - 1) * 2.0 * spec.half_range
    dz = np.pi / (n_syn * dk)
    rel = z_um - z_ref_um
    m = np.round(rel / dz).astype(np.int64)
    m_lo, m_hi = int(m.min()), int(m.max())
    value = amp * np.exp(2j * spec.k0 * (rel - m * dz))
    flat = col * (m_hi - m_lo + 1) + (m - m_lo)
    size = n_col * (m_hi - m_lo + 1)
    grid = (np.bincount(flat, weights=value.real, minlength=size)
            + 1j * np.bincount(flat, weights=value.imag, minlength=size)
            ).reshape(n_col, m_hi - m_lo + 1)
    # Double-pass Gaussian beam: amplitude response exp(-2 x^2 / w^2), w = spot radius.
    # The kernel is scaled to keep the rms of uncorrelated scatterers, so the configured
    # amplitudes do not depend on the A-scan spacing.
    sigma_cols = 0.25 * oc.lateral_spot_um / dx_um
    keep_rms = np.sqrt(2.0 * np.sqrt(np.pi) * sigma_cols) if sigma_cols > 0.5 else 1.0
    grid = keep_rms * (gaussian_filter1d(grid.real, sigma_cols, axis=0, mode="constant")
                       + 1j * gaussian_filter1d(grid.imag, sigma_cols, axis=0, mode="constant"))
    bins = np.arange(m_lo, m_hi + 1)
    full = np.zeros((n_col, n_syn), dtype=complex)
    full[:, bins % n_syn] = grid * np.exp(2j * k_start * bins * dz)
    field = np.fft.ifft(full, axis=1) * n_syn  # at k_start + n * dk
    pos = (spec.k - k_start) / dk
    i0 = np.floor(pos).astype(int)
    frac = pos - i0
    at_pixels = field[:, i0] * (1.0 - frac) + field[:, i0 + 1] * frac
    return at_pixels * np.exp(2j * spec.k * z_ref_um)


def simulate(p: ScanParams, cfg: Config, stream: str = "oct.scan") -> RawScan:
    """Raw spectra of one pellet crossing the beam."""
    oc = cfg.oct
    spec = physics.spectrometer(oc)
    gen = rng(f"{stream}.{p.seed}", cfg.seed)
    x_um, noise_scale = scan_axis(p, oc)
    faces = interface_amplitudes(p, x_um, oc)
    k = spec.k[None, :]

    def reflector(z_um, amplitude):
        z = np.atleast_1d(np.asarray(z_um, dtype=float))[:, None]
        a = np.atleast_1d(np.asarray(amplitude, dtype=float))[:, None]
        return a * physics.sensitivity_rolloff(z, spec.z_max_um) * np.exp(2j * k * z)

    sample = (reflector(faces["z_surface"], faces["a_surface"])
              + reflector(faces["z_inner"], faces["a_inner"])
              + reflector(oc.window_face_um, np.sqrt(oc.window_face_reflectance))
              + reflector(oc.reflector_um,
                          np.sqrt(oc.reflector_reflectance) * faces["through_window"] ** 2))
    col, z, amp, dx = _scatterers(p, x_um, faces, oc, gen)
    if col.size:
        amp = amp * physics.sensitivity_rolloff(z, spec.z_max_um)
        sample = sample + _scatter_field(col, z, amp, x_um.size, dx, p.standoff_um, spec, oc)

    reference = np.sqrt(oc.reference_reflectance)
    mismatch = np.exp(1j * physics.dispersion_phase(spec, spec.k, oc.dispersion_rad))
    fringes = 2.0 * reference * spec.source * np.real(mismatch[None, :] * sample)
    background = oc.reference_reflectance * spec.source

    # Detector noise for the requested SNR of the surface at normal incidence, clean window.
    peak_gain, noise_gain = chain_gains(oc)
    surface = abs(physics.fresnel_amplitude(1.0, p.n_coat)) * 2.0 * reference
    sigma = surface * peak_gain / (noise_gain * 10.0 ** (p.snr_db / 20.0)) * noise_scale
    spectra = background[None, :] + fringes + sigma * gen.standard_normal(fringes.shape)
    return RawScan(spectra=spectra.astype(np.float32), background=background, x_um=x_um,
                   params=p, noise_sigma=float(sigma))


def labels(p: ScanParams, cfg: Config, x_um: np.ndarray, crop_start_px: int) -> dict:
    """Ground truth in the coordinates of the stored scan, from the parameters alone.

    outer_px / inner_px: depth row of each surface per A-scan (NaN outside the pellet);
    mask: per-pixel class (above surface, coating, core); valid: columns where the
    surface return is expected to clear valid_snr_db.
    """
    oc = cfg.oct
    spec = physics.spectrometer(oc)
    faces = interface_amplitudes(p, x_um, oc)
    inside = faces["inside"]
    outer = np.where(inside, faces["z_surface"] / spec.depth_px_um - crop_start_px, np.nan)
    inner = np.where(inside, faces["z_inner"] / spec.depth_px_um - crop_start_px, np.nan)

    rows = np.arange(oc.depth_pixels)[None, :]
    mask = np.full((x_um.size, oc.depth_pixels), ABOVE, dtype=np.uint8)
    with np.errstate(invalid="ignore"):
        mask[inside[:, None] & (rows >= outer[:, None])] = COATING
        mask[inside[:, None] & (rows >= inner[:, None])] = CORE

    surface_peak = abs(physics.fresnel_amplitude(1.0, p.n_coat))
    with np.errstate(divide="ignore"):
        expected_snr = p.snr_db + 20.0 * np.log10(np.abs(faces["a_surface"]) / surface_peak)
    in_crop = (outer >= 0) & (inner < oc.depth_pixels)
    return {
        "outer_px": outer.astype(np.float32),
        "inner_px": inner.astype(np.float32),
        "mask": mask,
        "valid": inside & in_crop & (expected_snr >= oc.valid_snr_db),
        "depth_px_um": spec.depth_px_um,
    }
