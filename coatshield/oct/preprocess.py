"""Processing chain, mirroring a real spectral-domain OCT instrument.

background subtraction -> resampling from even wavelength to even wavenumber ->
dispersion compensation -> window and zero-padded FFT -> magnitude in dB above the
noise floor -> crop in depth -> lateral resampling -> averaging of neighbouring A-scans.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize
from scipy.signal import hilbert

from coatshield.config import OctCfg
from coatshield.oct.physics import Spectrometer, dispersion_phase, spectrometer
from coatshield.seeds import rng

_UPSAMPLE = 2  # spectral upsampling before the wavelength-to-wavenumber interpolation
# Share of the depth range used as noise floor: below the pellet, above the last tenth
# where the wavelength-to-wavenumber interpolation lets the noise roll off.
_NOISE_REGION = (0.64, 0.76)
_SAMPLE_GAP_UM = 30.0  # the pellet search starts this far below the reference reflector


@dataclass
class Processed:
    image: np.ndarray  # [out_ascans, depth_pixels] uint8, the stored scan
    db: np.ndarray  # [out_ascans, depth_pixels] dB above the noise floor
    x_um: np.ndarray  # lateral position of each output column
    crop_start_px: int  # first depth bin of the crop
    depth_px_um: float
    dispersion: tuple[float, float]
    noise_floor: float  # mean noise power of a single processed A-scan
    stages: dict = field(default_factory=dict)  # intermediates, when asked for


def to_wavenumber(spectra: np.ndarray, spec: Spectrometer) -> np.ndarray:
    """Resample spectra from the even-wavelength pixels to the even-wavenumber grid."""
    n = spectra.shape[-1]
    fine = np.fft.irfft(np.fft.rfft(spectra, axis=-1), n=_UPSAMPLE * n, axis=-1) * _UPSAMPLE
    lam_fine = spec.wavelength_um[0] + (spec.wavelength_um[1] - spec.wavelength_um[0]) * (
        np.arange(_UPSAMPLE * n) / _UPSAMPLE
    )
    pos = (2.0 * np.pi / spec.k_lin - lam_fine[0]) / (lam_fine[1] - lam_fine[0])
    i0 = np.clip(np.floor(pos).astype(int), 0, _UPSAMPLE * n - 2)
    frac = pos - i0
    return fine[..., i0] * (1.0 - frac) + fine[..., i0 + 1] * frac


def _window(oc: OctCfg, n: int) -> np.ndarray:
    return np.hanning(n) if oc.window == "hann" else np.ones(n)


def depth_profile(analytic: np.ndarray, spec: Spectrometer, oc: OctCfg, coeffs) -> np.ndarray:
    """Power against depth for analytic spectra on the even-wavenumber grid."""
    corrected = analytic * np.exp(-1j * dispersion_phase(spec, spec.k_lin, coeffs))
    ascan = np.fft.fft(corrected * _window(oc, spec.k_lin.size), n=spec.n_fft, axis=-1)
    half = ascan[..., : spec.n_fft // 2]
    return half.real**2 + half.imag**2


def estimate_dispersion(analytic: np.ndarray, spec: Spectrometer, oc: OctCfg,
                        z_range_um: tuple[float, float] | None = None) -> tuple[float, float]:
    """Dispersion coefficients that minimise the entropy (maximise sharpness) of the A-scan."""
    sl = slice(None)
    if z_range_um is not None:
        sl = slice(int(z_range_um[0] / spec.depth_px_um), int(z_range_um[1] / spec.depth_px_um))

    def entropy(coeffs) -> float:
        p = depth_profile(analytic, spec, oc, coeffs)[..., sl]
        p = p / p.sum(axis=-1, keepdims=True)
        return float(-(p * np.log(p + 1e-30)).sum(axis=-1).mean())

    # The entropy has side minima, so scan the quadratic term before refining both.
    coarse = np.arange(-16.0, 16.5, 1.0)
    a2 = coarse[int(np.argmin([entropy((a, 0.0)) for a in coarse]))]
    best = minimize(entropy, x0=[a2, 0.0], method="Nelder-Mead",
                    options={"xatol": 1e-3, "fatol": 1e-7, "maxiter": 400})
    return float(best.x[0]), float(best.x[1])


def noise_floor(power: np.ndarray) -> float:
    """Mean noise power from a signal-free band of depths below the pellet.

    Noise power is exponentially distributed, so mean = median / ln 2; the median
    ignores the thin trace of a pellet edge crossing the band.
    """
    lo, hi = (int(power.shape[-1] * f) for f in _NOISE_REGION)
    return float(np.median(power[..., lo:hi]) / np.log(2.0))


def find_crop_start(db: np.ndarray, spec: Spectrometer, oc: OctCfg) -> int:
    """First depth bin of the crop: a margin above the pellet apex found in the data."""
    first = int((oc.reflector_um + _SAMPLE_GAP_UM) / spec.depth_px_um)
    region = db[:, first:]
    smooth = (region[:, :-2] + region[:, 1:-1] + region[:, 2:]) / 3.0
    hit = smooth > oc.detect_snr_db
    found = hit.any(axis=1)
    if not found.any():
        return first
    apex = int(np.percentile(np.argmax(hit[found], axis=1), 5)) + first
    return max(apex - oc.crop_margin_px, first)


def _lateral_resample(power: np.ndarray, x_um: np.ndarray, oc: OctCfg):
    """Average raw A-scans into out_ascans columns of fixed pitch (set by the camera speed)."""
    pitch = oc.field_um / oc.out_ascans
    x_out = (np.arange(oc.out_ascans) - (oc.out_ascans - 1) / 2.0) * pitch
    col = np.floor((x_um + oc.field_um / 2.0) / pitch).astype(int)
    keep = (col >= 0) & (col < oc.out_ascans)
    count = np.bincount(col[keep], minlength=oc.out_ascans)
    if count.min() > 0:
        out = np.zeros((oc.out_ascans, power.shape[1]))
        np.add.at(out, col[keep], power[keep])
        return out / count[:, None], x_out
    # Fewer raw A-scans than columns: interpolate along the scan instead.
    pos = np.interp(x_out, x_um, np.arange(x_um.size))
    i0 = np.clip(np.floor(pos).astype(int), 0, x_um.size - 2)
    frac = (pos - i0)[:, None]
    return power[i0] * (1.0 - frac) + power[i0 + 1] * frac, x_out


def _neighbour_average(power: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return power
    pad = n // 2
    padded = np.pad(power, ((pad, n - 1 - pad), (0, 0)), mode="edge")
    return sum(padded[i : i + power.shape[0]] for i in range(n)) / n


def to_uint8(db: np.ndarray, oc: OctCfg) -> np.ndarray:
    lo, hi = oc.db_range
    return np.clip(np.round((db - lo) / (hi - lo) * 255.0), 0, 255).astype(np.uint8)


def process(
    spectra: np.ndarray,
    background: np.ndarray,
    x_um: np.ndarray,
    oc: OctCfg,
    dispersion: tuple[float, float] | None = None,
    crop_start_px: int | None = None,
    stages: bool = False,
) -> Processed:
    """Raw spectra [A-scan, pixel] -> the stored scan [out_ascans, depth_pixels].

    dispersion: coefficients from calibration; estimated from this scan when None.
    crop_start_px: first depth bin kept; found from the pellet apex when None.
    """
    spec = spectrometer(oc)
    fringes = to_wavenumber(spectra - background, spec)
    analytic = hilbert(fringes, axis=-1)
    if dispersion is None:
        step = max(1, analytic.shape[0] // 16)
        dispersion = estimate_dispersion(analytic[::step], spec, oc)
    power = depth_profile(analytic, spec, oc, dispersion)
    floor = noise_floor(power)

    resampled, x_out = _lateral_resample(power, x_um, oc)
    averaged = _neighbour_average(resampled, oc.average_ascans)
    db_full = 10.0 * np.log10(np.maximum(averaged, 1e-30) / floor)
    if crop_start_px is None:
        crop_start_px = find_crop_start(db_full, spec, oc)
    crop_start_px = int(np.clip(crop_start_px, 0, db_full.shape[1] - oc.depth_pixels))
    db = db_full[:, crop_start_px : crop_start_px + oc.depth_pixels]

    out = Processed(image=to_uint8(db, oc), db=db.astype(np.float32), x_um=x_out,
                    crop_start_px=crop_start_px, depth_px_um=spec.depth_px_um,
                    dispersion=tuple(dispersion), noise_floor=floor)
    if stages:
        out.stages = {"fringes_k": fringes, "power": power, "db_full": db_full}
    return out


@lru_cache(maxsize=8)
def _chain_gains(oc: OctCfg) -> tuple[float, float]:
    spec = spectrometer(oc)
    z = 0.5 * sum(oc.standoff_um)
    fringe = spec.source * np.cos(2.0 * spec.k * z)
    analytic = hilbert(to_wavenumber(fringe[None, :], spec), axis=-1)
    peak = float(np.sqrt(depth_profile(analytic, spec, oc, (0.0, 0.0)).max()))
    noise = rng("oct.chain_gains").standard_normal((64, oc.n_pixels))
    noise_k = hilbert(to_wavenumber(noise, spec), axis=-1)
    lo, hi = (int(v / spec.depth_px_um) for v in oc.standoff_um)
    return peak, float(np.sqrt(depth_profile(noise_k, spec, oc, (0.0, 0.0))[:, lo:hi].mean()))


def chain_gains(oc: OctCfg) -> tuple[float, float]:
    """(peak magnitude of a unit fringe, rms magnitude of unit white noise) after the chain.

    Both are taken at the depths where pellets sit (the standoff range).
    """
    return _chain_gains(oc)


@lru_cache(maxsize=8)
def _calibrated_dispersion(config_json: str) -> tuple[float, float]:
    from coatshield.config import Config
    from coatshield.oct.generator import default_params, simulate

    cfg = Config.model_validate_json(config_json)
    oc = cfg.oct
    raw = simulate(default_params(cfg, snr_db=max(oc.snr_db, 40.0)), cfg, stream="oct.calibration")
    spec = spectrometer(oc)
    analytic = hilbert(to_wavenumber(raw.spectra - raw.background, spec), axis=-1)
    step = max(1, analytic.shape[0] // 16)
    around_reflector = (oc.reflector_um - 20.0, oc.reflector_um + 20.0)
    return estimate_dispersion(analytic[::step], spec, oc, around_reflector)


def calibrated_dispersion(cfg) -> tuple[float, float]:
    """Dispersion coefficients from a clean-window calibration scan of the reference reflector.

    Found once per configuration by minimising the entropy of the reflector's A-scan,
    then reused for every scan, as an instrument would after its calibration.
    """
    return _calibrated_dispersion(cfg.model_dump_json())
