"""Synthetic pellet OCT dataset: parameter sampling and one-scan generation."""

from __future__ import annotations

import numpy as np

from coatshield.config import Config
from coatshield.oct import physics
from coatshield.oct.generator import ScanParams, labels, simulate
from coatshield.oct.preprocess import calibrated_dispersion, process
from coatshield.seeds import rng

SPLITS = ("train", "val", "locked")


def sample_params(cfg: Config, split: str, index: int) -> ScanParams:
    """Parameters of scan `index` of a split. Splits use seed ranges that never overlap."""
    dc = cfg.oct_dataset
    seed = dc.seed_offsets[split] + index
    gen = rng(f"oct.params.{seed}", cfg.seed)
    lo, hi = dc.thickness_um
    high_pigment = gen.random() < dc.pigment_high_share
    return ScanParams(
        thickness_um=float(np.exp(gen.uniform(np.log(lo), np.log(hi)))),  # log-uniform
        n_coat=float(gen.uniform(*dc.n_coat)),
        n_core=float(gen.uniform(*dc.n_core)),
        radius_um=float(gen.uniform(*dc.radius_um)),
        speed_m_s=float(gen.uniform(*dc.speed_m_s)),
        snr_db=float(gen.uniform(*dc.snr_db)),
        fouling=float(gen.beta(*dc.fouling_beta)),
        pigment=float(gen.uniform(*(dc.pigment_high if high_pigment else dc.pigment_low))),
        standoff_um=float(gen.uniform(*cfg.oct.standoff_um)),
        seed=seed,
    )


def make_scan(cfg: Config, split: str, index: int) -> dict:
    """One stored scan with its labels.

    The depth crop starts a jittered margin above the apex depth known from the
    parameters, so no label depends on anything measured in the image.
    """
    oc = cfg.oct
    p = sample_params(cfg, split, index)
    raw = simulate(p, cfg)
    spec = physics.spectrometer(oc)
    jitter = rng(f"oct.crop.{p.seed}", cfg.seed).integers(-oc.crop_margin_px // 2,
                                                          oc.crop_margin_px // 2 + 1)
    crop = int(p.standoff_um / spec.depth_px_um) - oc.crop_margin_px + int(jitter)
    out = process(raw.spectra, raw.background, raw.x_um, oc,
                  dispersion=calibrated_dispersion(cfg), crop_start_px=crop)
    lab = labels(p, cfg, out.x_um, out.crop_start_px)
    return {
        "image": out.image,
        "outer_px": lab["outer_px"],
        "inner_px": lab["inner_px"],
        "valid": lab["valid"],
        "params": {**p.as_dict(), "crop_start_px": out.crop_start_px,
                   "depth_px_um": out.depth_px_um, "reflector_db": out.reflector_db},
    }


def masks_from_rows(outer_px: np.ndarray, inner_px: np.ndarray, depth_pixels: int) -> np.ndarray:
    """Per-pixel class mask [..., A-scan, depth] from the stored surface rows."""
    rows = np.arange(depth_pixels)
    with np.errstate(invalid="ignore"):
        coating = rows >= outer_px[..., None]
        core = rows >= inner_px[..., None]
    return coating.astype(np.uint8) + core.astype(np.uint8)
