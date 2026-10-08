"""The whole measurement chain for one sampled object, every intermediate kept.

camera image -> gate -> OCT scan -> processing -> boundary finder -> graph search ->
solver with the batch's pooled index -> confidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from coatshield.compliance import confidence as conf
from coatshield.compliance import drift
from coatshield.config import Config
from coatshield.gate import silhouettes
from coatshield.gate.classical import classify
from coatshield.oct.generator import ScanParams, labels, simulate
from coatshield.oct.preprocess import calibrated_dispersion, process
from coatshield.seeds import rng
from coatshield.solve.thickness import measure_surfaces, pellet_thickness
from coatshield.twin.population import FINES, SINGLE, TWIN

_CORE_ROWS = 80  # depth rows below the inner surface used for the core speckle feature
_GATE_CLASS = {SINGLE: silhouettes.SINGLE, TWIN: silhouettes.TWIN, FINES: silhouettes.FINES}


@dataclass(frozen=True)
class SampledObject:
    """One object as the twin's window sampled it (truth, hidden from the chain's outputs)."""

    true_class: int  # SINGLE, TWIN or FINES (twin.population)
    diameter_um: float  # coated diameter
    thickness_um: float
    n_coat: float
    n_core: float
    speed_m_s: float
    snr_db: float
    fouling: float = 0.0
    pigment: float = 0.0
    standoff_um: float = 250.0
    seed: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ChainRecord:
    """Result for one object. status: gated | no_signal | undecided | measured."""

    obj: SampledObject
    status: str
    gate_class: str
    gate_confidence: float
    thickness_um: float = float("nan")
    uncertainty_um: float = float("nan")
    optical_um: float = float("nan")
    radius_um: float = float("nan")
    n_reflectance: float = float("nan")
    n_ratio: float = float("nan")
    confidence: float = 0.0
    seg_confidence: float = 0.0
    fit_confidence: float = 0.0
    fit_residual_um: float = float("nan")
    spread_um: float = float("nan")
    drift_features: tuple = ()  # compliance.drift.FEATURES of the scan (empty when gated)
    intermediates: dict = field(default_factory=dict)

    def summary(self) -> dict:
        """Flat row for tables (no arrays)."""
        skip = ("obj", "intermediates", "drift_features")
        row = {k: v for k, v in asdict(self).items() if k not in skip}
        row.update({f"f_{name}": value
                    for name, value in zip(drift.FEATURES, self.drift_features, strict=False)})
        row.update({f"true_{k}": v for k, v in self.obj.as_dict().items()})
        row["error_um"] = self.thickness_um - self.obj.thickness_um
        return row


class LabelSegmenter:
    """Stand-in boundary finder for tests and dry runs: the simulation's own surfaces plus
    localisation noise. It needs the truth, so it can never be used on real scans."""

    def __init__(self, cfg: Config, noise_px: float = 1.0) -> None:
        self.cfg, self.noise_px = cfg, noise_px
        self.truth: tuple[ScanParams, np.ndarray, int] | None = None

    def set_truth(self, params: ScanParams, x_um: np.ndarray, crop_start_px: int) -> None:
        self.truth = (params, x_um, crop_start_px)

    def surfaces(self, image: np.ndarray) -> dict[str, np.ndarray]:
        params, x_um, crop = self.truth
        lab = labels(params, self.cfg, x_um, crop)
        gen = rng(f"chain.label_segmenter.{params.seed}", self.cfg.seed)
        n = x_um.size
        outer = np.nan_to_num(lab["outer_px"]) + self.noise_px * gen.standard_normal(n)
        inner = np.nan_to_num(lab["inner_px"]) + self.noise_px * gen.standard_normal(n)
        margin = np.where(lab["valid"], 0.9, 0.1)
        return {"outer": outer, "inner": np.maximum(inner, outer + 1), "valid": lab["valid"],
                "margin_outer": margin, "margin_inner": margin, "prob": None}


def process_object(obj: SampledObject, cfg: Config, segmenter, pooled_n: float,
                   pooled_n_se: float = 0.0, keep: bool = True) -> ChainRecord:
    """Run one object through the chain. segmenter: seg.infer.Segmenter (or a stand-in with
    the same surfaces() method). keep=False drops the image-sized intermediates."""
    gv = cfg.gate_vision
    gen = rng(f"chain.object.{obj.seed}", cfg.seed)
    camera, _ = silhouettes.render(_GATE_CLASS[obj.true_class], cfg, gen, obj.diameter_um)
    verdict = classify(camera, cfg)
    gate_class = silhouettes.CLASSES[verdict.label]
    inter: dict = {"camera": camera, "gate_features": verdict.features.as_dict()
                   if verdict.features else None} if keep else {}
    passes = verdict.passes and verdict.confidence >= gv.single_confidence_min
    if not passes:
        return ChainRecord(obj, "gated", gate_class, verdict.confidence, intermediates=inter)

    # A twin that slips through is scanned as one pellet whose coat misreads (config model).
    seen = obj.thickness_um
    if obj.true_class == TWIN:
        mc = cfg.measurement
        seen *= {"thin": mc.twin_misread_thin, "thick": mc.twin_misread_thick,
                 "random": gen.uniform(*mc.twin_misread_random)}[mc.twin_misread_model]
    params = ScanParams(thickness_um=seen, n_coat=obj.n_coat, n_core=obj.n_core,
                        radius_um=obj.diameter_um / 2.0, speed_m_s=obj.speed_m_s,
                        snr_db=obj.snr_db, fouling=obj.fouling, pigment=obj.pigment,
                        standoff_um=obj.standoff_um, seed=obj.seed)
    raw = simulate(params, cfg, stream="chain.scan")
    scan = process(raw.spectra, raw.background, raw.x_um, cfg.oct,
                   dispersion=calibrated_dispersion(cfg))
    if isinstance(segmenter, LabelSegmenter):
        segmenter.set_truth(params, scan.x_um, scan.crop_start_px)
    found = segmenter.surfaces(scan.image)
    if keep:
        inter.update(spectrum=raw.spectra[raw.spectra.shape[0] // 2], scan=scan.image,
                     x_um=scan.x_um, depth_px_um=scan.depth_px_um, prob=found.get("prob"),
                     outer_px=found["outer"], inner_px=found["inner"], valid=found["valid"],
                     reflector_db=scan.reflector_db)
    # Features for the drift monitor: the core region is taken just below the inner surface.
    rows = np.arange(scan.db.shape[1])[None, :]
    inner_row = found["inner"][:, None]
    core = found["valid"][:, None] & (rows > inner_row) & (rows <= inner_row + _CORE_ROWS)
    features = tuple(float(v) for v in drift.scan_features(
        scan.db, scan.noise_floor, scan.reflector_db, found.get("prob"), core,
        verdict.confidence))
    surfaces = measure_surfaces(scan.db, found["outer"].astype(float),
                                found["inner"].astype(float), found["valid"], scan.x_um,
                                scan.depth_px_um, scan.crop_start_px, scan.reflector_db, cfg)
    if surfaces is None:
        return ChainRecord(obj, "no_signal", gate_class, verdict.confidence,
                           drift_features=features, intermediates=inter)

    result = pellet_thickness(surfaces, pooled_n, pooled_n_se)
    seg_c = conf.segmentation_confidence(found["margin_outer"], found["margin_inner"],
                                         surfaces.used)
    spread_um = result.intra_cv * result.thickness_um
    fit_c = conf.fit_confidence(result.fit_residual_um, spread_um, cfg)
    score = conf.fuse(seg_c, fit_c)
    if keep:
        inter.update(circle=(surfaces.circle.xc, surfaces.circle.zc, surfaces.circle.radius),
                     used=surfaces.used, intra_cv=result.intra_cv,
                     fit_residual_um=result.fit_residual_um)
    return ChainRecord(
        obj, "measured" if conf.is_decided(score, cfg) else "undecided", gate_class,
        verdict.confidence, thickness_um=result.thickness_um,
        uncertainty_um=result.uncertainty_um, optical_um=result.optical_um,
        radius_um=result.radius_um, n_reflectance=surfaces.n_reflectance,
        n_ratio=surfaces.n_ratio, confidence=score, seg_confidence=seg_c, fit_confidence=fit_c,
        fit_residual_um=result.fit_residual_um, spread_um=spread_um, drift_features=features,
        intermediates=inter)
