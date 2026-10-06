"""Load and validate configs/default.yaml, with preset and slider overrides."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_PATH = CONFIG_DIR / "default.yaml"
PRESET_DIR = CONFIG_DIR / "presets"

PRODUCT_PRESETS = ("enteric", "sustained_release", "taste_mask")
SCALE_PRESETS = ("multilab_2kg", "gpcg30_30kg", "fbc125")
FAULT_SCENARIOS = (
    "none",
    "substrate_shift",
    "nozzle_block",
    "over_wetting",
    "spray_drying",
    "maldistribution",
    "window_fouling",
)


# Sections that decide what a batch looks like and how it is analysed. Reports and
# dashboard bundles are named by a hash over these, so the names stay stable when
# sections for later modules are added.
BATCH_SECTIONS = ("seed", "pellet", "coating", "core", "batch", "wurster", "twin", "window",
                  "camera", "spec", "measurement", "gate", "fault")
ANALYSIS_SECTIONS = BATCH_SECTIONS + ("estimator", "controller", "diagnosis", "validation")


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PelletCfg(_Section):
    core_median_um: float = Field(gt=0)
    size_sigma_log: float = Field(ge=0)
    core_density_g_cm3: float = Field(gt=0)


class CoatingCfg(_Section):
    n: float = Field(gt=1)
    density_g_cm3: float = Field(gt=0)


class CoreCfg(_Section):
    n: float = Field(gt=1)


class BatchCfg(_Section):
    mass_kg: float = Field(gt=0)
    duration_h: float = Field(gt=0)
    step_s: float = Field(gt=0)
    n_pellets: int = Field(gt=0)
    target_mean_um: float = Field(gt=0)
    spray_efficiency: float = Field(gt=0, le=1)
    spray_solids_g_per_min: float | None = Field(default=None, gt=0)


class WursterCfg(_Section):
    cycle_time_s: float = Field(gt=0)
    cycle_time_rsd: float = Field(ge=0)
    deposit_nm_per_pass: float = Field(gt=0)
    cycle_size_exponent_b: float
    size_growth_exponent_k: float
    growth_ratio_target: tuple[float, float]
    fusion_rate_per_h: float = Field(ge=0)
    fines_per_min: float = Field(ge=0)
    fines_size_um: tuple[float, float]


class TwinCfg(_Section):
    rng_chunks: int = Field(gt=0)
    pass_classes: int = Field(gt=0)
    pass_tail: float = Field(gt=0, lt=1)
    stats_subsample: int = Field(gt=0)
    thickness_hist_bins: int = Field(gt=1)
    thickness_hist_max_factor: float = Field(gt=1)
    size_hist_bins: int = Field(gt=1)
    size_hist_range: tuple[float, float]


class WindowCfg(_Section):
    size_bias_m: float
    hidden_selection_gamma: float
    objects_per_min: float = Field(gt=0)


class CameraCfg(_Section):
    size_noise_um: float = Field(ge=0)


class SpecCfg(_Section):
    d10_min_um: float = Field(gt=0)


class MeasurementCfg(_Section):
    sigma_um: float = Field(ge=0)
    n_assumed: float = Field(gt=1)
    undecided_base: float = Field(ge=0, le=1)
    undecided_fouled: float = Field(ge=0, le=1)
    twin_misread_model: Literal["random", "thin", "thick"]
    twin_misread_random: tuple[float, float]
    twin_misread_thin: float = Field(gt=0)
    twin_misread_thick: float = Field(gt=0)
    error_model_path: str | None = None


class GateCfg(_Section):
    twin_recall_target: float = Field(ge=0, le=1)


class FaultCfg(_Section):
    scenario: Literal[FAULT_SCENARIOS]  # type: ignore[valid-type]
    start_h: float = Field(ge=0)
    substrate_shift_um: float
    nozzle_block_factor: float = Field(gt=0)
    over_wetting_fusion_factor: float = Field(gt=0)
    over_wetting_growth_factor: float = Field(gt=0)
    spray_drying_efficiency_factor: float = Field(gt=0)
    spray_drying_fines_factor: float = Field(gt=0)
    maldistribution_rsd: float = Field(ge=0)
    fouling_ramp_h: float = Field(gt=0)


class EstimatorCfg(_Section):
    window_min: float = Field(gt=0)
    growth_shift: bool
    n_size_bins: int = Field(gt=1)
    weight_trim_percentile: float = Field(gt=0, le=100)
    reference: Literal["oracle", "atline", "coa"]
    atline_n: int = Field(gt=0)
    min_bin_count: int = Field(gt=0)
    bootstrap_n: int = Field(gt=0)
    interval: float = Field(gt=0, lt=1)
    method: Literal["raw", "ipw", "model", "hybrid"]
    growth_window_min: float = Field(gt=0)
    deconvolve_noise: bool
    quantile_tol_um: float = Field(gt=0)


class ControllerCfg(_Section):
    p_d10_min: float = Field(gt=0, lt=1)
    cv_target: float = Field(gt=0)
    undecided_limit: float = Field(gt=0, le=1)
    min_objects: int = Field(gt=0)


class DiagnosisCfg(_Section):
    window_min: float = Field(gt=0)
    baseline_h: float = Field(gt=0)
    growth_low_factor: float = Field(gt=0, lt=1)
    agglomerate_high_pct: float = Field(gt=0)
    fines_high_factor: float = Field(gt=1)
    spread_high_factor: float = Field(gt=1)
    spread_reference_ratio: float = Field(gt=0)
    spread_from_h: float = Field(ge=0)
    undecided_high: float = Field(gt=0, le=1)
    size_shift_um: float = Field(gt=0)


class ValidationCfg(_Section):
    n_pellets: int = Field(gt=0)
    m_grid: tuple[float, ...]
    k_grid: tuple[float, ...]
    gamma_grid: tuple[float, ...]
    n_seeds: int = Field(gt=0)
    fault_seeds: int = Field(gt=0)
    eval_from_h: float = Field(ge=0)
    quick_n_seeds: int = Field(gt=0)
    quick_m_grid: tuple[float, ...]
    quick_k_grid: tuple[float, ...]


class AppCfg(_Section):
    n_pellets: int = Field(gt=0)
    n_pellets_live: int = Field(gt=0)
    frame_every: int = Field(gt=0)
    animation_pellets: int = Field(gt=0)
    animation_frames: int = Field(gt=1)
    sample_rows: int = Field(gt=0)


class DissolutionCfg(_Section):
    thickness_um: tuple[float, ...]
    t63_min: tuple[float, ...]

    @model_validator(mode="after")
    def _same_length(self) -> DissolutionCfg:
        if len(self.thickness_um) != len(self.t63_min) or len(self.thickness_um) < 2:
            raise ValueError("dissolution.thickness_um and t63_min need equal length >= 2")
        return self


class OctCfg(_Section):
    center_nm: float = Field(gt=0)
    fwhm_nm: float = Field(gt=0)
    lateral_spot_um: float = Field(gt=0)
    ascan_rate_hz: float = Field(gt=0)
    n_pixels: int = Field(gt=16)
    span_nm: float = Field(gt=0)
    exposure_duty: float = Field(gt=0, le=1)
    reference_reflectance: float = Field(gt=0, le=1)
    window_face_um: float = Field(gt=0)
    window_face_reflectance: float = Field(ge=0, le=1)
    reflector_um: float = Field(gt=0)
    reflector_reflectance: float = Field(ge=0, le=1)
    standoff_um: tuple[float, float]
    falloff_deg: float = Field(gt=0)
    field_um: float = Field(gt=0)
    max_raw_ascans: int = Field(gt=8)
    scatter_depth_um: float = Field(gt=0)
    coating_scatter_per_um2: float = Field(ge=0)
    coating_scatter_amp: float = Field(ge=0)
    coating_atten_per_mm: float = Field(ge=0)
    core_scatter_per_um2: float = Field(ge=0)
    core_scatter_amp: float = Field(ge=0)
    core_atten_per_mm: float = Field(ge=0)
    pigment_scatter_per_um2: float = Field(ge=0)
    pigment_scatter_amp: float = Field(ge=0)
    pigment_atten_per_mm: float = Field(ge=0)
    haze_thickness_um: float = Field(gt=0)
    haze_scatter_per_um2: float = Field(ge=0)
    haze_scatter_amp: float = Field(ge=0)
    fouling_transmission: float = Field(gt=0, le=1)
    dispersion_rad: tuple[float, float]
    window: Literal["hann", "none"]
    zero_pad: int = Field(ge=1)
    depth_pixels: int = Field(gt=8)
    crop_margin_px: int = Field(ge=0)
    out_ascans: int = Field(gt=8)
    average_ascans: int = Field(ge=1)
    db_range: tuple[float, float]
    valid_snr_db: float
    detect_snr_db: float
    snr_db: float


class UserCfg(_Section):
    name: str
    role: str
    pin_hash: str


class ComplianceCfg(_Section):
    pin_iterations: int = Field(gt=0)
    reason_codes: tuple[str, ...]
    meanings: tuple[str, ...]
    users: tuple[UserCfg, ...]


class SegCfg(_Section):
    widths: tuple[int, ...]
    pretrain_epochs: int = Field(gt=0)
    finetune_epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    lr: float = Field(gt=0)
    finetune_lr: float = Field(gt=0)
    encoder_lr_factor: float = Field(gt=0, le=1)
    weight_decay: float = Field(ge=0)
    dice_weight: float = Field(ge=0)
    oct5k_crop: tuple[int, int]
    oct5k_val_volumes: float = Field(gt=0, lt=1)
    oct5k_test_volumes: float = Field(gt=0, lt=1)
    synthetic_crop: tuple[int, int]
    aug_gain: tuple[float, float]
    aug_gamma: tuple[float, float]
    aug_speckle: float = Field(ge=0)
    aug_shift_px: int = Field(ge=0)
    dp_max_jump_px: int = Field(ge=0)
    dp_min_gap_px: int = Field(ge=1)
    valid_min_prob: float = Field(gt=0, lt=1)
    eval_min_snr_db: float
    separable_min_px: float = Field(gt=0)


class SolveCfg(_Section):
    max_angle_deg: float = Field(gt=0, lt=90)
    apex_angle_deg: float = Field(gt=0, lt=90)
    assumed_n: float = Field(gt=1)
    search_px: int = Field(ge=0)
    core_branch: Literal["below", "above"]
    anchor_pellets: int = Field(gt=0)
    microscopy_sigma_um: float = Field(ge=0)
    min_ascans: int = Field(gt=2)


class OctDatasetCfg(_Section):
    n_train: int = Field(gt=0)
    n_val: int = Field(gt=0)
    n_locked: int = Field(gt=0)
    thickness_um: tuple[float, float]
    n_coat: tuple[float, float]
    n_core: tuple[float, float]
    radius_um: tuple[float, float]
    snr_db: tuple[float, float]
    speed_m_s: tuple[float, float]
    fouling_beta: tuple[float, float]
    pigment_high_share: float = Field(ge=0, le=1)
    pigment_low: tuple[float, float]
    pigment_high: tuple[float, float]
    seed_offsets: dict[str, int]


class Config(_Section):
    seed: int
    pellet: PelletCfg
    coating: CoatingCfg
    core: CoreCfg
    batch: BatchCfg
    wurster: WursterCfg
    twin: TwinCfg
    window: WindowCfg
    camera: CameraCfg
    spec: SpecCfg
    measurement: MeasurementCfg
    gate: GateCfg
    fault: FaultCfg
    estimator: EstimatorCfg
    controller: ControllerCfg
    diagnosis: DiagnosisCfg
    validation: ValidationCfg
    app: AppCfg
    dissolution: DissolutionCfg
    oct: OctCfg
    compliance: ComplianceCfg
    seg: SegCfg
    solve: SolveCfg
    oct_dataset: OctDatasetCfg

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def analysis_hash(self) -> str:
        """Hash that names validation reports: the batch and everything computed from it."""
        return self.hash(include=ANALYSIS_SECTIONS)

    def with_overrides(self, overrides: dict[str, Any]) -> Config:
        """Return a validated copy with nested or dotted overrides applied.

        The dashboard's sliders use this, so they never touch module globals.
        """
        return Config.model_validate(_deep_merge(self.to_dict(), _expand_dotted(overrides)))

    def hash(self, exclude: tuple[str, ...] = (), include: tuple[str, ...] | None = None) -> str:
        """Short, stable hash of the configuration (first 12 hex chars of SHA-256).

        include limits the hash to the named sections, so adding a section for a later
        module does not rename results that never depended on it.
        """
        data = self.to_dict()
        if include is not None:
            data = {key: data[key] for key in include}
        for key in exclude:
            data.pop(key, None)
        blob = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _expand_dotted(overrides: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in overrides.items():
        if isinstance(value, dict):
            value = _expand_dotted(value)
        parts = key.split(".")
        node = out
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        if isinstance(value, dict) and isinstance(node.get(parts[-1]), dict):
            node[parts[-1]] = _deep_merge(node[parts[-1]], value)
        else:
            node[parts[-1]] = value
    return out


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def list_presets() -> list[str]:
    return sorted(p.stem for p in PRESET_DIR.glob("*.yaml"))


def load_config(
    presets: list[str] | tuple[str, ...] | str | None = None,
    overrides: dict[str, Any] | None = None,
    path: Path | str = DEFAULT_PATH,
) -> Config:
    """Load default.yaml, apply presets in order, then overrides, and validate."""
    data = _read_yaml(Path(path))
    if isinstance(presets, str):
        presets = [presets]
    for name in presets or ():
        preset_path = PRESET_DIR / f"{name}.yaml"
        if not preset_path.exists():
            raise FileNotFoundError(f"Unknown preset '{name}'. Available: {list_presets()}")
        data = _deep_merge(data, _read_yaml(preset_path))
    if overrides:
        data = _deep_merge(data, _expand_dotted(overrides))
    return Config.model_validate(data)
