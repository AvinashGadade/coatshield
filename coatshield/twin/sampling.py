"""What the measurement window sees: size-biased sampling, camera reading, measurement."""

from __future__ import annotations

import numpy as np

from coatshield.config import Config
from coatshield.twin.faults import FaultState
from coatshield.twin.population import ABSORBED, FINES, SINGLE, TWIN, Population


def pass_rate_z(pop: Population) -> np.ndarray:
    """Standardised pass rate: passes made relative to those expected for the pellet's size."""
    active = pop.kind != ABSORBED
    ratio = pop.cum_passes / np.maximum(pop.exp_passes, np.finfo(float).tiny)
    mu, sd = ratio[active].mean(), ratio[active].std()
    return (ratio - mu) / sd if sd > 0 else np.zeros(pop.n)


def draw_window(
    pop: Population, coated_um: np.ndarray, cfg: Config, n: int, gen: np.random.Generator
) -> np.ndarray:
    """Indices of n pellets drawn with probability ~ (D / D_median)^m * exp(gamma * z)."""
    m = cfg.window.size_bias_m
    gamma = cfg.window.hidden_selection_gamma
    if gamma != 0.0:
        # Hidden selection needs every pellet's z, so weight the whole bed directly.
        active = pop.kind != ABSORBED
        logw = np.full(pop.n, -np.inf)
        logw[active] = (m * np.log(coated_um[active] / pop.median_um)
                        + gamma * pass_rate_z(pop)[active])
        w = np.exp(logw - logw.max())
        cdf = np.cumsum(w)
        return np.searchsorted(cdf, gen.random(n) * cdf[-1], side="right")

    # Size-only selection: rejection sampling touches a few thousand pellets, not the bed.
    d_bound = coated_um.max() if m >= 0 else coated_um[pop.kind != ABSORBED].min()
    picked: list[np.ndarray] = []
    have, batch = 0, 8 * n
    while have < n:
        idx = gen.integers(0, pop.n, size=batch)
        idx = idx[pop.kind[idx] != ABSORBED]
        accept = gen.random(idx.size) < (coated_um[idx] / d_bound) ** m
        idx = idx[accept]
        picked.append(idx)
        have += idx.size
        batch = min(2 * batch, 1 << 20)
    return np.concatenate(picked)[:n]


def measure(
    cfg: Config,
    fault: FaultState,
    true_class: np.ndarray,
    true_thickness_um: np.ndarray,
    gen: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Placeholder gate and thickness measurement (replaced by the chain error model in Phase 8).

    Measured thickness = true thickness x n_true / n_assumed + Gaussian noise; the
    undecided probability rises with window fouling; twins that pass the gate misread.
    """
    mc = cfg.measurement
    n = true_class.size
    is_twin = true_class == TWIN

    gate_class = true_class.copy()
    missed = is_twin & (gen.random(n) >= cfg.gate.twin_recall_target)
    gate_class[missed] = SINGLE

    if mc.twin_misread_model == "random":
        factor = gen.uniform(*mc.twin_misread_random, size=n)
    else:
        factor = np.full(n, mc.twin_misread_thin if mc.twin_misread_model == "thin"
                         else mc.twin_misread_thick)
    seen = np.where(is_twin, true_thickness_um * factor, true_thickness_um)

    thickness = seen * (cfg.coating.n / mc.n_assumed) + mc.sigma_um * gen.standard_normal(n)
    np.maximum(thickness, 0.0, out=thickness)

    p_undecided = mc.undecided_base + (mc.undecided_fouled - mc.undecided_base) * fault.fouling
    decided = gen.random(n) >= p_undecided
    to_oct = gate_class == SINGLE
    accepted = to_oct & decided

    thickness[~accepted] = np.nan
    confidence = np.where(accepted, 1.0 - p_undecided, 0.0)
    confidence[~to_oct] = np.nan
    return {
        "thickness_um": thickness,
        "gate_class": gate_class,
        "accepted": accepted,
        "confidence": confidence,
    }


def sample_step(
    pop: Population,
    coated_um: np.ndarray,
    cfg: Config,
    fault: FaultState,
    t_s: float,
    dt_s: float,
    gen_window: np.random.Generator,
    gen_measure: np.random.Generator,
) -> dict[str, np.ndarray]:
    """One time step of window records: pellets (single or twin) plus fines."""
    n_obj = int(round(cfg.window.objects_per_min * dt_s / 60.0))
    idx = draw_window(pop, coated_um, cfg, n_obj, gen_window)

    true_size = coated_um[idx]
    true_class = pop.kind[idx].astype(np.int8)
    true_thickness = 0.5 * (true_size - pop.core_um[idx])
    size = true_size + cfg.camera.size_noise_um * gen_window.standard_normal(n_obj)
    out = measure(cfg, fault, true_class, true_thickness, gen_measure)

    n_fines = int(gen_window.poisson(cfg.wurster.fines_per_min * fault.fines_factor * dt_s / 60.0))
    fines_size = gen_window.uniform(*cfg.wurster.fines_size_um, size=n_fines)
    nan = np.full(n_fines, np.nan)

    return {
        "t_s": np.full(n_obj + n_fines, t_s),
        "size_um": np.concatenate([size, fines_size]),
        "thickness_um": np.concatenate([out["thickness_um"], nan]),
        "gate_class": np.concatenate([out["gate_class"], np.full(n_fines, FINES, np.int8)]),
        "confidence": np.concatenate([out["confidence"], nan]),
        "accepted": np.concatenate([out["accepted"], np.zeros(n_fines, bool)]),
        # Hidden truth, for validation only: an estimator must never read these.
        "true_class": np.concatenate([true_class, np.full(n_fines, FINES, np.int8)]),
        "true_thickness_um": np.concatenate([true_thickness, nan]),
        "true_size_um": np.concatenate([true_size, fines_size]),
        "pellet_idx": np.concatenate([idx, np.full(n_fines, -1)]),
    }
