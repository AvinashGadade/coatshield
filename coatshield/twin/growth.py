"""Pass-based, mass-conserving coating growth, the spray recipe and agglomeration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coatshield.config import Config
from coatshield.twin.faults import FaultState
from coatshield.twin.kernels import apply_growth, build_alias, draw_passes, mixed_poisson_pmf
from coatshield.twin.population import SINGLE, UM3_PER_CM3, Population


@dataclass(frozen=True)
class Recipe:
    """What the batch record would prescribe, calculated for nominal cores."""

    solids_g_per_min: float  # sprayed solids rate
    target_weight_gain_pct: float  # weight gain that gives batch.target_mean_um on nominal cores
    target_solids_g: float  # sprayed solids at which the gravimetric rule stops
    n_real_nominal: float  # pellets in the batch if the cores were nominal


def make_recipe(cfg: Config) -> Recipe:
    med = cfg.pellet.core_median_um
    s2 = cfg.pellet.size_sigma_log**2
    mass_g = cfg.batch.mass_kg * 1000.0
    mean_d2 = med**2 * np.exp(2.0 * s2)  # log-normal moments
    mean_d3 = med**3 * np.exp(4.5 * s2)
    n_real = mass_g / (cfg.pellet.core_density_g_cm3 * np.pi / 6.0 * mean_d3 / UM3_PER_CM3)

    rate = cfg.batch.spray_solids_g_per_min
    if rate is None:
        # One pass lays deposit_nm_per_pass on the bed's surface every cycle_time_s.
        um_per_s = cfg.wurster.deposit_nm_per_pass / 1000.0 / cfg.wurster.cycle_time_s
        volume_cm3_per_s = n_real * np.pi * mean_d2 * um_per_s / UM3_PER_CM3
        rate = volume_cm3_per_s * cfg.coating.density_g_cm3 * 60.0 / cfg.batch.spray_efficiency

    h = cfg.batch.target_mean_um
    gain = (cfg.coating.density_g_cm3 / cfg.pellet.core_density_g_cm3) * (
        ((med + 2.0 * h) ** 3 - med**3) / med**3
    )
    return Recipe(
        solids_g_per_min=float(rate),
        target_weight_gain_pct=float(100.0 * gain),
        target_solids_g=float(gain * mass_g / cfg.batch.spray_efficiency),
        n_real_nominal=float(n_real),
    )


class PassSampler:
    """Pass-rate classes and their tabulated pass-count distributions (see kernels.py)."""

    def __init__(self, pop: Population, cfg: Config, dt_s: float) -> None:
        self.dt_s = dt_s
        self.tail = cfg.twin.pass_tail
        log_rate = np.log(pop.rate_per_s[pop.rate_per_s > 0])
        lo, hi = float(log_rate.min()), float(log_rate.max())
        n = cfg.twin.pass_classes if hi > lo else 1
        self._lo, self._inv_width = lo, (n / (hi - lo) if hi > lo else 0.0)
        centres = lo + (np.arange(n) + 0.5) * ((hi - lo) / n)
        self.class_lam = np.exp(centres) * dt_s  # mean passes per step, by class
        self.cls = np.zeros(pop.n, dtype=np.int32)
        self.lam = np.zeros(pop.n)  # mean passes per step of each pellet's class
        self.share = np.zeros(pop.n)  # share_w rescaled so the expected share is exact
        self._tables: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        self.assign(pop, slice(None))

    def assign(self, pop: Population, idx) -> None:
        """(Re)classify pellets idx after their size terms change."""
        rate = pop.rate_per_s[idx]
        active = rate > 0
        safe = np.where(active, rate, 1.0)
        cls = np.clip(((np.log(safe) - self._lo) * self._inv_width).astype(np.int32),
                      0, self.class_lam.size - 1)
        lam = self.class_lam[cls]
        self.cls[idx] = cls
        self.lam[idx] = np.where(active, lam, 0.0)
        self.share[idx] = np.where(active, pop.share_w[idx] * safe * self.dt_s / lam, 0.0)

    def tables(self, cycle_rsd: float) -> tuple[np.ndarray, np.ndarray]:
        if cycle_rsd not in self._tables:
            sigma_ln = float(np.sqrt(np.log1p(cycle_rsd**2)))
            pmf = mixed_poisson_pmf(self.class_lam, sigma_ln, self.tail)
            self._tables[cycle_rsd] = build_alias(pmf)
        return self._tables[cycle_rsd]


def grow_step(
    pop: Population,
    cfg: Config,
    fault: FaultState,
    recipe: Recipe,
    sampler: PassSampler,
    step: int,
    key: int,
    bounds: np.ndarray,
    weight: np.ndarray,
) -> tuple[float, float]:
    """Advance coating by one time step.

    Every pellet makes a Poisson number of passes (rate spread log-normally from step
    to step); the coating volume sprayed in the step is shared among passes in
    proportion to share_w and added as an exact shell.
    Returns (sprayed solids in g at real scale, deposited volume in um^3 at simulated scale).
    """
    sprayed_g = recipe.solids_g_per_min * fault.spray_factor * sampler.dt_s / 60.0
    efficiency = cfg.batch.spray_efficiency * fault.efficiency_factor
    volume_um3 = sprayed_g * efficiency / cfg.coating.density_g_cm3 * UM3_PER_CM3 * pop.scale

    prob, alias = sampler.tables(fault.cycle_rsd)
    total = draw_passes(key, step, sampler.cls, prob, alias, sampler.share, pop.cum_passes,
                        weight, bounds)
    pop.exp_passes += sampler.lam
    if total <= 0:
        return sprayed_g, 0.0
    apply_growth(pop.coated3, pop.coated, weight, 6.0 / np.pi * volume_um3 / total)
    return sprayed_g, volume_um3


def fuse_step(
    pop: Population,
    cfg: Config,
    fault: FaultState,
    sampler: PassSampler,
    gen: np.random.Generator,
) -> int:
    """Merge random pairs of single pellets into twins; returns the number of new twins.

    A twin keeps the volume of both pellets (equivalent diameter 2^(1/3) D for equal
    pellets), is flagged, and leaves the spec population.
    """
    n_single = int(np.count_nonzero(pop.kind == SINGLE))
    rate = cfg.wurster.fusion_rate_per_h * fault.fusion_factor * sampler.dt_s / 3600.0
    n_pairs = int(gen.poisson(0.5 * rate * n_single))
    if n_pairs == 0:
        return 0
    cand = gen.integers(0, pop.n, size=4 * n_pairs + 16)
    cand = cand[pop.kind[cand] == SINGLE]
    _, first = np.unique(cand, return_index=True)
    cand = cand[np.sort(first)]
    n_pairs = min(n_pairs, cand.size // 2)
    a, b = cand[:n_pairs], cand[n_pairs : 2 * n_pairs]
    pop.absorb(a, b)
    sampler.assign(pop, a)
    sampler.assign(pop, b)
    return n_pairs
