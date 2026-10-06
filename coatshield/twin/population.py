"""Pellet population: core sizes, per-pellet state and the real batch scale."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coatshield.config import Config
from coatshield.seeds import rng

SINGLE, TWIN, FINES, ABSORBED = 0, 1, 2, 3  # object classes; ABSORBED = merged into a twin
CLASS_NAMES = {SINGLE: "single", TWIN: "twin", FINES: "fines"}

UM3_PER_CM3 = 1e12


def sphere_volume_um3(d_um):
    return np.pi / 6.0 * d_um**3


def core_median_um(cfg: Config) -> float:
    """Actual core median: the substrate-shift scenario delivers larger cores than nominal."""
    if cfg.fault.scenario == "substrate_shift":
        return cfg.pellet.core_median_um + cfg.fault.substrate_shift_um
    return cfg.pellet.core_median_um


@dataclass
class Population:
    """Current state of the N simulated pellets (the only per-pellet memory kept)."""

    core_um: np.ndarray  # core diameter (equivalent-sphere for twins; 0 once absorbed)
    coated3: np.ndarray  # coated diameter cubed, so shell volume adds exactly
    coated: np.ndarray  # coated diameter, kept in step with coated3
    kind: np.ndarray  # SINGLE, TWIN or ABSORBED
    rate_per_s: np.ndarray  # mean spray-zone passes per second (size-scaled)
    share_w: np.ndarray  # weight of one pass in the split of sprayed volume
    cum_passes: np.ndarray  # passes made so far
    exp_passes: np.ndarray  # passes expected so far for this pellet's size
    median_um: float  # actual core median
    n_real: float  # real pellets in the batch
    growth_exponent_a: float
    cycle_exponent_b: float
    cycle_time_s: float
    core3_total: float = 0.0  # sum of core diameter cubed; fusion conserves it

    @property
    def n(self) -> int:
        return self.core_um.size

    @property
    def scale(self) -> float:
        """Simulated pellets per real pellet; sprayed mass is scaled by this."""
        return self.n / self.n_real

    def thickness_um(self) -> np.ndarray:
        return 0.5 * (self.coated - self.core_um)

    def coating_volume_um3(self) -> float:
        """Coating volume on every pellet in the bed (twins hold what they absorbed)."""
        return float(np.pi / 6.0 * (self.coated3.sum() - self.core3_total))

    def core_volume_um3(self) -> float:
        return float(np.pi / 6.0 * self.core3_total)

    def absorb(self, a: np.ndarray, b: np.ndarray) -> None:
        """Fuse pellets b into pellets a; a becomes a twin, b leaves the bed."""
        self.core_um[a] = np.cbrt(self.core_um[a] ** 3 + self.core_um[b] ** 3)
        self.coated3[a] += self.coated3[b]
        self.coated[a] = np.cbrt(self.coated3[a])
        self.kind[a] = TWIN
        self.set_size_terms(a)
        self.kind[b] = ABSORBED
        for arr in (self.core_um, self.coated3, self.coated, self.rate_per_s, self.share_w):
            arr[b] = 0.0

    def set_size_terms(self, idx) -> None:
        """(Re)compute the size-dependent pass rate and share weight for pellets idx."""
        rel = self.core_um[idx] / self.median_um
        self.rate_per_s[idx] = rel ** (-self.cycle_exponent_b) / self.cycle_time_s
        # One pass deposits a thickness ~ D^a over the pellet's surface (~ D^2).
        self.share_w[idx] = rel ** (2.0 + self.growth_exponent_a)


def make_population(cfg: Config) -> Population:
    n = cfg.batch.n_pellets
    median = core_median_um(cfg)
    gen = rng("twin.population", cfg.seed)
    core = median * np.exp(cfg.pellet.size_sigma_log * gen.standard_normal(n))
    core_mass_g = cfg.pellet.core_density_g_cm3 * sphere_volume_um3(core).mean() / UM3_PER_CM3
    b = cfg.wurster.cycle_size_exponent_b
    pop = Population(
        core_um=core,
        coated3=core**3,
        coated=core.copy(),
        kind=np.full(n, SINGLE, dtype=np.int8),
        rate_per_s=np.empty(n),
        share_w=np.empty(n),
        cum_passes=np.zeros(n),
        exp_passes=np.zeros(n),
        median_um=median,
        n_real=cfg.batch.mass_kg * 1000.0 / core_mass_g,
        growth_exponent_a=cfg.wurster.size_growth_exponent_k + b,
        cycle_exponent_b=b,
        cycle_time_s=cfg.wurster.cycle_time_s,
        core3_total=float(np.sum(core**3)),
    )
    pop.set_size_terms(slice(None))
    return pop
