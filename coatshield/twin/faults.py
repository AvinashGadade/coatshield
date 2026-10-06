"""Fault scenarios, each switched on at fault.start_h.

substrate_shift is the exception: it is a property of the delivered cores, so it
applies from the start of the batch (see population.core_median_um).
"""

from __future__ import annotations

from dataclasses import dataclass

from coatshield.config import Config


@dataclass(frozen=True)
class FaultState:
    spray_factor: float = 1.0  # multiplies the sprayed solids rate
    efficiency_factor: float = 1.0  # multiplies spray efficiency (solids reaching pellets)
    fusion_factor: float = 1.0  # multiplies the agglomeration rate
    fines_factor: float = 1.0  # multiplies the fines rate
    cycle_rsd: float = 0.0  # cycle-time spread in force
    fouling: float = 0.0  # window fouling, 0 (clean) to 1


def fault_state(cfg: Config, t_s: float) -> FaultState:
    f = cfg.fault
    rsd = cfg.wurster.cycle_time_rsd
    since_h = t_s / 3600.0 - f.start_h
    if f.scenario in ("none", "substrate_shift") or since_h < 0:
        return FaultState(cycle_rsd=rsd)
    if f.scenario == "nozzle_block":
        return FaultState(spray_factor=f.nozzle_block_factor, cycle_rsd=rsd)
    if f.scenario == "over_wetting":
        return FaultState(
            efficiency_factor=f.over_wetting_growth_factor,
            fusion_factor=f.over_wetting_fusion_factor,
            cycle_rsd=rsd,
        )
    if f.scenario == "spray_drying":
        return FaultState(
            efficiency_factor=f.spray_drying_efficiency_factor,
            fines_factor=f.spray_drying_fines_factor,
            cycle_rsd=rsd,
        )
    if f.scenario == "maldistribution":
        return FaultState(cycle_rsd=f.maldistribution_rsd)
    if f.scenario == "window_fouling":
        return FaultState(cycle_rsd=rsd, fouling=min(1.0, since_h / f.fouling_ramp_h))
    raise ValueError(f"Unknown fault scenario: {f.scenario}")
