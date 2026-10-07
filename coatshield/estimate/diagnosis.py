"""Process diagnosis from what the window can see: signals and a rules table.

Signals use only observable quantities (estimates, gate counts, the at-line sizing and
the recipe). Each is compared with what a healthy batch would show: the recipe's growth
rate, the batch's own first hour for fines, and for the size-adjusted spread the
sqrt(1/t) level that the process parameters predict (the "golden batch" level).
No verdict is given during the warm-up, while the coat is too thin to measure well.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from coatshield.config import Config
from coatshield.twin.batch import BatchResult


@dataclass(frozen=True)
class Rule:
    fault: str
    label: str
    signature: str
    action: str


RULES = (
    Rule("over_wetting", "Over-wetting",
         "Agglomerate share high",
         "Lower the spray rate or raise inlet air temperature and flow; check atomising air."),
    Rule("spray_drying", "Spray-drying",
         "Fines rate high and growth low",
         "Lower inlet air temperature or atomising pressure; move the nozzle closer; "
         "raise the spray rate."),
    Rule("nozzle_block", "Nozzle partial block",
         "Growth low, fines and agglomerates normal",
         "Check the spray rate against the pump setpoint; inspect and clean the nozzle."),
    Rule("maldistribution", "Maldistribution",
         "Size-adjusted spread above the sqrt(1/t) expectation, agglomerates normal",
         "Check the partition gap, fluidising air and distributor plate for uneven circulation. "
         "(At 1 um measurement noise this signature is too weak to separate from a healthy "
         "batch, so the rule's limit is set where it does not fire on healthy batches.)"),
    Rule("window_fouling", "Window fouling",
         "Undecided share high",
         "Purge or clean the measurement window; hold stop decisions until the share recovers."),
    Rule("substrate_shift", "Substrate shift",
         "Core size above the certificate",
         "Re-check the core lot's size; a weight-gain target calculated for nominal cores "
         "will over- or under-coat."),
)
RULES_BY_FAULT = {r.fault: r for r in RULES}


def _trailing_slope(t_h: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """Least-squares slope of y against t over the last n points (NaN until 3 are finite)."""
    out = np.full(y.size, np.nan)
    for i in range(y.size):
        lo = max(i - n + 1, 0)
        tt, yy = t_h[lo : i + 1], y[lo : i + 1]
        ok = np.isfinite(yy)
        if np.count_nonzero(ok) >= 3:
            dt = tt[ok] - tt[ok].mean()
            out[i] = np.sum(dt * (yy[ok] - yy[ok].mean())) / np.sum(dt * dt)
    return out


def _trailing_mean(y: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(y).rolling(n, min_periods=1).mean().to_numpy()


def expected_growth_um_per_h(result: BatchResult, cfg: Config, mean_coat_um) -> np.ndarray:
    """Growth the recipe should give on nominal cores, at the current coat thickness."""
    med = cfg.pellet.core_median_um
    s2 = cfg.pellet.size_sigma_log**2
    k = cfg.wurster.size_growth_exponent_k
    recipe = result.recipe
    volume_um3_per_h = (recipe.solids_g_per_min * 60.0 * cfg.batch.spray_efficiency
                        / cfg.coating.density_g_cm3 * 1e12)
    area_um2 = recipe.n_real_nominal * np.pi * med**2 * np.exp(2.0 * s2)
    thin = volume_um3_per_h / area_um2 * np.exp(-2.0 * k * s2)  # number-mean of h ~ D^k
    return thin / (1.0 + 2.0 * np.asarray(mean_coat_um, dtype=float) / med) ** 2


def expected_spread2_per_h(cfg: Config) -> float:
    """Healthy size-adjusted CV^2 after one hour, from the cycle-time parameters.

    Each step a pellet's passes have relative variance tau / dt + rsd^2, and steps add
    independently, so CV^2(t) = (tau + rsd^2 * dt) / t.
    """
    w = cfg.wurster
    return (w.cycle_time_s + w.cycle_time_rsd**2 * cfg.batch.step_s) / 3600.0


def signals(result: BatchResult, est: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Per-step diagnosis signals, each with the ratio to its healthy value."""
    dc = cfg.diagnosis
    t_h = est.t_h.to_numpy()
    dt_h = float(t_h[1] - t_h[0]) if t_h.size > 1 else cfg.batch.step_s / 3600.0
    n = max(3, int(round(dc.window_min / 60.0 / dt_h)))
    base = t_h <= dc.baseline_h
    mean_coat = est[f"{cfg.estimator.method}_mean"].to_numpy()

    growth = _trailing_slope(t_h, mean_coat, n)
    growth_ratio = growth / expected_growth_um_per_h(result, cfg, np.nan_to_num(mean_coat))

    fines = _trailing_mean(est.fines_per_min.to_numpy(), n)
    fines_base = np.nanmean(est.fines_per_min.to_numpy()[base]) if base.any() else np.nan

    # Under the sqrt(1/t) law, spread^2 * t stays level. Averaging the unclipped
    # variance estimate keeps the signal unbiased while the coat is still thin.
    level2 = _trailing_mean(est.cv2_within_unclipped.to_numpy() * t_h, n)
    spread_ratio = (np.sqrt(np.maximum(level2, 0.0) / expected_spread2_per_h(cfg))
                    / dc.spread_reference_ratio)
    spread_ratio[t_h < dc.spread_from_h] = np.nan

    core_est = est.ref_median_um.to_numpy() - 2.0 * mean_coat
    return pd.DataFrame(
        {
            "t_h": t_h,
            "ready": t_h > 2.0 * dc.baseline_h,
            "growth_um_per_h": growth,
            "growth_ratio": growth_ratio,
            "agglomerate_pct": _trailing_mean(est.gate_twin_pct.to_numpy(), n),
            "fines_per_min": fines,
            "fines_ratio": fines / fines_base,
            "spread_ratio": spread_ratio,
            "undecided_rate": est.undecided_rate.to_numpy(),
            "core_shift_um": _trailing_mean(core_est, n) - cfg.pellet.core_median_um,
        }
    )


def diagnose(row: pd.Series, cfg: Config) -> list[Rule]:
    """Rules whose signature matches one row of signals (empty list = no fault seen)."""
    dc = cfg.diagnosis

    def above(value, limit) -> bool:
        return bool(np.isfinite(value) and value > limit)

    if not row.ready:
        return []
    growth_low = bool(np.isfinite(row.growth_ratio) and row.growth_ratio < dc.growth_low_factor)
    agglomerates = above(row.agglomerate_pct, dc.agglomerate_high_pct)
    fines = above(row.fines_ratio, dc.fines_high_factor)
    found = []
    if agglomerates:
        found.append("over_wetting")
    if fines and growth_low:
        found.append("spray_drying")
    if growth_low and not fines and not agglomerates:
        found.append("nozzle_block")
    if above(row.spread_ratio, dc.spread_high_factor) and not agglomerates:
        found.append("maldistribution")
    if above(row.undecided_rate, dc.undecided_high):
        found.append("window_fouling")
    if above(row.core_shift_um, dc.size_shift_um):
        found.append("substrate_shift")
    return [RULES_BY_FAULT[f] for f in found]


def diagnosis_matrix(sig: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Per step: one boolean column per fault."""
    verdicts = [{r.fault for r in diagnose(row, cfg)} for row in sig.itertuples(index=False)]
    out = pd.DataFrame({"t_h": sig.t_h})
    for rule in RULES:
        out[rule.fault] = [rule.fault in v for v in verdicts]
    return out
