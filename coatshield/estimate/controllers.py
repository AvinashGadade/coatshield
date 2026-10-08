"""Estimates over the whole batch and the four stopping rules evaluated on them."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from coatshield.config import Config
from coatshield.estimate.hybrid import estimate_window
from coatshield.estimate.model_based import class_sums, fit_trend
from coatshield.estimate.quantiles import weighted_quantile
from coatshield.estimate.weights import (
    Reference,
    atline_reference,
    classify,
    coa_reference,
    oracle_reference,
)
from coatshield.seeds import rng
from coatshield.twin.batch import BatchResult
from coatshield.twin.population import FINES, SINGLE, TWIN

CONTROLLERS = {
    "C0": "Gravimetric",
    "C1": "Raw mean",
    "C2": "Raw d10",
    "C3": "CoatShield",
}
Bootstrap = Literal["all", "needed", "none"]
ESTIMATE_COLUMNS = (
    "n", *(f"{method}_{stat}" for method in ("raw", "ipw", "model", "hybrid")
           for stat in ("mean", "d10", "d50", "d90", "cv")),
    "ipw_ess", "ipw_uncovered", "fit_beta", "cv_within", "cv2_within_unclipped",
    "d10_lo", "d10_hi", "p_spec", "ref_median_um",
)


def _reference(result: BatchResult, step: int, cfg: Config, size, thickness) -> Reference:
    mode = cfg.estimator.reference
    if mode == "oracle":
        return oracle_reference(result, step)
    if mode == "atline":
        return atline_reference(result, step, cfg)
    # Certificate of analysis: coat the core sizes with the trend seen in the window itself.
    log_size = np.log(size)
    own = Reference(log_size, np.full(size.size, 1.0 / size.size), None)
    classes = classify(log_size, own, cfg.estimator.n_size_bins)
    fit = fit_trend(*class_sums(np.ones((1, size.size)), classes.sample_bin, thickness,
                                classes.n_bins), classes, 0.0)
    mean_coat = float(thickness.mean())

    def coat_of_core(log_core: np.ndarray) -> np.ndarray:
        return fit.predict(np.log(np.exp(log_core) + 2.0 * mean_coat))[0]

    return coa_reference(result, coat_of_core)


def _relative_growth(t_s: np.ndarray, level: np.ndarray) -> float:
    """Relative growth rate (1/s) from a straight-line fit of the recent window means."""
    ok = np.isfinite(level)
    if np.count_nonzero(ok) < 3:
        return 0.0
    t, y = t_s[ok], level[ok]
    dt = t - t.mean()
    slope = np.sum(dt * (y - y.mean())) / np.sum(dt * dt)
    return max(float(slope / y.mean()), 0.0) if y.mean() > 0 else 0.0


def window_at(result: BatchResult, cfg: Config, step: int, growth_rel_per_s: float = 0.0):
    """(size, thickness) of the accepted single pellets in the window ending at `step`.

    Older readings are shifted forward by growth_rel_per_s x age when growth_shift is on.
    """
    s = result.samples
    times = result.truth.t_s.to_numpy()
    dt = float(times[1] - times[0]) if times.size > 1 else cfg.batch.step_s
    win_steps = max(1, int(round(cfg.estimator.window_min * 60.0 / dt)))
    t_all = s.t_s.to_numpy()
    lo = np.searchsorted(t_all, times[max(step - win_steps + 1, 0)] - 0.5 * dt, side="left")
    hi = np.searchsorted(t_all, times[step], side="right")
    acc = s.accepted.to_numpy()[lo:hi]
    size = s.size_um.to_numpy()[lo:hi][acc]
    thick = s.thickness_um.to_numpy()[lo:hi][acc]
    if cfg.estimator.growth_shift:
        thick = thick * (1.0 + growth_rel_per_s * (times[step] - t_all[lo:hi][acc]))
    return size, thick


def reference_at(result: BatchResult, cfg: Config, step: int, size, thickness) -> Reference:
    """The reference size distribution the estimator uses at `step`."""
    return _reference(result, step, cfg, size, thickness)


def run_estimators(
    result: BatchResult, cfg: Config, bootstrap: Bootstrap = "needed"
) -> pd.DataFrame:
    """Every estimator at every time step, from the window's sample records only.

    bootstrap: "all" runs the bootstrap at every step; "needed" only where the point
    estimate of d10 has reached spec (the only steps where P(d10 >= spec) can pass the
    controller's threshold); "none" skips it.
    """
    ec = cfg.estimator
    s = result.samples
    t_all = s.t_s.to_numpy()
    size_all = s.size_um.to_numpy()
    thick_all = s.thickness_um.to_numpy()
    accepted = s.accepted.to_numpy()
    gate = s.gate_class.to_numpy()

    times = result.truth.t_s.to_numpy()
    dt = float(times[1] - times[0]) if times.size > 1 else cfg.batch.step_s
    n_steps = times.size
    ends = np.searchsorted(t_all, times, side="right")
    starts = np.concatenate([[0], ends[:-1]])
    win_steps = max(1, int(round(ec.window_min * 60.0 / dt)))
    grow_steps = max(3, int(round(ec.growth_window_min * 60.0 / dt)))

    def per_step(mask: np.ndarray) -> np.ndarray:
        cum = np.concatenate([[0], np.cumsum(mask)])
        return (cum[ends] - cum[starts]).astype(float)

    def rolling(x: np.ndarray) -> np.ndarray:
        cum = np.concatenate([[0.0], np.cumsum(x)])
        lo = np.maximum(np.arange(n_steps) - win_steps + 1, 0)
        return cum[np.arange(n_steps) + 1] - cum[lo]

    n_acc = per_step(accepted)
    sum_thick = np.concatenate([[0.0], np.cumsum(np.where(accepted, thick_all, 0.0))])
    with np.errstate(invalid="ignore", divide="ignore"):
        step_mean = (sum_thick[ends] - sum_thick[starts]) / n_acc
    to_oct = rolling(per_step(gate == SINGLE))
    n_twin = rolling(per_step(gate == TWIN))
    n_fines = rolling(per_step(gate == FINES))
    n_pellets = to_oct + n_twin
    span_min = np.minimum(np.arange(n_steps) + 1, win_steps) * dt / 60.0

    rows = []
    for i in range(n_steps):
        lo = starts[max(i - win_steps + 1, 0)]
        sel = slice(lo, ends[i])
        acc = accepted[sel]
        size, thick, t = size_all[sel][acc], thick_all[sel][acc], t_all[sel][acc]
        g0 = max(i - grow_steps + 1, 0)
        growth = _relative_growth(times[g0 : i + 1], step_mean[g0 : i + 1])
        with np.errstate(invalid="ignore", divide="ignore"):
            row = {
                "t_s": times[i],
                "t_h": times[i] / 3600.0,
                "n_window": float(size.size),
                "undecided_rate": 1.0 - size.size / to_oct[i] if to_oct[i] > 0 else np.nan,
                "gate_twin_pct": 100.0 * n_twin[i] / n_pellets[i] if n_pellets[i] > 0 else np.nan,
                "fines_per_min": n_fines[i] / span_min[i],
                "size_median_um": float(np.median(size)) if size.size else np.nan,
                "growth_rel_per_h": growth * 3600.0,
            }
        if size.size >= cfg.controller.min_objects:
            if ec.growth_shift:
                thick = thick * (1.0 + growth * (times[i] - t))
            ref = _reference(result, i, cfg, size, thick)
            gen = None if bootstrap == "none" else rng(f"estimate.bootstrap.{i}", cfg.seed)
            est = estimate_window(
                size, thick, ref, cfg, boot_gen=gen,
                boot_from_d10=cfg.spec.d10_min_um if bootstrap == "needed" else None,
            )
            row.update(est)
            row["ref_median_um"] = float(np.exp(weighted_quantile(ref.log_size, ref.weight, 0.5)))
        rows.append(row)
    # Every column exists even if no window ever held enough objects to estimate from.
    return pd.DataFrame(rows).reindex(columns=[*rows[0], *(c for c in ESTIMATE_COLUMNS
                                                             if c not in rows[0])])


def stop_steps(result: BatchResult, est: pd.DataFrame, cfg: Config) -> dict[str, int | None]:
    """First time step at which each controller would stop the spray (None = never)."""
    truth = result.truth
    enough = (est.n_window >= cfg.controller.min_objects).to_numpy()
    cc = cfg.controller
    rules = {
        # Today's practice: stop at the sprayed solids calculated for nominal cores.
        "C0": (truth.sprayed_solids_g >= result.recipe.target_solids_g).to_numpy(),
        "C1": enough & (est.raw_mean >= cfg.batch.target_mean_um).to_numpy(),
        "C2": enough & (est.raw_d10 >= cfg.spec.d10_min_um).to_numpy(),
        "C3": enough
        & (est.p_spec > cc.p_d10_min).to_numpy()
        & (est[f"{cfg.estimator.method}_cv"] < cc.cv_target).to_numpy()
        & (est.undecided_rate < cc.undecided_limit).to_numpy(),
    }
    return {name: (int(np.argmax(hit)) if hit.any() else None) for name, hit in rules.items()}


def evaluate_controllers(result: BatchResult, est: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Outcome of each controller on this batch, judged against the hidden truth."""
    truth = result.truth
    spec = cfg.spec.d10_min_um
    reached = (truth.d10_um >= spec).to_numpy()
    ideal = int(np.argmax(reached)) if reached.any() else None
    believed = {
        "C0": pd.Series(np.nan, index=est.index),
        "C1": est.raw_d10,
        "C2": est.raw_d10,
        "C3": est[f"{cfg.estimator.method}_d10"],
    }
    rows = []
    for name, step in stop_steps(result, est, cfg).items():
        stopped = step is not None
        at = truth.iloc[step if stopped else -1]
        excess = np.nan
        if ideal is not None:
            excess = 100.0 * (at.sprayed_solids_g / truth.sprayed_solids_g.iloc[ideal] - 1.0)
        rows.append(
            {
                "controller": name,
                "label": CONTROLLERS[name],
                "stopped": stopped,
                "stop_h": float(at.t_h),
                "true_below_spec_pct": 100.0 * float(at.frac_below_spec),
                "true_d10_um": float(at.d10_um),
                "true_mean_um": float(at.mean_um),
                "believed_d10_um": float(believed[name].iloc[step]) if stopped else np.nan,
                "excess_coating_pct": float(excess),
                "agglomerate_pct": float(at.agglomerate_pct),
                "weight_gain_pct": float(at.weight_gain_pct),
            }
        )
    return pd.DataFrame(rows).set_index("controller")
