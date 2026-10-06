"""Validation harness: the m x k x gamma grid and the fault scenarios, one batch per case."""

from __future__ import annotations

import numpy as np
import pandas as pd

from coatshield.config import FAULT_SCENARIOS, Config, load_config
from coatshield.estimate.controllers import CONTROLLERS, evaluate_controllers, run_estimators
from coatshield.twin.batch import run_batch

ESTIMATORS = ("raw", "ipw", "model", "hybrid")
OUTCOMES = ("stop_h", "true_below_spec_pct", "true_d10_um", "true_mean_um", "believed_d10_um",
            "excess_coating_pct", "agglomerate_pct")


def grid_cases(cfg: Config, quick: bool = False) -> list[dict]:
    """One case per (m, k, gamma, seed): overrides for load_config plus labels."""
    vc = cfg.validation
    m_grid = vc.quick_m_grid if quick else vc.m_grid
    k_grid = vc.quick_k_grid if quick else vc.k_grid
    n_seeds = vc.quick_n_seeds if quick else vc.n_seeds
    return [
        {"kind": "grid", "scenario": "none", "m": m, "k": k, "gamma": g, "seed": cfg.seed + i}
        for m in m_grid
        for k in k_grid
        for g in vc.gamma_grid
        for i in range(n_seeds)
    ]


def fault_cases(cfg: Config, quick: bool = False) -> list[dict]:
    """Each fault scenario (and the fault-free batch) at the default m, k, gamma."""
    vc = cfg.validation
    n_seeds = vc.quick_n_seeds if quick else vc.fault_seeds
    return [
        {"kind": "fault", "scenario": sc, "m": cfg.window.size_bias_m,
         "k": cfg.wurster.size_growth_exponent_k, "gamma": cfg.window.hidden_selection_gamma,
         "seed": cfg.seed + i}
        for sc in FAULT_SCENARIOS
        for i in range(n_seeds)
    ]


def case_config(case: dict, base: Config) -> Config:
    return base.with_overrides(
        {
            "seed": case["seed"],
            "batch.n_pellets": base.validation.n_pellets,
            "window.size_bias_m": case["m"],
            "wurster.size_growth_exponent_k": case["k"],
            "window.hidden_selection_gamma": case["gamma"],
            "fault.scenario": case["scenario"],
        }
    )


def run_case(case: dict, base_json: str | None = None) -> dict:
    """Simulate one batch, estimate, run the controllers, and score against the truth.

    base_json is the base configuration as JSON (a snapshot taken when the run started,
    so a long run is not affected by later edits to the YAML); default: the config on disk.
    """
    base = Config.model_validate_json(base_json) if base_json else load_config()
    cfg = case_config(case, base)
    result = run_batch(cfg, cache=False)
    est = run_estimators(result, cfg, bootstrap="needed")
    outcome = evaluate_controllers(result, est, cfg)

    row = dict(case)
    truth = result.truth
    scored = (truth.t_h >= cfg.validation.eval_from_h).to_numpy()
    reached = (truth.d10_um >= cfg.spec.d10_min_um).to_numpy()
    ideal = int(np.argmax(reached)) if reached.any() else None
    row["ideal_stop_h"] = float(truth.t_h.iloc[ideal]) if ideal is not None else np.nan
    for name in ESTIMATORS:
        err = (est[f"{name}_d10"] - truth.d10_um).to_numpy()
        row[f"{name}_d10_mae"] = float(np.nanmedian(np.abs(err[scored])))
        row[f"{name}_d10_bias"] = float(np.nanmean(err[scored]))
        row[f"{name}_d10_err_at_spec"] = float(err[ideal]) if ideal is not None else np.nan
    for ctrl in CONTROLLERS:
        row[f"{ctrl}_stopped"] = bool(outcome.loc[ctrl, "stopped"])
        for key in OUTCOMES:
            row[f"{ctrl}_{key}"] = float(outcome.loc[ctrl, key])
    return row


def summarise_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Mean over seeds for every (m, k, gamma) cell."""
    cols = [c for c in df.columns if c not in ("kind", "scenario", "m", "k", "gamma", "seed")]
    return df.groupby(["gamma", "m", "k"])[cols].mean().reset_index()


def summarise_faults(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c not in ("kind", "scenario", "m", "k", "gamma", "seed")]
    order = [s for s in FAULT_SCENARIOS if s in set(df.scenario)]
    return df.groupby("scenario")[cols].mean().loc[order].reset_index()
