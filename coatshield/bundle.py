"""Everything the dashboard shows for one scenario, in a form that saves to small files.

A bundle is built once from a config (twin + estimators + controllers), either by
scripts/precompute_scenarios.py or live in the app, and stored as Parquet + NPZ + JSON
so the deployed app needs no heavy computation, no torch and no network.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from coatshield.config import ANALYSIS_SECTIONS, Config
from coatshield.estimate.controllers import (
    Bootstrap,
    evaluate_controllers,
    reference_at,
    run_estimators,
    stop_steps,
    window_at,
)
from coatshield.estimate.diagnosis import diagnosis_matrix, signals
from coatshield.estimate.hybrid import corrected_histogram
from coatshield.twin.batch import run_batch, twin_hash

BUNDLE_VERSION = "1"


@dataclass
class Bundle:
    key: str  # config hash this bundle belongs to
    meta: dict
    truth: pd.DataFrame  # per step: what the bed truly looks like
    est: pd.DataFrame  # per step: every estimator
    controllers: pd.DataFrame  # per controller: outcome against the truth
    signals: pd.DataFrame  # per step: diagnosis signals and verdicts
    samples: pd.DataFrame  # a thin, evenly spaced subset of the window records
    frame_steps: np.ndarray  # steps at which the distributions are stored
    thickness_edges: np.ndarray
    dist_truth: np.ndarray  # [frame, bin] shares
    dist_raw: np.ndarray
    dist_corrected: np.ndarray
    measured_cum: np.ndarray  # [step] pellets measured so far


def bundle_key(cfg: Config) -> str:
    """Hash of everything a bundle depends on: the analysed batch and the app's settings."""
    return cfg.hash(include=ANALYSIS_SECTIONS + ("diagnosis", "app"))


def _shares(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    return counts / total if total > 0 else counts.astype(float)


def build_bundle(cfg: Config, bootstrap: Bootstrap = "all", cache: bool = False) -> Bundle:
    result = run_batch(cfg, cache=cache)
    est = run_estimators(result, cfg, bootstrap=bootstrap)
    outcome = evaluate_controllers(result, est, cfg)
    stops = stop_steps(result, est, cfg)
    sig = signals(result, est, cfg)
    sig = sig.merge(diagnosis_matrix(sig, cfg), on="t_h")

    n_steps = len(result.truth)
    frames = np.unique(np.concatenate([
        np.arange(cfg.app.frame_every - 1, n_steps, cfg.app.frame_every),
        [s for s in stops.values() if s is not None],
        [n_steps - 1],
    ])).astype(int)
    edges = result.thickness_edges
    growth = est.growth_rel_per_h.to_numpy() / 3600.0
    raw = np.zeros((frames.size, edges.size - 1))
    corrected = np.zeros_like(raw)
    for j, step in enumerate(frames):
        size, thick = window_at(result, cfg, step, growth[step])
        if size.size >= cfg.controller.min_objects:
            raw[j] = _shares(np.histogram(thick, bins=edges)[0])
            ref = reference_at(result, cfg, step, size, thick)
            corrected[j] = corrected_histogram(size, thick, ref, cfg, edges)

    s = result.samples
    accepted_cum = np.cumsum(s.accepted.to_numpy())
    ends = np.searchsorted(s.t_s.to_numpy(), result.truth.t_s.to_numpy(), side="right")
    thin = s.iloc[:: max(1, len(s) // cfg.app.sample_rows)].reset_index(drop=True)

    return Bundle(
        key=bundle_key(cfg),
        meta={
            "bundle_version": BUNDLE_VERSION,
            "config": cfg.to_dict(),
            "config_hash": cfg.analysis_hash(),
            "twin_hash": twin_hash(cfg),
            "n_pellets": result.meta["n_pellets"],
            "n_real": float(result.meta["n_real"]),
            "core_median_um": float(result.meta["core_median_um"]),
            "target_solids_g": result.recipe.target_solids_g,
            "target_weight_gain_pct": result.recipe.target_weight_gain_pct,
            "stop_steps": {k: (int(v) if v is not None else None) for k, v in stops.items()},
            "bootstrap": bootstrap,
        },
        truth=result.truth,
        est=est,
        controllers=outcome.reset_index(),
        signals=sig,
        samples=thin,
        frame_steps=frames,
        thickness_edges=edges,
        dist_truth=np.vstack([_shares(result.thickness_hist[f]) for f in frames]),
        dist_raw=raw,
        dist_corrected=corrected,
        measured_cum=accepted_cum[np.maximum(ends - 1, 0)].astype(np.int64),
    )


_FRAMES = ("truth", "est", "controllers", "signals", "samples")


def save_bundle(bundle: Bundle, directory: Path) -> Path:
    path = Path(directory) / bundle.key
    path.mkdir(parents=True, exist_ok=True)
    for name in _FRAMES:
        getattr(bundle, name).to_parquet(path / f"{name}.parquet", index=False)
    np.savez_compressed(
        path / "arrays.npz",
        frame_steps=bundle.frame_steps,
        thickness_edges=bundle.thickness_edges.astype(np.float32),
        dist_truth=bundle.dist_truth.astype(np.float32),
        dist_raw=bundle.dist_raw.astype(np.float32),
        dist_corrected=bundle.dist_corrected.astype(np.float32),
        measured_cum=bundle.measured_cum,
    )
    (path / "meta.json").write_text(json.dumps(bundle.meta, indent=1, sort_keys=True))
    return path


def load_bundle(key: str, directory: Path) -> Bundle | None:
    path = Path(directory) / key
    if not (path / "meta.json").exists():
        return None
    meta = json.loads((path / "meta.json").read_text())
    if meta.get("bundle_version") != BUNDLE_VERSION:
        return None
    arrays = np.load(path / "arrays.npz")
    frames = {name: pd.read_parquet(path / f"{name}.parquet") for name in _FRAMES}
    return Bundle(key=key, meta=meta, **frames, **{k: arrays[k] for k in arrays.files})
