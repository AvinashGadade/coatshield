"""Replay data for the Live Console: one simulated batch as an operator screen would see it.

For each scenario the twin is run with the fitted measurement-chain error model and the
estimators, and four tables are written: events (window detections), timeline (one row per
estimator update), stops (when each rule would stop) and meta. A shared library of
representative scans (simulated, segmented by the trained boundary finder) backs the
hover card. Everything is a simulated replay; same seed, same files.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from coatshield.chain.error_model import config_with_error_model
from coatshield.compliance import drift, registry
from coatshield.config import REPO_ROOT, Config
from coatshield.estimate.controllers import (
    CONTROLLERS,
    evaluate_controllers,
    reference_at,
    run_estimators,
    stop_steps,
    window_at,
)
from coatshield.estimate.diagnosis import _trailing_slope
from coatshield.estimate.hybrid import corrected_histogram
from coatshield.estimate.weights import classify
from coatshield.seeds import rng
from coatshield.twin.batch import run_batch
from coatshield.twin.population import FINES, SINGLE, TWIN

ERROR_MODEL = "models/error_model.json"
SCENARIOS = {
    "default": {"label": "Default: window favours big pellets (m = 3)", "overrides": {}},
    "no_bias": {"label": "No window bias (m = 0)", "overrides": {"window.size_bias_m": 0.0}},
    "window_fouling": {"label": "Fault: window fouling",
                       "overrides": {"fault.scenario": "window_fouling"}},
    "spray_drying": {"label": "Fault: spray-drying",
                     "overrides": {"fault.scenario": "spray_drying"}},
}
RULE_NAMES = {"C0": "gravimetric", "C1": "raw_mean", "C2": "raw_d10", "C3": "coatshield"}
GATE_LABELS = ("single", "agglomerate", "fines", "held_back")
HEADLINE_SOURCE = "reports/final/final_numbers.json"


def headline() -> dict:
    """The validation averages quoted on screen (realistic measurement chain), from the
    final report's numbers file."""
    path = REPO_ROOT / HEADLINE_SOURCE
    if not path.exists():
        return {}
    final = json.loads(path.read_text())["headline_chain_error_model"]
    return {"raw_d10_below_spec_pct": round(final["rules"]["C2"]["true_below_spec_pct"], 1),
            "coatshield_below_spec_pct": round(final["rules"]["C3"]["true_below_spec_pct"], 1),
            "ratio": round(final["raw_over_coatshield"], 1), "source": HEADLINE_SOURCE}


def scenario_config(cfg: Config, name: str) -> Config:
    """The batch the console replays: dashboard size, measured through the error model."""
    over = {**SCENARIOS[name]["overrides"], "batch.n_pellets": cfg.app.n_pellets}
    return config_with_error_model(cfg.with_overrides(over), ERROR_MODEL)


def _clean(values) -> list:
    """JSON-ready list: NaN becomes null, numbers are rounded."""
    out = []
    for v in np.asarray(values).tolist():
        if isinstance(v, float):
            out.append(None if v != v else round(v, 4))
        else:
            out.append(v)
    return out


def _chain_draws(n: int, gen: np.random.Generator) -> dict[str, np.ndarray]:
    """Per-object values the error model does not carry (gate confidence, per-pellet index,
    confidence score), drawn from what the full chain measured on the gallery objects."""
    gallery = pd.read_parquet(REPO_ROOT / "app" / "assets" / "gallery.parquet")
    measured = gallery[gallery.status == "measured"]
    undecided = gallery[gallery.status == "undecided"]
    held = gallery[gallery.status == "gated"]

    def draw(series: pd.Series) -> np.ndarray:
        pool = series.dropna().to_numpy()
        return pool[gen.integers(0, pool.size, n)] if pool.size else np.full(n, np.nan)

    return {"n_est": draw(measured.n_reflectance), "conf_measured": draw(measured.confidence),
            "conf_undecided": draw(undecided.confidence),
            "gate_pass": draw(measured.gate_confidence), "gate_held": draw(held.gate_confidence)}


def build_scenario(cfg: Config, name: str, scans: pd.DataFrame | None = None) -> dict:
    """events, timeline, stops and meta of one scenario, as JSON-ready dictionaries."""
    scen = scenario_config(cfg, name)
    result = run_batch(scen, cache=False)
    est = run_estimators(result, scen, bootstrap="all")
    outcome = evaluate_controllers(result, est, scen)
    stops = stop_steps(result, est, scen)
    truth, s = result.truth, result.samples
    method = scen.estimator.method
    spec = scen.spec.d10_min_um
    gen = rng(f"console.{name}", scen.seed)
    draws = _chain_draws(len(s), gen)

    gate = s.gate_class.to_numpy()
    true_class = s.true_class.to_numpy()
    accepted = s.accepted.to_numpy()
    scanned = gate == SINGLE
    undecided = scanned & ~accepted
    label = np.where(gate == FINES, 2, np.where((gate == TWIN) & (true_class == SINGLE), 3,
                                                np.where(gate == TWIN, 1, 0)))
    n_est = np.where(accepted, draws["n_est"], np.nan)
    conf = np.where(accepted, draws["conf_measured"],
                    np.where(undecided, draws["conf_undecided"], np.nan))
    gate_conf = np.where(scanned, draws["gate_pass"], draws["gate_held"])

    # --- timeline: one row per estimator update ---------------------------------------
    t = truth.t_s.to_numpy()
    ends = np.searchsorted(s.t_s.to_numpy(), t, side="right")

    def cumulative(mask: np.ndarray) -> np.ndarray:
        return np.concatenate([[0], np.cumsum(mask)])[ends]

    dt = float(t[1] - t[0])
    win = max(1, int(round(scen.estimator.window_min * 60.0 / dt)))
    acc_n = np.concatenate([[0.0], np.cumsum(np.where(accepted, 1.0, 0.0))])
    acc_sum = np.concatenate([[0.0], np.cumsum(np.where(accepted, np.nan_to_num(n_est), 0.0))])
    acc_sq = np.concatenate([[0.0], np.cumsum(np.where(accepted, np.nan_to_num(n_est) ** 2, 0.0))])
    lo = ends[np.maximum(np.arange(t.size) - win, 0)]
    count = np.maximum(acc_n[ends] - acc_n[lo], 1.0)
    n_pooled = (acc_sum[ends] - acc_sum[lo]) / count
    n_var = np.maximum((acc_sq[ends] - acc_sq[lo]) / count - n_pooled**2, 0.0)
    d10 = est[f"{method}_d10"].to_numpy()
    eta_steps = max(3, int(round(scen.console.eta_window_min * 60.0 / dt)))
    slope = _trailing_slope(t / 3600.0, d10, eta_steps)  # um per hour
    with np.errstate(invalid="ignore", divide="ignore"):
        eta = np.where(d10 >= spec, 0.0,
                       np.where(slope > 0, (spec - d10) / slope * 3600.0, np.nan))
    detections = np.diff(np.concatenate([[0], ends])) / (dt / 60.0)
    timeline = {
        "t_s": _clean(t),
        "raw_mean": _clean(est.raw_mean), "raw_d10": _clean(est.raw_d10),
        "corr_d10": _clean(d10), "corr_d10_lo": _clean(est.d10_lo),
        "corr_d10_hi": _clean(est.d10_hi), "corr_d50": _clean(est[f"{method}_d50"]),
        "corr_d90": _clean(est[f"{method}_d90"]), "p_d10_ge_spec": _clean(est.p_spec),
        "cv": _clean(est[f"{method}_cv"]),
        "n_accepted": _clean(cumulative(accepted)),
        "n_rejected_agglom": _clean(cumulative(label == 1)),
        "n_held_back": _clean(cumulative(label == 3)),
        "n_undecided": _clean(cumulative(undecided)),
        "n_fines": _clean(cumulative(gate == FINES)),
        "undecided_rate": _clean(est.undecided_rate),
        "n_pooled": _clean(np.where(acc_n[ends] - acc_n[lo] > 0, n_pooled, np.nan)),
        "n_se": _clean(np.sqrt(n_var / count)),
        "eta_to_spec_s": _clean(eta),
        "detections_per_min": _clean(detections),
        "fouling": _clean(truth.fouling),
        "drift_state": drift_states(est.undecided_rate.to_numpy(), scen),
        "true_d10": _clean(truth.d10_um),  # hidden unless "reveal" is on
    }

    # --- events: an even subsample of the window's detections ----------------------------
    stride = max(1, int(np.ceil(len(s) / scen.console.max_events)))
    pick = np.arange(0, len(s), stride)
    fouling_at = np.interp(s.t_s.to_numpy()[pick], t, truth.fouling.to_numpy())
    scan_id = np.full(pick.size, -1)
    if scans is not None and len(scans):
        scan_id = nearest_scan(scans, s.thickness_um.to_numpy()[pick],
                               s.true_thickness_um.to_numpy()[pick],
                               s.size_um.to_numpy()[pick], fouling_at,
                               scen.console.scan_match_weights)
        scan_id[~scanned[pick]] = -1
    events = {
        "t_s": _clean(s.t_s.to_numpy()[pick]),
        "size_um": _clean(s.size_um.to_numpy()[pick]),
        "gate_class": [GATE_LABELS[i] for i in label[pick]],
        "gate_conf": _clean(gate_conf[pick]),
        "accepted": accepted[pick].tolist(),
        "undecided": undecided[pick].tolist(),
        "thickness_um": _clean(s.thickness_um.to_numpy()[pick]),
        "n_est": _clean(n_est[pick]),
        "conf": _clean(conf[pick]),
        "scan_id": scan_id.tolist(),
    }

    stop_rows = []
    for code, step in stops.items():
        row = outcome.loc[code]
        stop_rows.append({
            "rule": RULE_NAMES[code], "label": CONTROLLERS[code], "stopped": step is not None,
            "t_s": float(t[step]) if step is not None else None,
            "true_below_spec_pct": round(float(row.true_below_spec_pct), 2),
            "true_d10_um": round(float(row.true_d10_um), 3),
            "believed_d10_um": (None if row.believed_d10_um != row.believed_d10_um
                                else round(float(row.believed_d10_um), 3)),
        })
    meta = {
        "note": "Simulated replay. No claim of real-pellet accuracy.",
        "scenario": name, "label": SCENARIOS[name]["label"], "spec_d10_min_um": spec,
        "seed": scen.seed, "config_hash": scen.analysis_hash(),
        "model_hash": registry.combined_hash(),
        "models": {k: {"version": v["version"], "sha256": v["sha256"], "status": v["status"]}
                   for k, v in registry.active_versions().items()},
        "duration_s": float(t[-1]), "step_s": dt, "event_stride": stride,
        "n_detections": int(len(s)), "n_simulated_pellets": int(result.meta["n_pellets"]),
        "n_real_pellets": float(result.meta["n_real"]),
        "window_share_pct": round(100.0 * float(cumulative(accepted)[-1])
                                  / float(result.meta["n_real"]), 3),
        "pellet_assumption": (f"assumes ~{scen.pellet.core_median_um:.0f} um cores: "
                              f"{result.meta['n_real'] / 1e6:.0f} million pellets in "
                              f"{scen.batch.mass_kg:g} kg"),
        "p_stop": scen.controller.p_d10_min, "undecided_limit": scen.controller.undecided_limit,
        "true_n": scen.coating.n,
        "headline": headline(),
        "drift": {"undecided_warning": scen.drift.undecided_warning,
                  "undecided_alarm": scen.drift.undecided_alarm},
        "sources": {
            "events, timeline, stops": "coatshield/console_export.py (twin + error model "
                                       f"{ERROR_MODEL} + estimators)",
            "gate_conf, n_est, conf": "drawn from the full chain's results in "
                                      "app/assets/gallery.parquet",
            "headline": HEADLINE_SOURCE,
        },
    }
    return {"events": events, "timeline": timeline, "stops": stop_rows, "meta": meta,
            "batch": batch_frames(result, est, scen)}


def drift_states(undecided_rate: np.ndarray, cfg: Config) -> list[str]:
    """Drift monitor state per estimator update, from its undecided-rate channel: the rolling
    share of undecided scans against the monitor's warning and alarm limits."""
    rate = np.nan_to_num(undecided_rate, nan=0.0)
    rank = np.where(rate >= cfg.drift.undecided_alarm, 2,
                    np.where(rate >= cfg.drift.undecided_warning, 1, 0))
    return [(drift.OK, drift.WARNING, drift.ALARM)[i] for i in rank]


def batch_frames(result, est: pd.DataFrame, cfg: Config) -> dict:
    """Snapshots for the Batch tab: window against bed size shares, the weights between
    them, and the raw, corrected and true thickness distributions."""
    cc, ec = cfg.console, cfg.estimator
    n_steps = len(result.truth)
    steps = np.unique(np.linspace(0, n_steps - 1, cc.batch_frames).round().astype(int))
    merge = cc.thickness_merge
    n_merged = (result.thickness_edges.size - 1) // merge
    edges = result.thickness_edges[: n_merged * merge + 1 : merge]
    growth = est.growth_rel_per_h.to_numpy() / 3600.0

    def merged(shares: np.ndarray) -> np.ndarray:
        return shares[: n_merged * merge].reshape(n_merged, merge).sum(axis=1)

    frames = []
    for step in steps:
        size, thick = window_at(result, cfg, int(step), float(np.nan_to_num(growth[step])))
        frame: dict = {"t_s": float(result.truth.t_s.iloc[step]), "n_window": int(size.size)}
        true_counts = result.thickness_hist[step].astype(float)
        frame["truth"] = _clean(merged(true_counts / max(true_counts.sum(), 1.0)))
        if size.size >= cfg.controller.min_objects:
            ref = reference_at(result, cfg, int(step), size, thick)
            classes = classify(np.log(size), ref, ec.n_size_bins)
            f_obs = classes.n_obs / classes.n_obs.sum()
            with np.errstate(invalid="ignore", divide="ignore"):
                weight = np.where(f_obs > 0, classes.f_true / f_obs, np.nan)
            raw = np.histogram(thick, bins=result.thickness_edges)[0].astype(float)
            frame.update(
                size_um=_clean(np.exp(classes.lo + (np.arange(classes.n_bins) + 0.5)
                                      * classes.width)),
                f_obs=_clean(f_obs), f_true=_clean(classes.f_true), weight=_clean(weight),
                raw=_clean(merged(raw / raw.sum())),
                corrected=_clean(merged(corrected_histogram(size, thick, ref, cfg,
                                                            result.thickness_edges))))
        frames.append(frame)
    return {"thickness_um": _clean(0.5 * (edges[:-1] + edges[1:])), "frames": frames,
            "reference": ec.reference}


def nearest_scan(scans: pd.DataFrame, measured, true_thickness, size, fouling,
                 weights: tuple[float, float, float]) -> np.ndarray:
    """Index of the representative scan closest to each event in thickness, size and fouling."""
    w_thickness, w_size, w_fouling = weights
    thickness = np.where(np.isfinite(measured), measured, true_thickness)
    thickness = np.nan_to_num(thickness, nan=float(scans.thickness_um.median()))
    cost = (np.abs(np.log(np.maximum(thickness[:, None], 0.5)
                          / scans.thickness_um.to_numpy()[None, :])) * w_thickness
            + np.abs(size[:, None] - scans.diameter_um.to_numpy()[None, :]) * w_size
            + np.abs(fouling[:, None] - scans.fouling.to_numpy()[None, :]) * w_fouling)
    return scans.scan_id.to_numpy()[np.argmin(cost, axis=1)]


def write_scenario(data: dict, directory: Path) -> None:
    """Write the five JSON files, and one data.js so the page also opens from file://."""
    directory.mkdir(parents=True, exist_ok=True)
    for key in ("events", "timeline", "stops", "meta", "batch"):
        (directory / f"{key}.json").write_text(json.dumps(data[key], separators=(",", ":")))
    name = data["meta"]["scenario"]
    (directory / "data.js").write_text(
        "window.CONSOLE_DATA=window.CONSOLE_DATA||{};"
        f"window.CONSOLE_DATA[{json.dumps(name)}]={json.dumps(data, separators=(',', ':'))};")


# --- representative scans -----------------------------------------------------------------

SURFACE_TOP_BGR = (57, 40, 198)  # red, the brief's "stop / fail" colour, for the top surface
SURFACE_BOTTOM_BGR = (212, 190, 40)  # cyan for the coating-core surface
_UPSCALE = 3  # the 128 A-scans are drawn three pixels wide
_SIGNAL_POINTS = 512  # detector pixels kept, from the centre of the line


def scan_grid(cfg: Config) -> list[dict]:
    """Objects of the scan library: a grid over thickness, pellet size and window fouling."""
    cc = cfg.console
    lo, hi = cc.scan_thickness_um
    jobs = []
    for thickness in np.exp(np.linspace(np.log(lo), np.log(hi), cc.scan_thickness_steps)):
        for diameter in cc.scan_diameters_um:
            for fouling in cc.scan_fouling:
                for _ in range(cc.scan_repeats):
                    jobs.append({"scan_id": len(jobs), "thickness_um": float(thickness),
                                 "diameter_um": float(diameter), "fouling": float(fouling)})
    return jobs


def draw_scan(scan: np.ndarray, outer, inner, valid, rows: int) -> np.ndarray:
    """BGR picture [rows, 3 x A-scans] of a stored scan with both surfaces drawn on."""
    import cv2

    image = np.repeat(scan.T[:rows], _UPSCALE, axis=1)
    picture = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_GRAY2BGR)
    cols = np.flatnonzero(valid)
    for line, colour in ((outer, SURFACE_TOP_BGR), (inner, SURFACE_BOTTOM_BGR)):
        if cols.size > 1:
            points = np.column_stack([cols * _UPSCALE + _UPSCALE // 2,
                                      np.asarray(line)[cols]]).astype(np.int32)
            cv2.polylines(picture, [points], False, colour, 2, cv2.LINE_AA)
    return picture


def make_scan(job: dict, cfg: Config, segmenter, directory: Path) -> dict:
    """Simulate, segment and solve one representative pellet; write its pictures and return
    its row, with the centre A-scan's signal under "signal"."""
    import cv2

    from coatshield.chain.pipeline import SampledObject, process_object
    from coatshield.oct import physics
    from coatshield.oct.preprocess import to_wavenumber

    sid = job["scan_id"]
    obj = SampledObject(
        true_class=SINGLE, diameter_um=job["diameter_um"], thickness_um=job["thickness_um"],
        n_coat=cfg.coating.n, n_core=cfg.core.n,
        speed_m_s=float(np.mean(cfg.oct_dataset.speed_m_s)), snr_db=cfg.chain.operating_snr_db,
        pigment=cfg.chain.operating_pigment, fouling=job["fouling"],
        standoff_um=float(np.mean(cfg.oct.standoff_um)),
        seed=cfg.console.scan_seed_base + sid)
    rec = process_object(obj, cfg, segmenter, cfg.coating.n)
    it = rec.intermediates
    stem = directory / f"scan_{sid:04d}"
    quality = [cv2.IMWRITE_JPEG_QUALITY, cfg.console.jpeg_quality]
    cv2.imwrite(f"{stem}_camera.jpg", cv2.resize(it["camera"], (128, 128),
                                                 interpolation=cv2.INTER_AREA), quality)
    row = {**job, "status": rec.status, "gate_class": rec.gate_class,
           "gate_conf": round(rec.gate_confidence, 3), "has_scan": "scan" in it}
    if "scan" not in it:
        return row
    rows = cfg.console.scan_rows
    cv2.imwrite(f"{stem}.jpg", draw_scan(it["scan"], it["outer_px"], it["inner_px"], it["valid"],
                                         rows), quality)
    np.save(f"{stem}_input.npy", it["scan"])  # for the Grad-CAM pass; removed afterwards
    centre = it["scan"].shape[0] // 2
    lo, hi = cfg.oct.db_range
    spec = physics.spectrometer(cfg.oct)
    raw = it["spectrum"].astype(float)
    fringes = to_wavenumber((raw - it["background"])[None, :], spec)[0]
    mid = fringes.size // 2
    part = slice(mid - _SIGNAL_POINTS // 2, mid + _SIGNAL_POINTS // 2)
    wavelength_nm = spec.wavelength_um[part] * 1000.0
    row["signal"] = {
        # The centre of the detector line at full resolution: raw counts with the recorded
        # background, then the fringes after background removal and resampling to even
        # wavenumber.
        "spectrum": [round(float(v), 4) for v in raw[part]],
        "background": [round(float(v), 4) for v in it["background"][part]],
        "fringes_k": [round(float(v), 4) for v in fringes[part]],
        "ascan_db": [round(float(v), 2) for v in
                     it["scan"][centre, :rows].astype(float) / 255.0 * (hi - lo) + lo],
        "outer_row": float(it["outer_px"][centre]), "inner_row": float(it["inner_px"][centre]),
        "depth_px_um": float(it["depth_px_um"]),
        "wavelength_nm": [round(float(wavelength_nm[0]), 1), round(float(wavelength_nm[-1]), 1)],
    }
    measured = rec.status in ("measured", "undecided")
    row.update(
        measured_um=round(rec.thickness_um, 3) if measured else None,
        optical_um=round(rec.optical_um, 3) if measured else None,
        n_reflectance=(round(rec.n_reflectance, 4)
                       if measured and rec.n_reflectance == rec.n_reflectance else None),
        conf=round(rec.confidence, 3), radius_um=round(rec.radius_um, 1) if measured else None)
    return row
