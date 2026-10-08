"""Phase 8 consistency check: does the error model stand in for the full chain?

A reduced batch is simulated once. Every object its window sampled is then measured twice:
by the full chain (camera image, gate, OCT scan, boundary finder, solver) and by the fitted
error model. The estimators and stopping rules run on both, and their stop times and d10
estimates are compared. The short batch coats faster than a real one so that it reaches
spec within the hour.
"""

from __future__ import annotations

import argparse
import copy
import os
from multiprocessing import get_context

for _var in ("NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _common import REPORTS_DIR  # noqa: E402

from coatshield.chain.error_model import load_error_model  # noqa: E402
from coatshield.chain.pipeline import SampledObject, process_object  # noqa: E402
from coatshield.compliance.registry import verified_path  # noqa: E402
from coatshield.config import config_from_snapshot, load_config  # noqa: E402
from coatshield.estimate.controllers import run_estimators, stop_steps  # noqa: E402
from coatshield.seeds import rng  # noqa: E402
from coatshield.seg.infer import Segmenter  # noqa: E402
from coatshield.twin.batch import run_batch  # noqa: E402
from coatshield.twin.population import FINES, SINGLE, TWIN  # noqa: E402

ERROR_MODEL = "models/error_model.json"
_STATE: dict = {}


def _init(config_json: str) -> None:
    cfg = config_from_snapshot(config_json)
    _STATE.update(cfg=cfg, seg=Segmenter(verified_path("unet"), cfg))


def _one(job: dict) -> dict:
    cfg = _STATE["cfg"]
    rec = process_object(SampledObject(**job), cfg, _STATE["seg"], cfg.measurement.n_assumed,
                         keep=False)
    gate = SINGLE if rec.status != "gated" else (FINES if rec.gate_class == "fines" else TWIN)
    return {"thickness_um": rec.thickness_um if rec.status == "measured" else np.nan,
            "gate_class": gate, "accepted": rec.status == "measured",
            "confidence": rec.confidence}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = parser.parse_args()
    base = load_config()
    cc = base.chain
    model = load_error_model(ERROR_MODEL)
    cfg = base.with_overrides({
        "batch.n_pellets": cc.consistency_pellets, "batch.duration_h": cc.consistency_hours,
        "window.objects_per_min": cc.consistency_objects_per_min,
        "wurster.deposit_nm_per_pass": base.wurster.deposit_nm_per_pass * cc.consistency_speedup,
        "controller.min_objects": cc.consistency_min_objects,
        "measurement.sigma_um": model.operating_spread_um(base.batch.target_mean_um),
    })
    tag = cfg.hash(include=("seed", "oct", "seg", "solve", "gate_vision", "chain"))
    result = run_batch(cfg, cache=False)
    s = result.samples
    print(f"{len(s)} sampled objects; true d10 at the end {result.truth.d10_um.iloc[-1]:.2f} um")

    gen = rng("chain.consistency", cfg.seed)
    jobs = []
    for i, r in enumerate(s.itertuples(index=False)):
        fines = r.true_class == FINES
        jobs.append(dict(
            true_class=int(r.true_class),
            diameter_um=float(r.true_size_um if fines
                              else np.clip(r.true_size_um, *cfg.gate_vision.diameter_um)),
            thickness_um=float(r.true_thickness_um) if np.isfinite(r.true_thickness_um) else 1.0,
            n_coat=cfg.coating.n, n_core=cfg.core.n,
            speed_m_s=float(gen.uniform(*cfg.oct_dataset.speed_m_s)),
            snr_db=cc.operating_snr_db, pigment=cc.operating_pigment, fouling=0.0,
            standoff_um=float(gen.uniform(*cfg.oct.standoff_um)), seed=900_000 + i))
    with get_context("spawn").Pool(args.workers, initializer=_init,
                                   initargs=(cfg.model_dump_json(),)) as pool:
        chain = pd.DataFrame(pool.map(_one, jobs, chunksize=16))

    modelled = model.measure(cfg, 0.0, s.true_class.to_numpy(),
                             s.true_thickness_um.fillna(0.0).to_numpy(),
                             rng("chain.consistency.model", cfg.seed))
    runs = {}
    for name, columns in (("full chain", chain), ("error model", pd.DataFrame(modelled))):
        res = copy.copy(result)
        res.samples = s.assign(**{c: columns[c].to_numpy() for c in
                                  ("thickness_um", "gate_class", "accepted", "confidence")})
        est = run_estimators(res, cfg, bootstrap="needed")
        runs[name] = (est, stop_steps(res, est, cfg), res.samples)

    (est_c, stops_c, sam_c), (est_m, stops_m, sam_m) = runs["full chain"], runs["error model"]
    method = cfg.estimator.method
    both = est_c[f"{method}_d10"].notna() & est_m[f"{method}_d10"].notna()
    diff = (est_c[f"{method}_d10"] - est_m[f"{method}_d10"])[both]
    d10_gap = float(diff.abs().median())
    lines = [
        "# Full chain against the error model", "",
        f"Config `{tag}` · {cc.consistency_pellets:,} pellets, {cc.consistency_hours:g} h, "
        f"{cc.consistency_objects_per_min:g} objects per minute ({len(s)} objects), coating "
        f"{cc.consistency_speedup:g}x faster than normal · error model `{ERROR_MODEL}`", "",
        "| | Full chain | Error model |", "| --- | --- | --- |",
        f"| Objects accepted | {int(sam_c.accepted.sum())} | {int(sam_m.accepted.sum())} |",
        f"| Median |reading - truth| of accepted singles (um) | "
        f"{(sam_c.thickness_um - sam_c.true_thickness_um)[sam_c.accepted].abs().median():.3f} | "
        f"{(sam_m.thickness_um - sam_m.true_thickness_um)[sam_m.accepted].abs().median():.3f} |",
    ]
    worst_steps = 0
    for name in ("C1", "C2", "C3"):
        a, b = stops_c[name], stops_m[name]
        lines.append(f"| {name} stops at step | {a} | {b} |")
        if a is not None and b is not None:
            worst_steps = max(worst_steps, abs(a - b))
        elif a != b:
            worst_steps = 999
    lines += [
        "", f"Corrected d10, full chain minus error model: median absolute difference "
        f"{d10_gap:.3f} um over {int(both.sum())} time steps (mean {diff.mean():+.3f} um).", "",
        f"Target: stop times within one step and d10 within 0.3 um. Result: stop times differ by "
        f"at most {worst_steps} step(s); d10 differs by {d10_gap:.2f} um: "
        f"**{'met' if worst_steps <= 1 and d10_gap <= 0.3 else 'NOT met'}**.", "",
    ]
    text = "\n".join(lines)
    (REPORTS_DIR / f"chain_consistency_{tag}.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
