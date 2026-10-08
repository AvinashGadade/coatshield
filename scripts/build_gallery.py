"""Precompute chain results for the dashboard: scenario galleries and the drift sequence.

* app/assets/gallery.parquet: about 200 objects per fault scenario, taken from the window
  of that scenario's precomputed batch and pushed through the full chain.
* app/assets/drift.npz: scans at rising window fouling plus a clean baseline, with their
  drift features, so the Drift page can replay a fouling ramp instantly.
"""

from __future__ import annotations

import argparse
import os
from multiprocessing import get_context

for _var in ("NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _common import REPO_ROOT  # noqa: E402

from coatshield.bundle import bundle_key, load_bundle  # noqa: E402
from coatshield.chain.pipeline import SampledObject, process_object  # noqa: E402
from coatshield.compliance.drift import FEATURES  # noqa: E402
from coatshield.compliance.registry import verified_path  # noqa: E402
from coatshield.config import (  # noqa: E402
    FAULT_SCENARIOS,
    Config,
    config_from_snapshot,
    load_config,
)
from coatshield.seeds import rng  # noqa: E402
from coatshield.seg.infer import Segmenter  # noqa: E402
from coatshield.twin.faults import fault_state  # noqa: E402
from coatshield.twin.population import FINES, SINGLE  # noqa: E402

ASSETS = REPO_ROOT / "app" / "assets"
FOULING_LEVELS = np.round(np.arange(0.0, 1.0001, 0.05), 2)
SCANS_PER_LEVEL = 24
BASELINE_SCANS = 240
GALLERY_AFTER_FAULT_H = 2.0  # objects are taken this long after the fault starts
_STATE: dict = {}


def _init(config_json: str) -> None:
    cfg = config_from_snapshot(config_json)
    _STATE.update(cfg=cfg, seg=Segmenter(verified_path("unet"), cfg))


def _one(job: dict) -> dict:
    cfg: Config = _STATE["cfg"]
    obj = SampledObject(**job["obj"])
    row = process_object(obj, cfg, _STATE["seg"], cfg.coating.n, keep=False).summary()
    row.update(job["tags"])
    return row


def _object(cfg: Config, seed: int, **kw) -> dict:
    gen = rng(f"gallery.object.{seed}", cfg.seed)
    base = dict(true_class=SINGLE,
                diameter_um=float(cfg.pellet.core_median_um
                                  * np.exp(cfg.pellet.size_sigma_log * gen.standard_normal())),
                thickness_um=cfg.batch.target_mean_um, n_coat=cfg.coating.n, n_core=cfg.core.n,
                speed_m_s=float(gen.uniform(*cfg.oct_dataset.speed_m_s)),
                snr_db=cfg.chain.operating_snr_db, pigment=cfg.chain.operating_pigment,
                fouling=0.0, standoff_um=float(gen.uniform(*cfg.oct.standoff_um)), seed=seed)
    base.update(kw)
    if base["true_class"] != FINES:  # pellets must fit the camera frame
        base["diameter_um"] = float(np.clip(base["diameter_um"], *cfg.gate_vision.diameter_um))
    return base


def drift_jobs(cfg: Config) -> list[dict]:
    jobs, seed = [], 500_000
    for i in range(BASELINE_SCANS):
        jobs.append({"obj": _object(cfg, seed + i), "tags": {"set": "baseline", "level": 0.0}})
    seed += BASELINE_SCANS
    for level in FOULING_LEVELS:
        for _ in range(SCANS_PER_LEVEL):
            jobs.append({"obj": _object(cfg, seed, fouling=float(level)),
                         "tags": {"set": "ramp", "level": float(level)}})
            seed += 1
    return jobs


def gallery_jobs(cfg: Config) -> list[dict]:
    """Objects as the window of each scenario's precomputed batch saw them."""
    jobs = []
    n = cfg.chain.gallery_per_scenario
    for k, scenario in enumerate(FAULT_SCENARIOS):
        scen = cfg.with_overrides({"fault.scenario": scenario,
                                   "batch.n_pellets": cfg.app.n_pellets})
        bundle = load_bundle(bundle_key(scen), ASSETS / "bundles")
        if bundle is None:
            print(f"  no precomputed bundle for {scenario}: skipped")
            continue
        t_s = (scen.fault.start_h + GALLERY_AFTER_FAULT_H) * 3600.0
        s = bundle.samples
        near = s.iloc[(s.t_s - t_s).abs().argsort().to_numpy()[:n]]
        fouling = fault_state(scen, t_s).fouling
        for j, r in enumerate(near.itertuples(index=False)):
            thickness = r.true_thickness_um if np.isfinite(r.true_thickness_um) else 1.0
            obj = _object(cfg, 700_000 + 1000 * k + j, true_class=int(r.true_class),
                          diameter_um=float(r.true_size_um),
                          thickness_um=float(max(thickness, 0.5)), fouling=float(fouling))
            jobs.append({"obj": obj, "tags": {"set": "gallery", "scenario": scenario}})
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = parser.parse_args()
    cfg = load_config()
    jobs = drift_jobs(cfg) + gallery_jobs(cfg)
    print(f"{len(jobs)} objects through the chain, {args.workers} workers")
    with get_context("spawn").Pool(args.workers, initializer=_init,
                                   initargs=(cfg.model_dump_json(),)) as pool:
        rows = pd.DataFrame(pool.map(_one, jobs, chunksize=8))

    cols = [f"f_{name}" for name in FEATURES]
    d = rows[rows["set"].isin(("baseline", "ramp")) & rows[cols[0]].notna()]
    np.savez_compressed(
        ASSETS / "drift.npz",
        features=d[cols].to_numpy(dtype=np.float32), level=d.level.to_numpy(dtype=np.float32),
        baseline=(d["set"] == "baseline").to_numpy(),
        undecided=(d.status != "measured").to_numpy(),
        error_um=d.error_um.to_numpy(dtype=np.float32),
        confidence=d.confidence.to_numpy(dtype=np.float32))
    gallery = rows[rows["set"] == "gallery"].drop(columns=["set", "level"]).reset_index(drop=True)
    gallery.to_parquet(ASSETS / "gallery.parquet", index=False)
    print(f"drift.npz: {len(d)} scans; gallery.parquet: {len(gallery)} objects")
    print(gallery.groupby("scenario").status.value_counts(normalize=True).unstack().round(2))


if __name__ == "__main__":
    main()
