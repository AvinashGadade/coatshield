"""Phase 8 calibration: push objects through the full chain and fit the error model.

Objects are spread over thickness, index, SNR, fouling, pigment, size and class mix.
Writes the per-object table and a summary to reports/, and the error model to models/.
With --segmenter label (a dry run that uses the simulation's own surfaces instead of the
trained boundary finder) the model is written as error_model_dryrun.json and marked so.
"""

from __future__ import annotations

import argparse
import json
import os
from multiprocessing import get_context

for _var in ("NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _common import REPO_ROOT, REPORTS_DIR  # noqa: E402

from coatshield.chain.error_model import fit_error_model  # noqa: E402
from coatshield.chain.pipeline import LabelSegmenter, SampledObject, process_object  # noqa: E402
from coatshield.compliance import confidence as conf  # noqa: E402
from coatshield.config import Config, config_from_snapshot, load_config  # noqa: E402
from coatshield.seeds import rng  # noqa: E402
from coatshield.twin.population import FINES, SINGLE, TWIN  # noqa: E402

MODELS = REPO_ROOT / "models"
_STATE: dict = {}


def sample_object(cfg: Config, index: int) -> SampledObject:
    cc, dc, gv = cfg.chain, cfg.oct_dataset, cfg.gate_vision
    gen = rng(f"chain.calibration.{index}", cfg.seed)
    u = gen.random()
    klass = TWIN if u < cc.twin_share else FINES if u < cc.twin_share + cc.fines_share else SINGLE
    high_pigment = gen.random() < dc.pigment_high_share
    lo, hi = dc.thickness_um
    diameter = gen.uniform(*cfg.wurster.fines_size_um) if klass == FINES \
        else gen.uniform(*gv.diameter_um)
    return SampledObject(
        true_class=klass, diameter_um=float(diameter),
        thickness_um=float(np.exp(gen.uniform(np.log(lo), np.log(hi)))),
        n_coat=float(gen.uniform(*dc.n_coat)), n_core=float(gen.uniform(*dc.n_core)),
        speed_m_s=float(gen.uniform(*dc.speed_m_s)), snr_db=float(gen.uniform(*dc.snr_db)),
        fouling=float(gen.random()),
        pigment=float(gen.uniform(*(dc.pigment_high if high_pigment else dc.pigment_low))),
        standoff_um=float(gen.uniform(*cfg.oct.standoff_um)), seed=index)


def _init(config_json: str, segmenter: str) -> None:
    cfg = config_from_snapshot(config_json)
    if segmenter == "label":
        seg = LabelSegmenter(cfg)
    else:
        from coatshield.compliance.registry import verified_path
        from coatshield.seg.infer import Segmenter

        seg = Segmenter(verified_path("unet"), cfg)
    _STATE.update(cfg=cfg, seg=seg)


def _one(index: int) -> dict:
    cfg = _STATE["cfg"]
    obj = sample_object(cfg, index)
    # The pooled index is set to the truth here, so the tables hold the chain's own error.
    return process_object(obj, cfg, _STATE["seg"], obj.n_coat, keep=False).summary()


def conf_product(seg, fit):
    """The confidence score for whole columns (same formula as compliance.confidence.fuse)."""
    return seg * fit


def register_error_model(path, tag: str, n: int) -> None:
    """List the error model in models/manifest.json so its hash is checked like a model's."""
    import datetime as dt

    from _common import file_hash

    manifest_path = MODELS / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"models": {}}
    boundary_finder = manifest["models"].get("unet", {})
    manifest["models"]["error_model"] = {
        "file": path.name, "version": "0.1.0", "sha256": file_hash(path),
        "calibration_objects": n, "config_hash": tag,
        "fitted_with_unet_sha256": boundary_finder.get("sha256", "unknown"),
        "date_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "status": "candidate",
    }
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--segmenter", choices=("onnx", "label"), default="onnx")
    parser.add_argument("--n", type=int, default=None, help="objects (default: config)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--refit", default=None, metavar="CSV",
                        help="reuse the chain results in this calibration table: recompute "
                             "the confidence with the current settings and refit, without "
                             "running the chain again")
    args = parser.parse_args()
    cfg = load_config()
    tag = cfg.hash(include=("seed", "oct", "seg", "solve", "gate_vision", "chain", "oct_dataset"))
    if args.refit:
        rows = pd.read_csv(args.refit)
        n = len(rows)
        cc = cfg.chain
        fit = (np.exp(-((rows.fit_residual_um / cc.fit_residual_scale_um) ** 2))
               * np.exp(-((rows.spread_um / cc.spread_scale_um) ** 2)))
        has = rows.fit_residual_um.notna()
        rows.loc[has, "fit_confidence"] = fit[has]
        rows.loc[has, "confidence"] = conf_product(rows.seg_confidence[has], fit[has])
    else:
        n = args.n or cfg.chain.calibration_objects
        with get_context("spawn").Pool(args.workers, initializer=_init,
                                       initargs=(cfg.model_dump_json(), args.segmenter)) as pool:
            rows = pd.DataFrame(pool.map(_one, range(n), chunksize=16))
    # Tune the undecided threshold first, then fit the error model with that threshold in
    # force, so the tables describe the readings that will actually be accepted.
    scanned = rows[(rows.true_true_class == SINGLE) & (rows.status != "gated")
                   & rows.thickness_um.notna()]
    tuned = conf.tune_threshold(scanned.confidence.to_numpy(),
                                np.abs(scanned.error_um.to_numpy()), cfg)
    has_reading = rows.thickness_um.notna() & (rows.status != "gated")
    rows.loc[has_reading, "status"] = np.where(
        rows.loc[has_reading, "confidence"] >= tuned["confidence_min"], "measured", "undecided")
    suffix = "" if args.segmenter == "onnx" else "_dryrun"
    rows.to_csv(REPORTS_DIR / f"chain_calibration{suffix}_{tag}.csv", index=False)
    model = fit_error_model(rows, cfg, {"segmenter": args.segmenter, "config_hash": tag,
                                        "confidence_min": tuned["confidence_min"]})
    path = MODELS / f"error_model{suffix}.json"
    digest = model.save(path)
    if not suffix:
        register_error_model(path, tag, n)

    measured = rows[(rows.status == "measured") & (rows.true_true_class == SINGLE)]
    clean = measured[(measured.true_fouling < 0.2) & (measured.true_snr_db >= 25)]

    def spread(err) -> float:
        return float(1.4826 * (err - err.median()).abs().median())

    lines = [
        f"# Chain calibration ({'trained boundary finder' if suffix == '' else 'DRY RUN'})", "",
        f"Config `{tag}` · {n} objects · error model `{path.name}` sha256 `{digest}`", "",
        "| Outcome | Share of objects |", "| --- | --- |",
        *[f"| {k} | {100 * v:.1f}% |" for k, v in rows.status.value_counts(normalize=True).items()],
        "",
        f"Thickness error of measured single pellets: median {measured.error_um.median():+.2f} um, "
        f"robust spread {spread(measured.error_um):.2f} um (all conditions); "
        f"{clean.error_um.median():+.2f} um and {spread(clean.error_um):.2f} um with a clean "
        "window and SNR of at least 25 dB.", "",
        f"Gate: {100 * model.twin_leak:.1f}% of twins pass as single; "
        f"{100 * model.single_reject:.1f}% of single pellets are held back.", "",
        f"Undecided curve (logistic): {json.dumps(model.undecided)}", "",
        f"Confidence threshold for at most {100 * cfg.chain.max_error_share:g}% of accepted "
        f"pellets off by more than {cfg.chain.max_error_um:g} um: {tuned['confidence_min']:.3f} "
        f"(leaves {100 * tuned['undecided_share']:.1f}% undecided; config has "
        f"{cfg.chain.confidence_min:g}).", "",
    ]
    if suffix:
        lines += ["This is a dry run with the simulation's own surfaces plus 1 pixel of noise, "
                  "not the trained boundary finder. Its numbers are a best case.", ""]
    text = "\n".join(lines)
    (REPORTS_DIR / f"chain_calibration{suffix}_{tag}.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
