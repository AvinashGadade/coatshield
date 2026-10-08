"""Build the Live Console's replay data in web/console/data/ (also: `make console-data`).

Scans first (simulated, segmented by the locked boundary finder, with camera thumbnails,
signal traces and Grad-CAM overlays), then one replay per scenario. Nothing is retrained
and no model file is changed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from multiprocessing import get_context

for _var in ("NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _common import REPO_ROOT, REPORTS_DIR  # noqa: E402

from coatshield import console_export as ce  # noqa: E402
from coatshield.bundle import build_bundle, save_bundle  # noqa: E402
from coatshield.compliance.registry import load_manifest, verified_path  # noqa: E402
from coatshield.config import config_from_snapshot, load_config  # noqa: E402

CONSOLE = REPO_ROOT / "web" / "console"
DATA = CONSOLE / "data"
BUNDLES = REPO_ROOT / "app" / "assets" / "bundles"
_STATE: dict = {}


def _init(config_json: str) -> None:
    from coatshield.seg.infer import Segmenter

    cfg = config_from_snapshot(config_json)
    _STATE.update(cfg=cfg, seg=Segmenter(verified_path("unet"), cfg))


def _scan(job: dict) -> dict:
    return ce.make_scan(job, _STATE["cfg"], _STATE["seg"], DATA / "scans")


def gradcam_pass(cfg, scans: pd.DataFrame) -> int:
    """Grad-CAM overlay per scan from the registered model's checkpoint, if it is on disk."""
    import cv2

    stage = load_manifest()["models"]["unet"].get("stage")
    name = "unet_pellets_scratch" if stage == "scratch" else "unet_pellets"
    checkpoint = REPO_ROOT / "models" / "checkpoints" / f"{name}.pt"
    done = 0
    if checkpoint.exists():
        from coatshield.seg.explain import gradcam_coating
        from coatshield.seg.train import load_checkpoint

        model, _ = load_checkpoint(checkpoint)
    for sid in scans.scan_id[scans.has_scan]:
        stem = DATA / "scans" / f"scan_{sid:04d}"
        source = stem.with_name(stem.name + "_input.npy")
        if checkpoint.exists():
            scan = np.load(source)
            rows = cfg.console.scan_rows
            cam = gradcam_coating(model, scan.T.astype(np.float32) / 255.0)[:rows]
            heat = cv2.applyColorMap(np.round(cam * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
            base = cv2.cvtColor(np.ascontiguousarray(scan.T[:rows]), cv2.COLOR_GRAY2BGR)
            overlay = cv2.addWeighted(base, 0.45, heat, 0.55, 0.0)
            overlay = np.repeat(overlay, 3, axis=1)
            cv2.imwrite(str(stem) + "_gradcam.jpg", overlay,
                        [cv2.IMWRITE_JPEG_QUALITY, cfg.console.jpeg_quality])
            done += 1
        source.unlink()
    return done


def thumbnails(cfg) -> None:
    """One camera frame per gate verdict, for events that never reach the scanner."""
    import cv2

    from coatshield.gate import silhouettes
    from coatshield.seeds import rng

    classes = {"single": silhouettes.SINGLE, "agglomerate": silhouettes.TWIN,
               "fines": silhouettes.FINES, "held_back": silhouettes.PARTIAL}
    for name, label in classes.items():
        image, _ = silhouettes.render(label, cfg, rng(f"console.thumb.{name}", cfg.seed))
        cv2.imwrite(str(DATA / "scans" / f"camera_{name}.jpg"),
                    cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, cfg.console.jpeg_quality])


def index_methods(cfg) -> dict:
    """The three refractive-index estimates shown side by side, from the solver report."""
    tables = sorted(REPORTS_DIR.glob("solver_index_*.csv"), key=lambda p: p.stat().st_mtime)
    if not tables:
        return {}
    t = pd.read_csv(tables[-1])
    row = t[np.isclose(t.snr, cfg.oct.snr_db)].iloc[0]
    return {"source": f"reports/{tables[-1].name}", "true_n": cfg.coating.n,
            "reflectance": round(cfg.coating.n + float(row.A_pooled_err), 4),
            "fusion": round(cfg.coating.n + float(row.B_fusion_err), 4),
            "anchor": round(cfg.coating.n + float(row.C_anchor_err), 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--scenarios", nargs="*", default=list(ce.SCENARIOS))
    parser.add_argument("--keep-scans", action="store_true", help="reuse the scan library")
    args = parser.parse_args()
    cfg = load_config()
    scans_dir = DATA / "scans"
    index = scans_dir / "index.json"
    if args.keep_scans and index.exists():
        scans = pd.DataFrame(json.loads(index.read_text())["scans"])
    else:
        shutil.rmtree(scans_dir, ignore_errors=True)
        scans_dir.mkdir(parents=True)
        jobs = ce.scan_grid(cfg)
        print(f"{len(jobs)} representative scans, {args.workers} workers")
        with get_context("spawn").Pool(args.workers, initializer=_init,
                                       initargs=(cfg.model_dump_json(),)) as pool:
            rows = pool.map(_scan, jobs, chunksize=4)
        signals = {str(r["scan_id"]): r.pop("signal") for r in rows if "signal" in r}
        scans = pd.DataFrame(rows)
        (scans_dir / "signals.json").write_text(json.dumps(signals, separators=(",", ":")))
        (scans_dir / "signals.js").write_text(
            f"window.CONSOLE_SIGNALS={json.dumps(signals, separators=(',', ':'))};")
        n_cam = gradcam_pass(cfg, scans)
        thumbnails(cfg)
        library = {"scans": json.loads(scans.to_json(orient="records")),
                   "index_methods": index_methods(cfg), "gradcam": n_cam > 0,
                   "note": "representative simulated scans"}
        index.write_text(json.dumps(library, separators=(",", ":")))
        (scans_dir / "index.js").write_text(
            f"window.CONSOLE_SCANS={json.dumps(library, separators=(',', ':'))};")
        print(f"  {int(scans.has_scan.sum())} with a scan, {n_cam} Grad-CAM overlays")
    usable = scans[scans.has_scan & scans.status.isin(("measured", "undecided"))]
    for name in args.scenarios:
        data = ce.build_scenario(cfg, name, usable)
        ce.write_scenario(data, DATA / name)
        # The same batch as a dashboard bundle, so the batch record is built from it.
        save_bundle(build_bundle(ce.scenario_config(cfg, name)), BUNDLES)
        stops = {s["rule"]: s["true_below_spec_pct"] for s in data["stops"]}
        print(f"  {name}: {len(data['events']['t_s'])} events, below spec at each stop {stops}")
    (DATA / "scenarios.js").write_text(
        "window.CONSOLE_SCENARIOS=" + json.dumps(
            [{"id": n, "label": ce.SCENARIOS[n]["label"]} for n in args.scenarios]) + ";")
    size = sum(f.stat().st_size for f in DATA.rglob("*") if f.is_file())
    print(f"{DATA}: {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
