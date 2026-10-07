"""Evaluate the locked boundary finder on the locked test set. Owner-only; run once per version.

Every run appends who, when, which model hash and how many items to the access log, and a
copy of the log is written to reports/ so the dashboard can show it. This script and
scripts/make_locked_testset.py are the only code allowed to touch data/test_locked/.
"""

from __future__ import annotations

import argparse
import datetime as dt
import getpass
import json

import numpy as np
from _common import REPO_ROOT, REPORTS_DIR

from coatshield.compliance.registry import load_manifest, verified_path
from coatshield.config import load_config
from coatshield.seg.infer import Segmenter
from coatshield.seg.metrics import boundary_error, thinnest_separable

LOCKED = REPO_ROOT / "data" / "test_locked"
FINAL = REPORTS_DIR / "final"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--owner", default=getpass.getuser(), help="name of the test owner")
    parser.add_argument("--confirm", action="store_true",
                        help="required: confirms you are the test owner and the model is locked")
    args = parser.parse_args()
    if not args.confirm:
        raise SystemExit("Refusing to run without --confirm. Only the test owner runs this.")
    cfg = load_config()
    entry = load_manifest()["models"]["unet"]
    segmenter = Segmenter(verified_path("unet"), cfg)

    root = LOCKED / "oct"
    images = np.load(root / "images.npy", mmap_mode="r")
    labels = np.load(root / "labels.npz")
    params = [json.loads(line) for line in (root / "params.jsonl").read_text().splitlines()]
    outer_err, inner_err, separated = [], [], []
    for i in range(images.shape[0]):
        found = segmenter.surfaces(images[i])
        valid = labels["valid"][i] & found["valid"]
        o = boundary_error(found["outer"], labels["outer_px"][i], valid)
        n = boundary_error(found["inner"], labels["inner_px"][i], valid)
        outer_err.append(o["mae_px"])
        inner_err.append(n["mae_px"])
        gap = np.median((found["inner"] - found["outer"])[valid]) if valid.any() else 0.0
        separated.append(bool(valid.any() and gap >= cfg.seg.separable_min_px
                              and o["mae_px"] <= 2.0 and n["mae_px"] <= 2.0))
    snr = np.array([p["snr_db"] for p in params])
    good = snr >= cfg.seg.eval_min_snr_db
    result = {
        "model": "unet", "version": entry["version"], "model_sha256": entry["sha256"],
        "n_items": int(images.shape[0]),
        "outer_mae_px": float(np.nanmean(outer_err)), "inner_mae_px": float(np.nanmean(inner_err)),
        "outer_mae_px_good_snr": float(np.nanmean(np.array(outer_err)[good])),
        "inner_mae_px_good_snr": float(np.nanmean(np.array(inner_err)[good])),
        "thinnest_separable_um": thinnest_separable(
            np.array([p["thickness_um"] for p in params]), np.array(separated)),
    }
    log_entry = {"action": "evaluate", "who": args.owner,
                 "when_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                 "model_sha256": entry["sha256"], "n_items": result["n_items"]}
    with open(LOCKED / "access_log.jsonl", "a") as log:
        log.write(json.dumps(log_entry) + "\n")
    (REPORTS_DIR / "locked_access_log.jsonl").write_text(
        (LOCKED / "access_log.jsonl").read_text())
    FINAL.mkdir(parents=True, exist_ok=True)
    out = FINAL / f"locked_eval_unet_{entry['sha256'][:12]}.json"
    out.write_text(json.dumps({**result, **log_entry}, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
