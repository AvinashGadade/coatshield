"""Build the synthetic pellet OCT training and validation sets in data/synthetic/.

Each split is stored as images.npy [N, 128, 512] uint8, labels.npz (surface rows per
A-scan and the valid-signal mask) and params.jsonl (the scalars behind every scan).
The locked test set is not built here: only scripts/make_locked_testset.py does that.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from functools import partial
from multiprocessing import get_context

for _var in ("NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
from _common import REPO_ROOT, file_hash  # noqa: E402

from coatshield.config import Config, load_config  # noqa: E402
from coatshield.oct.dataset import make_scan  # noqa: E402

SYNTHETIC = REPO_ROOT / "data" / "synthetic"


def _one(index: int, config_json: str, split: str) -> dict:
    return make_scan(Config.model_validate_json(config_json), split, index)


def build_split(cfg: Config, split: str, n: int, out_dir, workers: int) -> dict:
    """Generate n scans of a split into out_dir; returns the manifest entry."""
    oc = cfg.oct
    out_dir.mkdir(parents=True, exist_ok=True)
    images = np.lib.format.open_memmap(out_dir / "images.npy", mode="w+", dtype=np.uint8,
                                       shape=(n, oc.out_ascans, oc.depth_pixels))
    outer = np.empty((n, oc.out_ascans), np.float32)
    inner = np.empty((n, oc.out_ascans), np.float32)
    valid = np.empty((n, oc.out_ascans), bool)
    start = time.perf_counter()
    job = partial(_one, config_json=cfg.model_dump_json(), split=split)
    with get_context("spawn").Pool(workers) as pool, open(out_dir / "params.jsonl", "w") as log:
        for i, scan in enumerate(pool.imap(job, range(n), chunksize=8)):
            images[i], outer[i], inner[i], valid[i] = (scan["image"], scan["outer_px"],
                                                       scan["inner_px"], scan["valid"])
            log.write(json.dumps(scan["params"]) + "\n")
            if (i + 1) % 500 == 0 or i + 1 == n:
                rate = (time.perf_counter() - start) / (i + 1)
                print(f"  {split}: {i + 1}/{n} scans, about {rate * (n - i - 1) / 60:.1f} min left",
                      flush=True)
    images.flush()
    np.savez_compressed(out_dir / "labels.npz", outer_px=outer, inner_px=inner, valid=valid)
    manifest = {
        "split": split,
        "n": n,
        "config_hash": cfg.hash(include=("seed", "oct", "oct_dataset")),
        "images_sha256": file_hash(out_dir / "images.npy"),
        "labels_sha256": file_hash(out_dir / "labels.npz"),
        "depth_px_um": float(json.loads((out_dir / "params.jsonl").read_text()
                                        .splitlines()[0])["depth_px_um"]),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--train", type=int, default=None, help="training scans (default: config)")
    parser.add_argument("--val", type=int, default=None, help="validation scans (default: config)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    cfg = load_config()
    for split, n in (("train", args.train or cfg.oct_dataset.n_train),
                     ("val", args.val or cfg.oct_dataset.n_val)):
        manifest = build_split(cfg, split, n, SYNTHETIC / split, args.workers)
        print(json.dumps(manifest))


if __name__ == "__main__":
    main()
