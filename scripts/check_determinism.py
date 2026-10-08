"""Determinism suite: the same inputs give byte-identical outputs in separate processes.

  python scripts/check_determinism.py --n 1000

Runs the boundary finder (ONNX, CPU) and the classical gate on n fixed inputs, and the twin
plus estimator on a small batch, each in two fresh Python processes, and compares SHA-256
digests of everything they produce.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys

import numpy as np
from _common import REPO_ROOT

from coatshield.config import load_config
from coatshield.seeds import rng

PARTS = ("unet", "gate", "twin")


def digest_unet(n: int) -> str:
    from coatshield.compliance.registry import verified_path
    from coatshield.seg.infer import Segmenter

    cfg = load_config()
    seg = Segmenter(verified_path("unet"), cfg)
    gen = rng("determinism.unet", cfg.seed)
    h = hashlib.sha256()
    for _ in range(n):
        scan = gen.integers(0, 256, (cfg.oct.out_ascans, cfg.oct.depth_pixels), dtype=np.uint8)
        found = seg.surfaces(scan)
        h.update(found["prob"].tobytes())
        h.update(found["outer"].tobytes())
        h.update(found["inner"].tobytes())
    return h.hexdigest()


def digest_gate(n: int) -> str:
    from coatshield.gate.classical import classify
    from coatshield.gate.silhouettes import make_set

    cfg = load_config()
    images, _, _ = make_set(cfg, "determinism", n)
    h = hashlib.sha256(images.tobytes())
    for image in images:
        verdict = classify(image, cfg)
        h.update(f"{verdict.label}:{verdict.confidence!r}".encode())
    return h.hexdigest()


def digest_twin(n: int) -> str:
    from coatshield.estimate.controllers import run_estimators
    from coatshield.twin.batch import run_batch

    cfg = load_config(overrides={"batch.n_pellets": 20000, "batch.duration_h": 10.0})
    result = run_batch(cfg, cache=False)
    est = run_estimators(result, cfg, bootstrap="needed")
    h = hashlib.sha256()
    for frame in (result.truth, result.samples, est):
        h.update(frame.to_numpy(dtype=float, na_value=np.nan).tobytes())
    return h.hexdigest()


def run_twice(part: str, n: int) -> tuple[str, str]:
    """Digest of one part from two separate interpreter processes."""
    out = []
    for _ in range(2):
        done = subprocess.run([sys.executable, __file__, "--part", part, "--n", str(n)],
                              capture_output=True, text=True, check=True, cwd=REPO_ROOT)
        out.append(json.loads(done.stdout.strip().splitlines()[-1])["digest"])
    return out[0], out[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=1000, help="fixed inputs per model")
    parser.add_argument("--part", choices=PARTS, default=None,
                        help="(internal) compute one digest in this process and print it")
    args = parser.parse_args()
    if args.part:
        digest = {"unet": digest_unet, "gate": digest_gate, "twin": digest_twin}[args.part](args.n)
        print(json.dumps({"part": args.part, "digest": digest}))
        return
    failed = False
    for part in PARTS:
        first, second = run_twice(part, args.n)
        same = first == second
        failed |= not same
        what = f"{args.n} inputs" if part != "twin" else "20,000-pellet batch + estimators"
        print(f"{part:5s} {what:34s} {'identical' if same else 'DIFFERENT'}  {first[:16]}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
