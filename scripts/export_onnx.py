"""Export a trained boundary finder to ONNX (opset 17), check it against PyTorch, and
record it in models/manifest.json (hash, parameters, metrics, data hashes, git commit)."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess

import numpy as np
import torch
from _common import REPO_ROOT, file_hash

from coatshield.config import load_config
from coatshield.seg.infer import make_session
from coatshield.seg.train import CHECKPOINTS, load_checkpoint

MODELS = REPO_ROOT / "models"
MANIFEST = MODELS / "manifest.json"
OPSET = 17


def export(model: torch.nn.Module, path, height: int, width: int) -> float:
    """Write the ONNX file and return the largest difference from PyTorch on a test input."""
    model.eval()
    example = torch.rand(1, 1, height, width, generator=torch.Generator().manual_seed(0))
    torch.onnx.export(model, example, str(path), opset_version=OPSET, input_names=["scan"],
                      output_names=["logits"], dynamo=False,
                      dynamic_axes={"scan": {0: "batch"}, "logits": {0: "batch"}})
    session = make_session(path)
    with torch.no_grad():
        expected = model(example).numpy()
    got = session.run(None, {"scan": example.numpy()})[0]
    return float(np.abs(got - expected).max())


def update_manifest(name: str, entry: dict) -> None:
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"models": {}}
    manifest["models"][name] = entry
    MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")


def evaluation_metrics(checkpoint: str, cfg) -> dict:
    """Headline numbers from the latest scripts/evaluate_seg.py table for this checkpoint."""
    import pandas as pd

    tables = sorted((REPO_ROOT / "reports").glob(f"seg_scans_{checkpoint}_*.csv"),
                    key=lambda p: p.stat().st_mtime)
    if not tables:
        return {"note": "run scripts/evaluate_seg.py to fill the metrics"}
    t = pd.read_csv(tables[-1])
    clear = t[(t.snr_db >= cfg.seg.eval_min_snr_db) & (t.pigment <= cfg.oct_dataset.pigment_low[1])]
    return {
        "source": tables[-1].name,
        "validation_scans": int(len(t)),
        "outer_mae_px": round(float(t.outer_mae_px.mean()), 3),
        "inner_mae_px": round(float(t.inner_mae_px.mean()), 3),
        "outer_mae_px_clear_good_snr": round(float(clear.outer_mae_px.mean()), 3),
        "inner_mae_px_clear_good_snr": round(float(clear.inner_mae_px.mean()), 3),
        "inner_median_px_clear_good_snr": round(float(clear.inner_mae_px.median()), 3),
        "dice_coating": round(float(t.dice_coating.mean()), 4),
    }


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=REPO_ROOT, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", default="unet_pellets", help="name in models/checkpoints/")
    parser.add_argument("--name", default="unet", help="model name in the manifest")
    parser.add_argument("--version", default="0.1.0")
    args = parser.parse_args()
    cfg = load_config()
    model, info = load_checkpoint(CHECKPOINTS / f"{args.checkpoint}.pt")
    path = MODELS / f"{args.name}.onnx"
    diff = export(model, path, cfg.oct.depth_pixels, cfg.oct.out_ascans)
    if diff > 1e-3:
        raise SystemExit(f"ONNX output differs from PyTorch by {diff:.2e}")
    data_hashes = {}
    for split in ("train", "val"):
        meta = REPO_ROOT / "data" / "synthetic" / split / "manifest.json"
        if meta.exists():
            data_hashes[split] = json.loads(meta.read_text())["images_sha256"]
    entry = {
        "file": path.name,
        "version": args.version,
        "sha256": file_hash(path),
        "parameters": info["parameters"],
        "opset": OPSET,
        "max_abs_diff_vs_pytorch": diff,
        "metrics": evaluation_metrics(args.checkpoint, cfg),
        "stage": info.get("stage"),
        "training_data_sha256": data_hashes,
        "git_commit": _git_commit(),
        "date_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "status": "candidate",
    }
    update_manifest(args.name, entry)
    print(json.dumps(entry, indent=1))


if __name__ == "__main__":
    main()
