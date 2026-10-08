"""Stored explanations: what the chain saw and decided, saved per decision by record ID.

For each measured object the camera image, the gate's shape measures, the boundary finder's
probability map and the two surfaces are written to one file whose name is the record ID.
The audit trail can then point at the evidence behind any number.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def record_id(summary: dict, model_hash: str) -> str:
    """Stable ID of a decision: the object's truth-free inputs plus the models in force."""
    keys = sorted(k for k in summary if k.startswith("true_"))
    blob = json.dumps({"object": {k: summary[k] for k in keys}, "models": model_hash},
                      sort_keys=True, default=float)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def save_explanation(summary: dict, intermediates: dict, model_hash: str,
                     directory: Path) -> str:
    """Write the evidence for one decision; returns its record ID."""
    rid = record_id(summary, model_hash)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {"camera": intermediates["camera"]}
    if "scan" in intermediates:
        arrays.update(
            scan=intermediates["scan"],
            # Probability of "coating", stored as 8-bit to keep the file small.
            coating_probability=np.round(intermediates["prob"][1] * 255).astype(np.uint8),
            outer_px=np.asarray(intermediates["outer_px"], dtype=np.float32),
            inner_px=np.asarray(intermediates["inner_px"], dtype=np.float32),
            valid=np.asarray(intermediates["valid"], dtype=bool))
    meta = {"record_id": rid, "model_hash": model_hash, "summary": summary,
            "gate_features": intermediates.get("gate_features")}
    np.savez_compressed(directory / f"{rid}.npz", meta=json.dumps(meta, default=float), **arrays)
    return rid


def load_explanation(rid: str, directory: Path) -> dict:
    data = np.load(Path(directory) / f"{rid}.npz")
    out = {k: data[k] for k in data.files if k != "meta"}
    out["meta"] = json.loads(str(data["meta"]))
    return out
