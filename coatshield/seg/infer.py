"""Deterministic inference for the deployed app: ONNX Runtime on CPU, then graph search.

No torch import here: the app must run without it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from coatshield.config import Config
from coatshield.seg.dp import find_surfaces


def make_session(path: Path):
    """ONNX Runtime session on one CPU thread, so the same input gives the same bytes."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), sess_options=options,
                                providers=["CPUExecutionProvider"])


class Segmenter:
    """Boundary finder: stored scan [A-scan, depth] uint8 -> class probabilities -> surfaces."""

    def __init__(self, path: Path, cfg: Config) -> None:
        self.session = make_session(path)
        self.input = self.session.get_inputs()[0].name
        self.cfg = cfg

    def probabilities(self, image: np.ndarray) -> np.ndarray:
        """Class probabilities [3, depth, A-scan] for one stored scan."""
        x = (np.asarray(image, dtype=np.float32).T / 255.0)[None, None]
        logits = self.session.run(None, {self.input: x})[0][0]
        logits = logits - logits.max(axis=0, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=0, keepdims=True)

    def surfaces(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Two ordered surfaces per A-scan, a valid flag, margins and the probabilities."""
        sc = self.cfg.seg
        prob = self.probabilities(image)
        out = find_surfaces(prob, sc.dp_max_jump_px, sc.dp_min_gap_px, sc.valid_min_prob)
        out["prob"] = prob
        return out
