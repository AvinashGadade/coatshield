"""Grad-CAM for the boundary finder's coating class (development side; needs torch)."""

from __future__ import annotations

import numpy as np
import torch
from pytorch_grad_cam import GradCAM

from coatshield.oct.generator import COATING


class _CoatingTarget:
    """Sum of the coating logit over the pixels the network itself calls coating."""

    def __init__(self, mask: torch.Tensor) -> None:
        self.mask = mask

    def __call__(self, output: torch.Tensor) -> torch.Tensor:
        return (output[COATING] * self.mask).sum()


def gradcam_coating(model: torch.nn.Module, image: np.ndarray) -> np.ndarray:
    """Heat map [depth, A-scan] in 0..1 of what drives the "coating" decision.

    image: one stored scan as the network sees it, [depth, A-scan] floats in 0..1.
    """
    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))[None, None]
    with torch.no_grad():
        mask = (model(x)[0].argmax(dim=0) == COATING).float()
    with GradCAM(model=model, target_layers=[model.decoders[-1]]) as cam:
        return cam(input_tensor=x, targets=[_CoatingTarget(mask)])[0]
