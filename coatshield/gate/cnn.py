"""Small CNN gate: four convolution blocks, global average pooling, one linear layer.

It replaces the classical baseline only if it beats it on the held-out table
(scripts/tune_gate.py). Confidence is calibrated by temperature scaling on validation data.
"""

from __future__ import annotations

import torch
from torch import nn

from coatshield.gate.silhouettes import CLASSES


class GateCNN(nn.Module):
    def __init__(self, channels: tuple[int, ...] = (32, 64, 128, 256),
                 n_classes: int = len(CLASSES)) -> None:
        super().__init__()
        layers, c_in = [], 1
        for c in channels:
            layers += [nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(c_in, n_classes)
        self.log_temperature = nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calibrated logits (divided by the fitted temperature)."""
        pooled = self.features(x).mean(dim=(2, 3))
        return self.head(pooled) / self.log_temperature.exp()


def fit_temperature(model: GateCNN, logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Temperature scaling: one scalar that minimises validation cross-entropy."""
    raw = logits * model.log_temperature.exp()  # undo any earlier scaling
    log_t = torch.zeros((), requires_grad=True)
    optim = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        optim.zero_grad()
        loss = nn.functional.cross_entropy(raw / log_t.exp(), labels)
        loss.backward()
        return loss

    optim.step(closure)
    with torch.no_grad():
        model.log_temperature.copy_(log_t)
    return float(log_t.exp())
