"""Ablation model: the compact U-Net with a small transformer block at its bottleneck."""

from __future__ import annotations

import torch
from torch import nn

from coatshield.seg.unet import CompactUNet, _block


class _TransformerBottleneck(nn.Module):
    """Convolution block followed by a 2-layer, 4-head transformer over the feature map."""

    def __init__(self, c_in: int, c_out: int, layers: int = 2, heads: int = 4) -> None:
        super().__init__()
        self.conv = _block(c_in, c_out)
        layer = nn.TransformerEncoderLayer(c_out, heads, dim_feedforward=2 * c_out, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        n, c, h, w = x.shape
        tokens = self.transformer(x.flatten(2).transpose(1, 2))
        return x + tokens.transpose(1, 2).reshape(n, c, h, w)


class HybridUNet(CompactUNet):
    """CompactUNet whose bottleneck adds long-range attention (comparison run only)."""

    def _make_bottleneck(self, c_in: int, c_out: int) -> nn.Module:
        return _TransformerBottleneck(c_in, c_out)
