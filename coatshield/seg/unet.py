"""Compact U-Net: 4 levels, batch norm, ReLU, upsampling followed by convolution."""

from __future__ import annotations

import torch
from torch import nn


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class CompactUNet(nn.Module):
    """Encoder-decoder with skip connections. Input [N, 1, H, W], output class logits.

    H and W must be divisible by 2^(levels). No dropout: inference is deterministic.
    """

    def __init__(self, n_classes: int, widths: tuple[int, ...] = (16, 32, 64, 128, 256),
                 in_channels: int = 1) -> None:
        super().__init__()
        self.widths = tuple(widths)
        self.encoders = nn.ModuleList()
        c = in_channels
        for w in widths[:-1]:
            self.encoders.append(_block(c, w))
            c = w
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = self._make_bottleneck(c, widths[-1])
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        c = widths[-1]
        for w in reversed(widths[:-1]):
            self.ups.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(c, w, 3, padding=1, bias=False), nn.BatchNorm2d(w),
                nn.ReLU(inplace=True)))
            self.decoders.append(_block(2 * w, w))
            c = w
        self.head = nn.Conv2d(c, n_classes, 1)

    def _make_bottleneck(self, c_in: int, c_out: int) -> nn.Module:
        return _block(c_in, c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips), strict=True):
            x = decoder(torch.cat([up(x), skip], dim=1))
        return self.head(x)

    def encoder_parameters(self):
        yield from self.encoders.parameters()
        yield from self.bottleneck.parameters()

    def decoder_parameters(self):
        yield from self.ups.parameters()
        yield from self.decoders.parameters()
        yield from self.head.parameters()

    def replace_head(self, n_classes: int) -> None:
        """Swap the output layer, keeping every other weight (pretrain -> fine-tune)."""
        self.head = nn.Conv2d(self.head.in_channels, n_classes, 1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
