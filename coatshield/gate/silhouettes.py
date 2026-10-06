"""Synthetic dark-field camera images of pellets: dark ground, bright rim, faint interior.

Eight classes: single, twin (two fused discs with a neck), touching (tangent, not fused),
partial (cut by the frame), defocused, fouled (haze, low contrast), empty and fines.
"""

from __future__ import annotations

import cv2
import numpy as np

from coatshield.config import Config
from coatshield.seeds import rng

CLASSES = ("single", "twin", "touching", "partial", "defocused", "fouled", "empty", "fines")
SINGLE, TWIN, TOUCHING, PARTIAL, DEFOCUSED, FOULED, EMPTY, FINES = range(8)
_NECK_SMOOTH = 0.18  # smooth-union radius of a fused neck, as a share of the pellet radius


def _disc_distance(yy, xx, cx, cy, radius, ellipticity=0.0, angle=0.0):
    """Approximate signed distance (px) to an ellipse outline: negative inside."""
    dx, dy = xx - cx, yy - cy
    c, s = np.cos(angle), np.sin(angle)
    u, v = c * dx + s * dy, -s * dx + c * dy
    a, b = radius * (1 + ellipticity / 2), radius * (1 - ellipticity / 2)
    scale = np.sqrt((u / a) ** 2 + (v / b) ** 2)
    return (scale - 1.0) * radius


def _smooth_min(a, b, k):
    """Smooth union of two distance fields: fills the neck between fused discs."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


def _shade(distance, gv, gen):
    """Dark-field look from a signed distance field: bright rim, faint textured interior."""
    rim = gv.rim_level * np.exp(-((distance / gv.rim_px) ** 2))
    inside = distance < 0
    texture = cv2.GaussianBlur(gen.standard_normal(distance.shape).astype(np.float32), (0, 0), 2.0)
    interior = np.where(inside, gv.interior_level + gv.texture * 3.0 * texture, 0.0)
    return np.maximum(rim, interior)


def _motion_blur(image, length_px, angle):
    n = int(round(length_px))
    if n < 1:
        return image
    size = 2 * n + 1
    kernel = np.zeros((size, size), np.float32)
    for t in np.linspace(-n, n, 4 * n + 1):
        kernel[int(round(n + t * np.sin(angle))), int(round(n + t * np.cos(angle)))] = 1.0
    return cv2.filter2D(image, -1, kernel / kernel.sum())


def render(label: int, cfg: Config, gen: np.random.Generator) -> tuple[np.ndarray, dict]:
    """One image [frame, frame] uint8 of the given class, and what was drawn."""
    gv = cfg.gate_vision
    n = gv.frame_px
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    centre = n / 2.0
    radius = gen.uniform(*gv.diameter_um) / gv.pixel_um / 2.0
    info: dict = {"label": int(label), "class": CLASSES[label]}

    def one_disc(cx, cy, r):
        return _disc_distance(yy, xx, cx, cy, r, gen.uniform(0, gv.ellipticity),
                              gen.uniform(0, np.pi))

    if label in (TWIN, TOUCHING):
        radius = min(radius, 0.23 * n)  # the pair has to fit in the frame
        r2 = radius * gen.uniform(0.8, 1.0)
        # Twins overlap; touching pellets sit up to a pixel apart.
        overlap = (gen.uniform(*gv.twin_overlap) if label == TWIN
                   else -gen.uniform(0.0, 1.0) / radius)
        gap = (radius + r2) - overlap * radius
        direction = gen.uniform(0, 2 * np.pi)
        ox, oy = 0.5 * gap * np.cos(direction), 0.5 * gap * np.sin(direction)
        jitter = gen.uniform(-4, 4, 2)
        d1 = one_disc(centre - ox + jitter[0], centre - oy + jitter[1], radius)
        d2 = one_disc(centre + ox + jitter[0], centre + oy + jitter[1], r2)
        distance = (_smooth_min(d1, d2, _NECK_SMOOTH * radius) if label == TWIN
                    else np.minimum(d1, d2))
        info.update(radius_px=float(radius), overlap=float(overlap))
        image = _shade(distance, gv, gen)
    elif label == EMPTY:
        image = np.zeros((n, n), np.float32)
    elif label == FINES:
        image = np.zeros((n, n), np.float32)
        for _ in range(int(gen.integers(1, 7))):
            r = gen.uniform(*gv.fines_um) / gv.pixel_um / 2.0
            d = _disc_distance(yy, xx, gen.uniform(r, n - r), gen.uniform(r, n - r), r)
            image = np.maximum(image, _shade(d, gv, gen))
    else:
        if label == PARTIAL:  # centre near or beyond a frame edge, so the frame cuts the pellet
            edge = gen.integers(0, 4)
            along, across = gen.uniform(0.2 * n, 0.8 * n), gen.uniform(-0.5, 0.6) * radius
            cx, cy = [(along, across), (along, n - across), (across, along),
                      (n - across, along)][edge]
        else:
            room = max(n / 2.0 - radius - 4.0, 0.0)
            cx, cy = centre + gen.uniform(-room, room, 2)
        image = _shade(one_disc(cx, cy, radius), gv, gen)
        info.update(radius_px=float(radius))

    image = image.astype(np.float32)
    if label == DEFOCUSED:
        image = cv2.GaussianBlur(image, (0, 0), gen.uniform(*gv.defocus_px))
    if label == FOULED:
        haze = cv2.GaussianBlur(gen.random((n, n)).astype(np.float32), (0, 0), 12.0)
        haze = (haze - haze.min()) / (np.ptp(haze) + 1e-9)
        level = gen.uniform(*gv.haze_level)
        image = (1.0 - level) * 0.6 * image + level * (0.6 + 0.4 * haze)
    image = _motion_blur(image, gen.uniform(0, gv.motion_blur_px), gen.uniform(0, np.pi))
    image = image + gv.noise * gen.standard_normal((n, n)).astype(np.float32)
    return np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8), info


def make_set(cfg: Config, name: str, n: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """n balanced images (classes in rotation) from the stream `name`, with labels and info."""
    images = np.empty((n, cfg.gate_vision.frame_px, cfg.gate_vision.frame_px), np.uint8)
    labels = np.arange(n) % len(CLASSES)
    infos = []
    for i in range(n):
        images[i], info = render(int(labels[i]), cfg, rng(f"gate.{name}.{i}", cfg.seed))
        infos.append(info)
    return images, labels, infos
