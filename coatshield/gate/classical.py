"""Classical gate: decide from shape whether an object is a single pellet worth measuring.

Median background subtraction, blur, Otsu threshold, morphological clean-up, contours,
then shape measures. Low solidity or a deep convexity defect marks the neck of a fused
twin; a distance-transform split separates objects that only touch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from coatshield.config import Config
from coatshield.gate.silhouettes import (
    DEFOCUSED,
    EMPTY,
    FINES,
    FOULED,
    PARTIAL,
    SINGLE,
    TOUCHING,
    TWIN,
)


@dataclass(frozen=True)
class Features:
    n_objects: int
    area_px: float
    equivalent_diameter_um: float
    circularity: float  # 4 pi area / perimeter^2
    solidity: float  # area / convex hull area
    defect_depth: float  # deepest convexity defect, relative to the equivalent radius
    aspect_ratio: float
    border_contact: bool
    neck_width: float  # narrowest waist between two lobes, relative to the radius (inf if one lobe)
    sharpness: float  # mean gradient magnitude on the outline (0-1 scale)
    contrast: float  # outline brightness minus background, over their sum

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Verdict:
    label: int  # class index as in silhouettes.CLASSES
    confidence: float  # 0-1
    features: Features | None

    @property
    def passes(self) -> bool:
        return self.label == SINGLE


def _fill(binary: np.ndarray) -> np.ndarray:
    """Fill the rings left by a bright rim so each pellet is one solid blob."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def segment(image: np.ndarray, cfg: Config) -> tuple[np.ndarray, float]:
    """Binary mask of objects and the background level that was subtracted."""
    gv = cfg.gate_vision
    background = float(np.median(image))
    work = np.clip(image.astype(np.int16) - int(round(background)), 0, 255).astype(np.uint8)
    if gv.blur_px > 0:
        work = cv2.GaussianBlur(work, (0, 0), gv.blur_px)
    if work.max() < gv.empty_level * 255.0:  # nothing but noise: Otsu would split the noise
        return np.zeros_like(work), background
    _, binary = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return _fill(binary), background


def _neck_width(mask: np.ndarray, radius_px: float) -> float:
    """Waist between two lobes found by a distance-transform split, relative to the radius."""
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    peaks = (dist > 0.6 * dist.max()).astype(np.uint8)
    n_lobes, lobes = cv2.connectedComponents(peaks)
    if n_lobes - 1 < 2:
        return float("inf")
    # Walk the line between the two largest lobes' centres; the smallest distance-to-edge
    # along it is half the waist.
    sizes = [(lobes == i).sum() for i in range(1, n_lobes)]
    first, second = (np.argsort(sizes)[-2:] + 1).tolist()
    p1 = np.array(np.nonzero(lobes == first)).mean(axis=1)
    p2 = np.array(np.nonzero(lobes == second)).mean(axis=1)
    line = np.linspace(p1, p2, 64).round().astype(int)
    waist = 2.0 * float(dist[line[:, 0], line[:, 1]].min())
    return waist / radius_px


def measure(image: np.ndarray, cfg: Config) -> Features | None:
    """Shape features of the largest object in the image; None if nothing is there."""
    gv = cfg.gate_vision
    mask, background = segment(image, cfg)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.contourArea(c) >= gv.min_area_px]
    if not contours:
        return None
    main = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(main))
    perimeter = float(cv2.arcLength(main, True))
    radius_px = np.sqrt(area / np.pi)
    hull_idx = cv2.convexHull(main, returnPoints=False)
    hull_area = float(cv2.contourArea(cv2.convexHull(main)))
    depth = 0.0
    if len(main) > 3 and len(hull_idx) > 3:
        try:
            defects = cv2.convexityDefects(main, hull_idx)
        except cv2.error:
            defects = None
        if defects is not None:
            depth = float(np.asarray(defects).reshape(-1, 4)[:, 3].max()) / 256.0
    (_, _), (w, h), _ = cv2.minAreaRect(main)
    x, y, bw, bh = cv2.boundingRect(main)
    n = image.shape[0]
    border = x <= 1 or y <= 1 or x + bw >= n - 1 or y + bh >= n - 1

    own = np.zeros_like(mask)
    cv2.drawContours(own, [main], -1, 255, thickness=cv2.FILLED)
    img = image.astype(np.float32) / 255.0
    gx, gy = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    outline = cv2.morphologyEx(own, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0
    rim = float(np.percentile(img[outline], 90))
    back = background / 255.0
    return Features(
        n_objects=len(contours),
        area_px=area,
        equivalent_diameter_um=2.0 * radius_px * gv.pixel_um,
        circularity=float(4.0 * np.pi * area / max(perimeter**2, 1e-9)),
        solidity=area / max(hull_area, 1e-9),
        defect_depth=depth / radius_px,
        aspect_ratio=float(max(w, h) / max(min(w, h), 1e-9)),
        border_contact=bool(border),
        neck_width=_neck_width(own, radius_px),
        sharpness=float(np.hypot(gx, gy)[outline].mean() / 4.0),
        contrast=float((rim - back) / max(rim + back, 1e-9)),
    )


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def classify(image: np.ndarray, cfg: Config, solidity_min: float | None = None,
             defect_max: float | None = None) -> Verdict:
    """Rule-based class and a confidence from the distance to the deciding threshold."""
    gv = cfg.gate_vision
    s_min = gv.solidity_min if solidity_min is None else solidity_min
    d_max = gv.defect_max if defect_max is None else defect_max
    f = measure(image, cfg)
    if f is None:
        return Verdict(EMPTY, 1.0, None)
    scale = gv.confidence_scale
    if f.contrast < gv.contrast_min:
        return Verdict(FOULED, _sigmoid((gv.contrast_min - f.contrast) / scale), f)
    if f.sharpness < gv.sharpness_min:
        return Verdict(DEFOCUSED, _sigmoid((gv.sharpness_min - f.sharpness) / (scale / 2)), f)
    if f.border_contact:
        return Verdict(PARTIAL, 0.95, f)
    if f.equivalent_diameter_um < gv.fines_max_um:
        return Verdict(FINES, _sigmoid((gv.fines_max_um - f.equivalent_diameter_um)
                                       / (scale * gv.fines_max_um)), f)
    # Margin by which the shape clears (positive) or breaks (negative) the twin rule.
    margin = min(f.solidity - s_min, d_max - f.defect_depth)
    if margin < 0:
        label = TOUCHING if f.neck_width < gv.touching_neck_max else TWIN
        return Verdict(label, _sigmoid(-margin / scale), f)
    return Verdict(SINGLE, _sigmoid(margin / scale), f)


def tune_twin_rule(features: list[Features | None], labels: np.ndarray, cfg: Config) -> dict:
    """Grid search of the twin rule: highest twin recall with false twins within the limit.

    Recall counts twins the rule does not let through as single; false twins are single
    pellets the rule calls twin or touching.
    """
    gv = cfg.gate_vision

    def table(labels_wanted):
        rows = [(f.solidity, f.defect_depth) for f, lab in zip(features, labels, strict=True)
                if lab in labels_wanted and f is not None]
        return np.array(rows) if rows else np.zeros((0, 2))

    singles, twins = table({SINGLE}), table({TWIN})
    best = None
    for s_min in gv.solidity_grid:
        for d_max in gv.defect_grid:
            false_twin = float(((singles[:, 0] < s_min) | (singles[:, 1] > d_max)).mean())
            recall = float(((twins[:, 0] < s_min) | (twins[:, 1] > d_max)).mean())
            if false_twin <= gv.false_twin_max and (best is None or recall > best["twin_recall"]):
                best = {"solidity_min": s_min, "defect_max": d_max, "twin_recall": recall,
                        "false_twin_rate": false_twin}
    return best or {"solidity_min": gv.solidity_min, "defect_max": gv.defect_max,
                    "twin_recall": float("nan"), "false_twin_rate": float("nan")}
