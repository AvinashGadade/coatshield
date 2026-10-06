"""Data loaders: OCT5k retinal scans (pretraining) and synthetic pellet scans (fine-tuning).

Images are returned as [1, depth, lateral] floats in 0..1 with a class mask of the
same size and a per-pixel weight (0 where the label is not trustworthy).
Labels come from human annotation (OCT5k) or simulation parameters, never from an
edge-detection algorithm.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from coatshield.config import REPO_ROOT, Config
from coatshield.oct.dataset import masks_from_rows
from coatshield.seeds import rng

OCT5K_ROOT = REPO_ROOT / "data" / "raw" / "oct5k" / "OCT5k"
SYNTHETIC_ROOT = REPO_ROOT / "data" / "synthetic"
OCT5K_CLASSES = 6  # five annotated boundaries give six layer classes
PELLET_CLASSES = 3  # above surface, coating, core


def oct5k_pairs(root: Path = OCT5K_ROOT,
                grading: str = "Grading_1") -> list[tuple[Path, Path, str]]:
    """(image, mask, patient volume) for every manually graded scan."""
    images = root / "Images" / "Images_Manual"
    masks = root / "Masks" / "Masks_Manual" / grading
    pairs = []
    for mask in sorted(masks.rglob("*.png")):
        rel = mask.relative_to(masks)
        image = images / rel
        if image.exists():
            pairs.append((image, mask, f"{rel.parts[0]}/{rel.parts[1]}"))
    return pairs


def split_by_volume(pairs, cfg: Config) -> dict[str, list]:
    """Train / val / test split by patient volume, never by scan.

    Each volume is assigned by a hash of its name, so the split is stable and no
    patient appears in two splits.
    """
    sc = cfg.seg
    out: dict[str, list] = {"train": [], "val": [], "test": []}
    for pair in pairs:
        digest = hashlib.sha256(f"{cfg.seed}:{pair[2]}".encode()).digest()
        u = int.from_bytes(digest[:8], "big") / 2**64
        name = ("test" if u < sc.oct5k_test_volumes
                else "val" if u < sc.oct5k_test_volumes + sc.oct5k_val_volumes else "train")
        out[name].append(pair)
    return out


def augment(image: np.ndarray, mask: np.ndarray, weight: np.ndarray, crop: tuple[int, int],
            cfg: Config, gen: np.random.Generator):
    """Random crop, horizontal flip, gain, gamma, speckle noise and a small vertical shift.

    No rotation: layers must stay ordered top to bottom.
    """
    sc = cfg.seg
    h, w = crop
    shift = int(gen.integers(-sc.aug_shift_px, sc.aug_shift_px + 1))
    if shift:
        image, mask, weight = (np.roll(a, shift, axis=0) for a in (image, mask, weight))
        edge = slice(0, shift) if shift > 0 else slice(shift, None)
        weight = weight.copy()
        weight[edge] = 0.0  # rows that wrapped around carry no label
    top = int(gen.integers(0, image.shape[0] - h + 1))
    left = int(gen.integers(0, image.shape[1] - w + 1))
    image, mask, weight = (a[top : top + h, left : left + w] for a in (image, mask, weight))
    if gen.random() < 0.5:
        image, mask, weight = (a[:, ::-1] for a in (image, mask, weight))
    image = np.clip(image * gen.uniform(*sc.aug_gain), 0.0, 1.0) ** gen.uniform(*sc.aug_gamma)
    image = np.clip(image * (1.0 + sc.aug_speckle * gen.standard_normal(image.shape)), 0.0, 1.0)
    return (np.ascontiguousarray(image, dtype=np.float32), np.ascontiguousarray(mask),
            np.ascontiguousarray(weight, dtype=np.float32))


class _Base(Dataset):
    def __init__(self, cfg: Config, crop: tuple[int, int], train: bool, name: str) -> None:
        self.cfg, self.crop, self.train, self.name = cfg, tuple(crop), train, name
        self.epoch = 0

    def _finish(self, index: int, image, mask, weight):
        if self.train:
            gen = rng(f"seg.aug.{self.name}.{self.epoch}.{index}", self.cfg.seed)
            image, mask, weight = augment(image, mask, weight, self.crop, self.cfg, gen)
        else:  # deterministic centre crop
            h, w = self.crop
            top, left = (image.shape[0] - h) // 2, (image.shape[1] - w) // 2
            image, mask, weight = (np.ascontiguousarray(a[top : top + h, left : left + w])
                                   for a in (image, mask, weight))
        return (torch.from_numpy(image.astype(np.float32))[None],
                torch.from_numpy(mask.astype(np.int64)),
                torch.from_numpy(weight.astype(np.float32)))


class Oct5kDataset(_Base):
    """Real retinal OCT with six layer classes."""

    def __init__(self, pairs, cfg: Config, train: bool) -> None:
        super().__init__(cfg, cfg.seg.oct5k_crop, train, "oct5k")
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        import cv2

        image_path, mask_path, _ = self.pairs[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        return self._finish(index, image, mask, np.ones(mask.shape, np.float32))


class PelletDataset(_Base):
    """Synthetic pellet scans. Stored [A-scan, depth]; returned as [depth, A-scan] images."""

    def __init__(self, split: str, cfg: Config, train: bool, root: Path = SYNTHETIC_ROOT,
                 limit: int | None = None) -> None:
        super().__init__(cfg, cfg.seg.synthetic_crop, train, f"pellet.{split}")
        folder = root / split
        self.images = np.load(folder / "images.npy", mmap_mode="r")
        labels = np.load(folder / "labels.npz")
        self.outer, self.inner, self.valid = (labels[k] for k in ("outer_px", "inner_px", "valid"))
        self.n = min(limit, self.images.shape[0]) if limit else self.images.shape[0]

    def __len__(self) -> int:
        return self.n

    def raw(self, index: int):
        """Image [depth, A-scan] in 0..1, mask and per-pixel weight, without augmentation."""
        image = np.asarray(self.images[index], dtype=np.float32).T / 255.0
        mask = masks_from_rows(self.outer[index], self.inner[index], image.shape[0]).T
        # Columns without usable signal still teach "above surface" outside the pellet,
        # but the surfaces inside them are not trustworthy, so they are left unweighted.
        outside = ~np.isfinite(self.outer[index])
        weight = np.repeat((self.valid[index] | outside)[None, :], image.shape[0], axis=0)
        return image, mask, weight.astype(np.float32)

    def __getitem__(self, index: int):
        return self._finish(index, *self.raw(index))
