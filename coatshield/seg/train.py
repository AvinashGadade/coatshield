"""Training for the boundary finder: OCT5k pretraining, pellet fine-tuning, from-scratch run.

Run as a module:  python -m coatshield.seg.train --stage pretrain|finetune|scratch
Training may run on a GPU with warn_only determinism; the deterministic guarantee is
on CPU ONNX inference (see coatshield/seg/infer.py).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from coatshield.config import REPO_ROOT, Config, load_config
from coatshield.seeds import set_torch_determinism
from coatshield.seg.datasets import (
    OCT5K_CLASSES,
    PELLET_CLASSES,
    Oct5kDataset,
    PelletDataset,
    oct5k_pairs,
    split_by_volume,
)
from coatshield.seg.dp import find_surfaces
from coatshield.seg.metrics import boundary_error, dice_per_class
from coatshield.seg.unet import CompactUNet, count_parameters

CHECKPOINTS = REPO_ROOT / "models" / "checkpoints"


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor,
                      dice_weight: float) -> torch.Tensor:
    """Cross-entropy plus soft Dice, both restricted to pixels with weight > 0."""
    ce = nn.functional.cross_entropy(logits, target, reduction="none")
    total = weight.sum().clamp_min(1.0)
    loss = (ce * weight).sum() / total
    if dice_weight > 0:
        prob = torch.softmax(logits, dim=1) * weight[:, None]
        onehot = nn.functional.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2)
        onehot = onehot * weight[:, None]
        inter = (prob * onehot).sum(dim=(0, 2, 3))
        size = prob.sum(dim=(0, 2, 3)) + onehot.sum(dim=(0, 2, 3))
        present = onehot.sum(dim=(0, 2, 3)) > 0
        dice = (2.0 * inter + 1.0) / (size + 1.0)
        loss = loss + dice_weight * (1.0 - dice[present].mean())
    return loss


@torch.no_grad()
def evaluate(model: nn.Module, dataset, n_classes: int, device: str, batch_size: int) -> dict:
    """Mean Dice per class over a dataset (centre crops, no augmentation)."""
    model.eval()
    scores = []
    for image, mask, weight in DataLoader(dataset, batch_size=batch_size):
        pred = model(image.to(device)).argmax(dim=1).cpu().numpy()
        for p, m, w in zip(pred, mask.numpy(), weight.numpy(), strict=True):
            scores.append(dice_per_class(p, m, n_classes, w > 0))
    dice = np.nanmean(np.array(scores), axis=0)
    return {"dice_per_class": [float(d) for d in dice], "dice_mean": float(np.nanmean(dice))}


@torch.no_grad()
def evaluate_boundaries(model: nn.Module, dataset: PelletDataset, cfg: Config, device: str,
                        limit: int | None = None) -> dict:
    """Boundary error of U-Net + graph search on whole pellet scans, in pixels."""
    sc = cfg.seg
    model.eval()
    errors = {"outer": [], "inner": []}
    for index in range(min(len(dataset), limit or len(dataset))):
        image, _, _ = dataset.raw(index)
        logits = model(torch.from_numpy(image)[None, None].to(device))
        prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
        found = find_surfaces(prob, sc.dp_max_jump_px, sc.dp_min_gap_px, sc.valid_min_prob)
        valid = dataset.valid[index] & found["valid"]
        for name, truth in (("outer", dataset.outer[index]), ("inner", dataset.inner[index])):
            err = boundary_error(found[name], truth, valid)
            if err["n"]:
                errors[name].append(err["mae_px"])
    return {f"{k}_mae_px": float(np.mean(v)) if v else float("nan") for k, v in errors.items()}


def train_model(model: CompactUNet, train_ds, val_ds, n_classes: int, epochs: int, lr: float,
                cfg: Config, device: str = "cpu", encoder_lr_factor: float = 1.0,
                workers: int = 0, log=print) -> list[dict]:
    """AdamW with cosine decay; returns one history row per epoch."""
    sc = cfg.seg
    model.to(device)
    optim = torch.optim.AdamW(
        [{"params": list(model.encoder_parameters()), "lr": lr * encoder_lr_factor},
         {"params": list(model.decoder_parameters()), "lr": lr}],
        weight_decay=sc.weight_decay)
    loader = DataLoader(train_ds, batch_size=sc.batch_size, shuffle=True, num_workers=workers,
                        drop_last=len(train_ds) > sc.batch_size,
                        generator=torch.Generator().manual_seed(cfg.seed))
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max(1, epochs * len(loader)))
    history = []
    for epoch in range(epochs):
        train_ds.epoch = epoch  # fresh augmentation each epoch, reproducible per seed
        model.train()
        start, losses = time.perf_counter(), []
        for image, mask, weight in loader:
            optim.zero_grad(set_to_none=True)
            loss = segmentation_loss(model(image.to(device)), mask.to(device), weight.to(device),
                                     sc.dice_weight)
            loss.backward()
            optim.step()
            schedule.step()
            losses.append(float(loss))
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses)),
               "seconds": time.perf_counter() - start}
        if val_ds is not None and len(val_ds):
            row.update(evaluate(model, val_ds, n_classes, device, sc.batch_size))
        history.append(row)
        log(json.dumps(row))
    return history


def save_checkpoint(model: CompactUNet, name: str, info: dict, directory: Path = CHECKPOINTS):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.pt"
    torch.save({"state_dict": model.state_dict(), "widths": model.widths,
                "n_classes": model.head.out_channels, "info": info}, path)
    (directory / f"{name}.json").write_text(json.dumps(info, indent=1))
    return path


def load_checkpoint(path: Path, model_cls=CompactUNet) -> tuple[CompactUNet, dict]:
    data = torch.load(path, map_location="cpu", weights_only=True)
    model = model_cls(data["n_classes"], tuple(data["widths"]))
    model.load_state_dict(data["state_dict"])
    return model, data["info"]


def run_stage(stage: str, cfg: Config, device: str, epochs: int | None = None,
              limit: int | None = None, workers: int = 0, model_cls=CompactUNet,
              name: str | None = None) -> dict:
    """pretrain: OCT5k from scratch. finetune: pellets from the pretrained weights.
    scratch: pellets from random weights (the honest comparison)."""
    sc = cfg.seg
    set_torch_determinism(cfg.seed, warn_only=True)
    if stage == "pretrain":
        splits = split_by_volume(oct5k_pairs(), cfg)
        pairs = {k: v[:limit] if limit else v for k, v in splits.items()}
        model = model_cls(OCT5K_CLASSES, sc.widths)
        history = train_model(model, Oct5kDataset(pairs["train"], cfg, True),
                              Oct5kDataset(pairs["val"], cfg, False), OCT5K_CLASSES,
                              epochs or sc.pretrain_epochs, sc.lr, cfg, device, workers=workers)
        test = evaluate(model, Oct5kDataset(pairs["test"], cfg, False), OCT5K_CLASSES, device,
                        sc.batch_size)
        info = {"stage": stage, "history": history, "test": test,
                "volumes": {k: len({p[2] for p in v}) for k, v in splits.items()},
                "scans": {k: len(v) for k, v in pairs.items()}}
    else:
        if stage == "finetune":
            model, _ = load_checkpoint(CHECKPOINTS / "unet_oct5k.pt", model_cls)
            model.replace_head(PELLET_CLASSES)
            lr, factor = sc.finetune_lr, sc.encoder_lr_factor
        else:
            model = model_cls(PELLET_CLASSES, sc.widths)
            lr, factor = sc.lr, 1.0
        train_ds = PelletDataset("train", cfg, True, limit=limit)
        val_ds = PelletDataset("val", cfg, False, limit=limit)
        history = train_model(model, train_ds, val_ds, PELLET_CLASSES,
                              epochs or sc.finetune_epochs, lr, cfg, device, factor, workers)
        info = {"stage": stage, "history": history,
                "val_boundaries": evaluate_boundaries(model, val_ds, cfg, device),
                "scans": {"train": len(train_ds), "val": len(val_ds)}}
    info["parameters"] = count_parameters(model)
    default_name = {"pretrain": "unet_oct5k", "finetune": "unet_pellets",
                    "scratch": "unet_pellets_scratch"}[stage]
    info["checkpoint"] = str(save_checkpoint(model.cpu(), name or default_name, info))
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stage", choices=("pretrain", "finetune", "scratch"), required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="use only this many scans")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    info = run_stage(args.stage, load_config(), args.device, args.epochs, args.limit,
                     args.workers)
    print(json.dumps({k: v for k, v in info.items() if k != "history"}, indent=1))


if __name__ == "__main__":
    main()
