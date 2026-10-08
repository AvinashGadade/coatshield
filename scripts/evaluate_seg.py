"""Evaluate trained boundary finders on the synthetic validation set (never the locked set).

Reports boundary error per surface (all scans and SNR >= seg.eval_min_snr_db), Dice per
class, the thinnest film at which the two surfaces still separate, and the pretrained
against from-scratch comparison. Writes a CSV and a summary to reports/.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from _common import REPO_ROOT, REPORTS_DIR

from coatshield.config import load_config
from coatshield.seg.datasets import (
    OCT5K_ROOT,
    PELLET_CLASSES,
    Oct5kDataset,
    PelletDataset,
    oct5k_pairs,
    split_by_volume,
)
from coatshield.seg.dp import find_surfaces
from coatshield.seg.metrics import (
    boundary_error,
    dice_per_class,
    surfaces_separate,
    thinnest_separable,
)
from coatshield.seg.train import CHECKPOINTS, load_checkpoint

RUNS = {"unet_pellets": "OCT5k-pretrained, fine-tuned", "unet_pellets_scratch": "From scratch"}


@torch.no_grad()
def evaluate(model, dataset: PelletDataset, params: list[dict], cfg, limit: int | None):
    sc = cfg.seg
    model.eval()
    rows = []
    for i in range(min(len(dataset), limit or len(dataset))):
        image, mask, weight = dataset.raw(i)
        prob = torch.softmax(model(torch.from_numpy(image)[None, None]), dim=1)[0].numpy()
        found = find_surfaces(prob, sc.dp_max_jump_px, sc.dp_min_gap_px, sc.valid_min_prob)
        valid = dataset.valid[i] & found["valid"]
        outer = boundary_error(found["outer"], dataset.outer[i], valid, params[i]["depth_px_um"])
        inner = boundary_error(found["inner"], dataset.inner[i], valid, params[i]["depth_px_um"])
        dice = dice_per_class(prob.argmax(axis=0), mask, PELLET_CLASSES, weight > 0)
        rows.append({
            "scan": i, "thickness_um": params[i]["thickness_um"], "snr_db": params[i]["snr_db"],
            "pigment": params[i]["pigment"], "fouling": params[i]["fouling"],
            "outer_mae_px": outer["mae_px"], "inner_mae_px": inner["mae_px"],
            "outer_mae_um": outer["mae_um"], "inner_mae_um": inner["mae_um"],
            "dice_above": dice[0], "dice_coating": dice[1], "dice_core": dice[2],
            "coverage": float(valid.sum() / max(dataset.valid[i].sum(), 1)),
            "separated": surfaces_separate(found["outer"], found["inner"], dataset.outer[i],
                                           dataset.inner[i], valid, sc.separable_min_px,
                                           sc.separable_tolerance),
        })
    return pd.DataFrame(rows)


OCT5K_BOUNDARIES = ("ILM", "OPL-Henle", "IS/OS", "inner RPE", "outer RPE")


@torch.no_grad()
def oct5k_boundary_errors(cfg) -> dict | None:
    """Per-boundary error (pixels) of the pretrained network on held-out OCT5k patients.

    Layers are ordered top to bottom, so boundary k of a column sits at the number of
    pixels whose class is below k; that holds for the human masks and the prediction.
    """
    path = CHECKPOINTS / "unet_oct5k.pt"
    pairs = split_by_volume(oct5k_pairs(), cfg)["test"] if OCT5K_ROOT.exists() else []
    if not path.exists() or not pairs:
        return None
    model, _ = load_checkpoint(path)
    model.eval()
    errors = []
    for image, mask, _ in Oct5kDataset(pairs, cfg, train=False):
        pred = model(image[None]).argmax(dim=1)[0].numpy()
        truth = mask.numpy()
        errors.append([np.abs((pred < k).sum(axis=0) - (truth < k).sum(axis=0)).mean()
                       for k in range(1, len(OCT5K_BOUNDARIES) + 1)])
    return {"scans": len(pairs), "volumes": len({p[2] for p in pairs}),
            "mae_px": dict(zip(OCT5K_BOUNDARIES, np.mean(errors, axis=0).round(2).tolist(),
                               strict=True))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--runs", nargs="*", default=list(RUNS))
    args = parser.parse_args()
    cfg = load_config()
    tag = cfg.hash(include=("seed", "oct", "oct_dataset", "seg"))
    dataset = PelletDataset("val", cfg, train=False)
    params = [json.loads(line) for line in
              (REPO_ROOT / "data" / "synthetic" / "val" / "params.jsonl").read_text().splitlines()]
    snr_min = cfg.seg.eval_min_snr_db
    clear_max = cfg.oct_dataset.pigment_low[1]
    lines = ["# Boundary finder evaluation", "",
             f"Config `{tag}` · synthetic validation set, {len(dataset)} scans · mean absolute "
             "boundary error in depth pixels over columns that both the labels and the graph "
             "search call valid (median over scans in brackets)", "",
             f"| Model | Parameters | Scans | Outer, all | Inner, all | Outer, clear coat and SNR "
             f">= {snr_min:g} dB | Inner, clear coat and SNR >= {snr_min:g} dB | Inner, pigmented "
             f"and SNR >= {snr_min:g} dB | Valid columns kept | Dice coating | Thinnest separable "
             "film (um) |",
             "| --- |" + " --- |" * 10]

    def cell(series) -> str:
        return f"{series.mean():.2f} ({series.median():.2f})"

    for name in args.runs:
        path = CHECKPOINTS / f"{name}.pt"
        if not path.exists():
            lines.append(f"| {RUNS.get(name, name)} | not run: no checkpoint {path.name} |"
                         + " |" * 9)
            continue
        model, info = load_checkpoint(path)
        table = evaluate(model, dataset, params, cfg, args.limit)
        table.to_csv(REPORTS_DIR / f"seg_scans_{name}_{tag}.csv", index=False)
        good = table[table.snr_db >= snr_min]
        clear, pigmented = good[good.pigment <= clear_max], good[good.pigment > clear_max]
        clear_all = table[table.pigment <= clear_max]
        thin = thinnest_separable(clear_all.thickness_um.to_numpy(),
                                  clear_all.separated.to_numpy())
        lines.append(
            f"| {RUNS.get(name, name)} | {info['parameters']:,} | {len(table)} | "
            f"{cell(table.outer_mae_px)} | {cell(table.inner_mae_px)} | "
            f"{cell(clear.outer_mae_px)} | {cell(clear.inner_mae_px)} | "
            f"{cell(pigmented.inner_mae_px)} | {100 * table.coverage.mean():.0f}% | "
            f"{table.dice_coating.mean():.3f} | {thin:.1f} (clear coats) |")
    lines += ["", f"Clear coat: pigment at most {clear_max:g}. In pigmented coats the "
              "coating-core interface is hidden by scatter, as in the published catalogue "
              "where only 6 of 22 commercial coatings were readable; those scans must end "
              "undecided rather than be measured."]
    real = oct5k_boundary_errors(cfg)
    lines += ["", "## Real retinal OCT (OCT5k), held-out patients", ""]
    if real is None:
        lines.append("Not run: OCT5k images or the pretrained checkpoint are not on this machine.")
    else:
        lines += [f"Pretrained network on {real['scans']} scans from {real['volumes']} patient "
                  "volumes that were never used for training (split by patient). Mean absolute "
                  "boundary error in pixels:", "",
                  "| " + " | ".join(real["mae_px"]) + " |",
                  "|" + " --- |" * len(real["mae_px"]),
                  "| " + " | ".join(f"{v:.2f}" for v in real["mae_px"].values()) + " |"]
    lines += ["", "Ablation (ResNet-18 U-Net, transformer hybrid): not run unless listed above.",
              ""]
    text = "\n".join(lines)
    (REPORTS_DIR / f"seg_evaluation_{tag}.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
