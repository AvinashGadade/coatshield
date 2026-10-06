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
from coatshield.seg.datasets import PELLET_CLASSES, PelletDataset
from coatshield.seg.dp import find_surfaces
from coatshield.seg.metrics import boundary_error, dice_per_class, thinnest_separable
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
        gap = float(np.median((found["inner"] - found["outer"])[valid])) if valid.any() else 0.0
        rows.append({
            "scan": i, "thickness_um": params[i]["thickness_um"], "snr_db": params[i]["snr_db"],
            "pigment": params[i]["pigment"], "fouling": params[i]["fouling"],
            "outer_mae_px": outer["mae_px"], "inner_mae_px": inner["mae_px"],
            "outer_mae_um": outer["mae_um"], "inner_mae_um": inner["mae_um"],
            "dice_above": dice[0], "dice_coating": dice[1], "dice_core": dice[2],
            "separated": bool(valid.any() and gap >= sc.separable_min_px
                              and inner["mae_px"] <= 2.0 and outer["mae_px"] <= 2.0),
        })
    return pd.DataFrame(rows)


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
    lines = ["# Boundary finder evaluation", "",
             f"Config `{tag}` · synthetic validation set, {len(dataset)} scans", "",
             "| Model | Parameters | Outer MAE (px) | Inner MAE (px) | Outer MAE, SNR >= "
             f"{cfg.seg.eval_min_snr_db:g} dB | Inner MAE, SNR >= {cfg.seg.eval_min_snr_db:g} dB "
             "| Dice coating | Thinnest separable film (um) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name in args.runs:
        path = CHECKPOINTS / f"{name}.pt"
        if not path.exists():
            lines.append(f"| {RUNS.get(name, name)} | not run: no checkpoint {path.name} "
                         "| | | | | | |")
            continue
        model, info = load_checkpoint(path)
        table = evaluate(model, dataset, params, cfg, args.limit)
        table.to_csv(REPORTS_DIR / f"seg_scans_{name}_{tag}.csv", index=False)
        good = table[table.snr_db >= cfg.seg.eval_min_snr_db]
        thin = thinnest_separable(table.thickness_um.to_numpy(), table.separated.to_numpy())
        lines.append(
            f"| {RUNS.get(name, name)} | {info['parameters']:,} | {table.outer_mae_px.mean():.2f} "
            f"| {table.inner_mae_px.mean():.2f} | {good.outer_mae_px.mean():.2f} | "
            f"{good.inner_mae_px.mean():.2f} | {table.dice_coating.mean():.3f} | {thin:.1f} |")
    lines += ["", "Ablation (ResNet-18 U-Net, transformer hybrid): not run unless listed above.", ""]
    text = "\n".join(lines)
    (REPORTS_DIR / f"seg_evaluation_{tag}.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
