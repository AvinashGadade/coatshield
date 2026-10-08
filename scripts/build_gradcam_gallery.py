"""Grad-CAM gallery for the boundary finder: where the "coating" decision comes from.

Writes reports/gradcam_unet_<model hash>.png with a few validation scans, the network's
probability of "coating" and the Grad-CAM heat map for that class.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from _common import REPO_ROOT, REPORTS_DIR  # noqa: E402

from coatshield import style  # noqa: E402
from coatshield.compliance.registry import load_manifest  # noqa: E402
from coatshield.config import load_config  # noqa: E402
from coatshield.seg.datasets import PelletDataset  # noqa: E402
from coatshield.seg.explain import gradcam_coating  # noqa: E402
from coatshield.seg.train import CHECKPOINTS, load_checkpoint  # noqa: E402

N_SCANS = 4
ROWS_SHOWN = 320


def main() -> None:
    cfg = load_config()
    entry = load_manifest()["models"]["unet"]
    name = "unet_pellets_scratch" if entry.get("stage") == "scratch" else "unet_pellets"
    model, _ = load_checkpoint(CHECKPOINTS / f"{name}.pt")
    dataset = PelletDataset("val", cfg, train=False)
    params = [json.loads(line) for line in
              (REPO_ROOT / "data" / "synthetic" / "val" / "params.jsonl").read_text().splitlines()]
    # Clear coats at good SNR, spread over thickness.
    good = [i for i, p in enumerate(params)
            if p["snr_db"] >= 30 and p["pigment"] <= 0.15 and p["fouling"] < 0.2]
    picks = sorted(good, key=lambda i: params[i]["thickness_um"])
    picks = [picks[int(j)] for j in np.linspace(0, len(picks) - 1, N_SCANS)]

    style.apply_matplotlib()
    fig, axes = plt.subplots(3, N_SCANS, figsize=(3.4 * N_SCANS, 9.5))
    for col, index in enumerate(picks):
        image, _, _ = dataset.raw(index)
        with torch.no_grad():
            prob = torch.softmax(model(torch.from_numpy(image)[None, None]), dim=1)[0, 1].numpy()
        cam = gradcam_coating(model, image)
        for row, (data, cmap, title) in enumerate((
                (image, "gray", f"scan, coat {params[index]['thickness_um']:.1f} um"),
                (prob, style.sequential_cmap(), "probability of coating"),
                (cam, "magma", "Grad-CAM, coating class"))):
            ax = axes[row, col]
            ax.imshow(data[:ROWS_SHOWN], cmap=cmap, aspect="auto")
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=9)
    fig.suptitle("Boundary finder: what the coating decision rests on", x=0.01, ha="left",
                 fontweight="bold")
    fig.text(0.995, 0.005, f"unet {entry['version']} sha256 {entry['sha256'][:12]}", ha="right",
             fontsize=7, color=style.TEXT_SECONDARY)
    out = REPORTS_DIR / f"gradcam_unet_{entry['sha256'][:12]}.png"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
