"""Show any scan of the synthetic dataset with its labels: view_scan.py val 17 [--save x.png]"""

from __future__ import annotations

import argparse
import json

import matplotlib
import numpy as np
from _common import REPO_ROOT

from coatshield import style


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("train", "val"))
    parser.add_argument("index", type=int)
    parser.add_argument("--save", default=None, help="write a PNG instead of opening a window")
    args = parser.parse_args()
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = REPO_ROOT / "data" / "synthetic" / args.split
    image = np.load(root / "images.npy", mmap_mode="r")[args.index]
    lab = np.load(root / "labels.npz")
    outer, inner, valid = (lab[k][args.index] for k in ("outer_px", "inner_px", "valid"))
    params = json.loads((root / "params.jsonl").read_text().splitlines()[args.index])

    style.apply_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)
    cols = np.arange(image.shape[0])
    for ax, with_labels in zip(axes, (False, True), strict=True):
        ax.imshow(image.T, cmap="gray", aspect="auto", vmin=0, vmax=255)
        ax.grid(False)
        ax.set_xlabel("A-scan")
        if with_labels:
            ax.plot(cols, outer, color=style.BLUE, linewidth=1, alpha=0.5)
            ax.plot(cols, inner, color=style.YELLOW, linewidth=1, alpha=0.5)
            ax.plot(cols[valid], outer[valid], color=style.BLUE, linewidth=2,
                    label="air-coating surface")
            ax.plot(cols[valid], inner[valid], color=style.YELLOW, linewidth=2,
                    label="coating-core surface")
            ax.legend(loc="lower right", facecolor="white", framealpha=0.9, frameon=True)
            ax.set_title("Labels from the simulation parameters (bold: valid signal)")
        else:
            ax.set_title(f"{args.split} scan {args.index}")
    axes[0].set_ylabel(f"depth (pixels of {params['depth_px_um']:.2f} um optical path)")
    text = (f"thickness {params['thickness_um']:.1f} um · n_coat {params['n_coat']:.3f} · n_core "
            f"{params['n_core']:.3f} · R {params['radius_um']:.0f} um · v "
            f"{params['speed_m_s']:.2f} m/s · SNR {params['snr_db']:.0f} dB · fouling "
            f"{params['fouling']:.2f} · pigment {params['pigment']:.2f}")
    fig.text(0.01, 0.01, text, fontsize=9, color=style.TEXT_SECONDARY)
    if args.save:
        fig.savefig(args.save)
        print(f"saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
