"""Compare speckle statistics of synthetic pellet scans with the Zenodo in-vivo skin scans.

Writes a CSV and a figure to reports/, named with the hash of the OCT configuration.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _common import RAW_DIR, REPO_ROOT, REPORTS_DIR  # noqa: E402

from coatshield import style  # noqa: E402
from coatshield.config import load_config  # noqa: E402
from coatshield.oct import physics, realism  # noqa: E402
from coatshield.oct.dataset import masks_from_rows  # noqa: E402
from coatshield.oct.generator import CORE  # noqa: E402

STATS = ("skewness", "excess_kurtosis", "grain_axial_px", "grain_lateral_px", "grain_ratio")
SHAPE_STATS = ("skewness", "excess_kurtosis")  # comparable across instruments
ZENODO_PX_UM = 6.0  # pixel size stated with the Zenodo record
CORE_DEPTH_PX = 40  # rows below the coating-core surface used as the scattering region
MIN_LEVEL_DB = 8.0  # the region's mean must sit this far above the noise floor


def synthetic_rows(cfg, split: str, limit: int):
    root = REPO_ROOT / "data" / "synthetic" / split
    images = np.load(root / "images.npy", mmap_mode="r")
    lab = np.load(root / "labels.npz")
    lo, hi = cfg.oct.db_range
    rows, pooled, shown = [], [], None
    for i in range(min(limit, images.shape[0])):
        mask = masks_from_rows(lab["outer_px"][i], lab["inner_px"][i], cfg.oct.depth_pixels)
        depth_in_core = np.cumsum(mask == CORE, axis=1)
        roi = (mask == CORE) & (depth_in_core <= CORE_DEPTH_PX) & lab["valid"][i][:, None]
        if roi.sum() < 1000:
            continue
        db = images[i].astype(float) / 255.0 * (hi - lo) + lo
        if db[roi].mean() < MIN_LEVEL_DB:
            continue
        # Arrays are [A-scan, depth]; statistics expect [depth, lateral].
        s = realism.speckle_stats(db.T, roi.T)
        s["contrast_linear"] = realism.linear_contrast(db.T, roi.T)
        pooled.append(s.pop("standardised"))
        rows.append({"source": "synthetic", "scan": i, **s})
        shown = shown if shown is not None else images[i]
    return rows, np.concatenate(pooled), shown


def zenodo_rows():
    rows, pooled, shown = [], [], None
    for i, path in enumerate(sorted((RAW_DIR / "zenodo_skin" / "DATASET_NPY").rglob("*.npy"))):
        image = np.load(path)
        s = realism.speckle_stats(image, realism.bright_band_roi(image))
        pooled.append(s.pop("standardised"))
        rows.append({"source": "zenodo_skin", "scan": i, **s})
        shown = shown if shown is not None else image
    return rows, pooled, shown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=300)
    args = parser.parse_args()
    cfg = load_config()
    tag = cfg.hash(include=("seed", "oct", "oct_dataset"))

    syn, syn_pool, syn_image = synthetic_rows(cfg, args.split, args.limit)
    zen, zen_values, zen_image = zenodo_rows()
    zen_pool = np.concatenate(zen_values)
    table = pd.DataFrame(syn + zen)
    table.to_csv(REPORTS_DIR / f"oct_realism_{tag}.csv", index=False)

    ks = realism.ks_distance(syn_pool, zen_pool)
    summary = table.groupby("source")[list(STATS) + ["log_std"]].agg(["median", "min", "max"])
    print(summary.round(2).to_string())
    print(f"\nKS distance of standardised log intensity, synthetic vs Zenodo: {ks:.3f}")
    syn_table = table[table.source == "synthetic"]
    print(f"synthetic speckle contrast (linear intensity): "
          f"median {syn_table.contrast_linear.median():.2f}")
    zen_table = table[table.source == "zenodo_skin"]
    for stat in SHAPE_STATS:
        lo, hi = zen_table[stat].agg(["min", "max"])
        med = syn_table[stat].median()
        print(f"  {stat:16s} synthetic median {med:6.2f}  Zenodo range {lo:6.2f} to {hi:6.2f}  "
              f"{'inside' if lo <= med <= hi else 'OUTSIDE'}")
    ks_each = [realism.ks_distance(v, zen_pool) for v in zen_values]
    print(f"  KS distance      synthetic {ks:6.3f}  single Zenodo scans against their pool "
          f"{min(ks_each):.3f} to {max(ks_each):.3f}  "
          f"{'inside' if ks <= max(ks_each) else 'OUTSIDE'}")
    # Grain size depends on pixel size and resolution, so it is stated, not range-checked.
    spec = physics.spectrometer(cfg.oct)
    resolution = physics.axial_resolution_um(cfg.oct.center_nm, cfg.oct.fwhm_nm)
    grain_um = syn_table.grain_axial_px.median() * spec.depth_px_um
    print(f"  axial speckle grain: synthetic {grain_um:.2f} um optical = "
          f"{grain_um / resolution:.2f} x the source-limited resolution; Zenodo "
          f"{zen_table.grain_axial_px.median() * ZENODO_PX_UM:.0f} um (resolution not published)")

    style.apply_matplotlib()
    fig = plt.figure(figsize=(13, 7.5))
    grid = fig.add_gridspec(2, 4)
    for ax, image, title in ((fig.add_subplot(grid[0, 0]), zen_image, "Real: Zenodo skin scan"),
                             (fig.add_subplot(grid[0, 1]), syn_image.T, "Synthetic pellet scan")):
        ax.imshow(image, cmap="gray", aspect="auto")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10)
    ax = fig.add_subplot(grid[0, 2:])
    bins = np.linspace(-4, 4, 81)
    ax.hist(zen_pool, bins=bins, density=True, histtype="step", color=style.NEUTRAL, linewidth=2,
            label="Zenodo skin (real)")
    ax.hist(syn_pool, bins=bins, density=True, histtype="step", color=style.BLUE, linewidth=2,
            label="Synthetic pellets")
    ax.set_title(f"Log intensity in scattering regions, standardised (KS distance {ks:.3f})",
                 fontsize=10)
    ax.set_xlabel("standard deviations from the local mean")
    ax.set_yticks([])
    ax.legend(loc="upper left")
    for j, stat in enumerate(("skewness", "excess_kurtosis", "grain_axial_px", "grain_ratio")):
        ax = fig.add_subplot(grid[1, j])
        data = [table[table.source == s][stat] for s in ("zenodo_skin", "synthetic")]
        parts = ax.boxplot(data, widths=0.5, patch_artist=True, medianprops={"color": style.TEXT},
                           flierprops={"markersize": 3})
        for patch, color in zip(parts["boxes"], (style.NEUTRAL, style.BLUE), strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_xticks([1, 2], ["Zenodo", "Synthetic"])
        ax.set_title(stat.replace("_", " "), fontsize=10)
    fig.suptitle("Synthetic speckle against real OCT: dimensionless statistics only",
                 x=0.01, ha="left", fontweight="bold")
    fig.text(0.995, 0.005, f"config {tag} · {len(syn)} synthetic, {len(zen)} real scans",
             ha="right", fontsize=7, color=style.TEXT_SECONDARY)
    fig.savefig(REPORTS_DIR / f"oct_realism_{tag}.png")
    print(f"wrote reports/oct_realism_{tag}.csv and .png")


if __name__ == "__main__":
    main()
