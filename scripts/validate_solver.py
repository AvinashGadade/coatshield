"""Phase 6 validation: refractive-index methods and thickness on synthetic scans with known truth.

Surfaces are the simulation's own surface rows plus Gaussian localisation noise (the
size of the boundary finder's target error), so the solver is tested on its own.
Writes CSVs, a figure and a summary to reports/, named with the config hash.
"""

from __future__ import annotations

import argparse
import os
from multiprocessing import get_context

for _var in ("NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _common import REPORTS_DIR  # noqa: E402

from coatshield import style  # noqa: E402
from coatshield.config import Config, load_config  # noqa: E402
from coatshield.oct.generator import default_params, labels, simulate  # noqa: E402
from coatshield.oct.preprocess import calibrated_dispersion, process  # noqa: E402
from coatshield.seeds import rng  # noqa: E402
from coatshield.solve.index import (  # noqa: E402
    assumed_index_error,
    index_from_anchor,
    index_from_fusion,
    pool_index,
)
from coatshield.solve.refraction import thickness_along_normal, thickness_vertical  # noqa: E402
from coatshield.solve.thickness import measure_surfaces, pellet_thickness  # noqa: E402

SNRS = (15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0)
THICKNESSES = (4.5, 7.0, 10.0, 14.0, 18.0, 22.0, 26.0, 30.0)
BOUNDARY_NOISE_PX = 1.0  # target error of the boundary finder
CALIBRATION_ERRORS = (-0.10, 0.0, 0.10)
CAMERA_PELLETS = 2500  # pellets in one estimator window, for the fusion method


def one_scan(job: dict) -> dict:
    cfg = Config.model_validate_json(job["config"])
    gen = rng(f"solver.validation.{job['seed']}", cfg.seed)
    lo, hi = cfg.oct_dataset.radius_um
    p = default_params(cfg, thickness_um=job["thickness"], n_coat=job["n_coat"],
                       snr_db=job["snr"], radius_um=float(gen.uniform(lo, hi)),
                       standoff_um=float(gen.uniform(*cfg.oct.standoff_um)), seed=job["seed"])
    raw = simulate(p, cfg, stream="solver.validation.scan")
    out = process(raw.spectra, raw.background, raw.x_um, cfg.oct,
                  dispersion=calibrated_dispersion(cfg))
    lab = labels(p, cfg, out.x_um, out.crop_start_px)
    outer = lab["outer_px"] + BOUNDARY_NOISE_PX * gen.standard_normal(lab["outer_px"].size)
    inner = lab["inner_px"] + BOUNDARY_NOISE_PX * gen.standard_normal(lab["inner_px"].size)
    row = {k: job[k] for k in ("group", "snr", "thickness", "n_coat", "seed")}
    row["radius"] = p.radius_um
    for err in CALIBRATION_ERRORS:
        surf = measure_surfaces(out.db, outer, inner, lab["valid"], out.x_um, out.depth_px_um,
                                out.crop_start_px, out.reflector_db, cfg, calibration_error=err)
        if surf is None:
            row["solved"] = False
            return row
        row[f"n_reflectance_cal{err:+.2f}"] = surf.n_reflectance
    row["solved"] = True
    row["n_ratio"] = surf.n_ratio
    res = pellet_thickness(surf, p.n_coat)
    row.update(thickness_true_n=res.thickness_um, optical=res.optical_um,
               radius_fit=res.radius_um, fit_residual=res.fit_residual_um,
               intra_cv=res.intra_cv, n_ascans=res.n_ascans,
               thickness_assumed=pellet_thickness(surf, cfg.solve.assumed_n).thickness_um)
    # Per-A-scan thickness error with and without the refraction correction, by angle.
    angle = np.degrees(np.abs(surf.theta))
    with_c = thickness_along_normal(surf.optical_um, surf.theta, surf.circle.radius, p.n_coat)
    without = thickness_vertical(surf.optical_um, p.n_coat)
    for name, sel in (("0_10", angle <= 10), ("10_20", angle > 10)):
        row[f"err_refr_{name}"] = float(np.mean(with_c[sel] - p.thickness_um)) if sel.any() \
            else np.nan
        row[f"err_vert_{name}"] = float(np.mean(without[sel] - p.thickness_um)) if sel.any() \
            else np.nan
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--per-snr", type=int, default=12)
    parser.add_argument("--per-thickness", type=int, default=6)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = parser.parse_args()
    cfg = load_config()
    tag = cfg.hash(include=("seed", "oct", "solve", "coating", "core"))
    config = cfg.model_dump_json()
    n_true = cfg.coating.n
    jobs, seed = [], 0
    for snr in SNRS:
        for _ in range(args.per_snr):
            jobs.append(dict(group="snr", snr=snr, thickness=cfg.batch.target_mean_um,
                             n_coat=n_true, seed=seed, config=config))
            seed += 1
    for t in THICKNESSES:
        for _ in range(args.per_thickness):
            jobs.append(dict(group="thickness", snr=cfg.oct.snr_db, thickness=t, n_coat=n_true,
                             seed=seed, config=config))
            seed += 1
    print(f"config {tag}: {len(jobs)} scans, {args.workers} workers")
    with get_context("spawn").Pool(args.workers) as pool:
        table = pd.DataFrame(pool.map(one_scan, jobs, chunksize=4))
    table.to_csv(REPORTS_DIR / f"solver_scans_{tag}.csv", index=False)
    ok = table[table.solved]

    # Index methods against SNR: A per pellet and pooled, B and C at batch level.
    gen = rng("solver.validation.batch", cfg.seed)
    by_snr = ok[ok.group == "snr"]
    # The anchor is made once, on well-resolved scans, and then reused at every SNR.
    anchor = ok[ok.snr >= cfg.oct.snr_db].head(cfg.solve.anchor_pellets)
    microscopy = (anchor.thickness
                  + cfg.solve.microscopy_sigma_um * gen.standard_normal(len(anchor)))
    n_anchor = index_from_anchor(anchor.optical.to_numpy(), microscopy.to_numpy())
    rows = []
    for snr, part in by_snr.groupby("snr"):
        a = part["n_reflectance_cal+0.00"].to_numpy()
        pooled = {err: pool_index(part[f"n_reflectance_cal{err:+.2f}"].to_numpy())
                  for err in CALIBRATION_ERRORS}
        # Fusion: the camera sizes CAMERA_PELLETS pellets before and during coating.
        core_d = 2 * (part.radius.mean() - part.thickness.mean())
        noise = cfg.camera.size_noise_um / np.sqrt(CAMERA_PELLETS)
        coated_d = core_d + 2 * part.thickness.mean() + noise * gen.standard_normal()
        fused = index_from_fusion(part.optical.mean(), coated_d,
                                  core_d + noise * gen.standard_normal())
        rows.append({
            "snr": snr, "pellets": len(part),
            "A_per_pellet_abs_err": float(np.nanmean(np.abs(a - n_true))),
            "A_pooled_err": pooled[0.0].n - n_true,
            "A_pooled_err_cal_minus10": pooled[-0.10].n - n_true,
            "A_pooled_err_cal_plus10": pooled[0.10].n - n_true,
            "A_ratio_pooled_err": pool_index(part.n_ratio.to_numpy()).n - n_true,
            "B_fusion_err": fused - n_true,
            "C_anchor_err": n_anchor - n_true,
        })
    index_table = pd.DataFrame(rows)
    index_table.to_csv(REPORTS_DIR / f"solver_index_{tag}.csv", index=False)

    by_t = ok[ok.group == "thickness"]
    thick = by_t.groupby("thickness").agg(
        error=("thickness_true_n", lambda v: float(np.mean(v - by_t.loc[v.index, "thickness"]))),
        abs_error=("thickness_true_n",
                   lambda v: float(np.mean(np.abs(v - by_t.loc[v.index, "thickness"])))),
        assumed_error=("thickness_assumed",
                       lambda v: float(np.mean(v - by_t.loc[v.index, "thickness"]))),
        refr_10_20=("err_refr_10_20", "mean"), vert_10_20=("err_vert_10_20", "mean"),
        refr_0_10=("err_refr_0_10", "mean"), vert_0_10=("err_vert_0_10", "mean"),
        scans=("seed", "size")).reset_index()
    thick.to_csv(REPORTS_DIR / f"solver_thickness_{tag}.csv", index=False)

    style.apply_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    series = (("A_per_pellet_abs_err", "A reflectance, one pellet (|error|)", style.BLUE, "o"),
              ("A_pooled_err", "A reflectance, pooled", style.BLUE, "s"),
              ("B_fusion_err", "B camera-OCT fusion (batch)", style.ORANGE, "o"),
              ("C_anchor_err", "C at-line anchor (batch)", style.AQUA, "o"))
    for key, label, color, marker in series:
        ax.plot(index_table.snr, index_table[key].abs(), marker=marker, color=color, label=label,
                linestyle="--" if "one pellet" in label else "-")
    ax.axhline(0.01, color=style.TEXT_SECONDARY, linewidth=0.8, linestyle=":")
    ax.annotate("target 0.01", (index_table.snr.min(), 0.01), xytext=(2, 3),
                textcoords="offset points", fontsize=8, color=style.TEXT_SECONDARY)
    ax.set_yscale("log")
    ax.set_xlabel("surface SNR (dB)")
    ax.set_title(f"|error| in refractive index (true n = {n_true:g})", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    ax = axes[1]
    ax.plot(index_table.snr, index_table.A_pooled_err_cal_plus10, marker="o", color=style.ORANGE,
            label="reflector calibration +10%")
    ax.plot(index_table.snr, index_table.A_pooled_err, marker="o", color=style.BLUE,
            label="calibration exact")
    ax.plot(index_table.snr, index_table.A_pooled_err_cal_minus10, marker="o", color=style.AQUA,
            label="reflector calibration -10%")
    ax.axhline(0, color=style.TEXT_SECONDARY, linewidth=0.8)
    ax.set_xlabel("surface SNR (dB)")
    ax.set_title("Method A: pooled index error under a calibration error", fontsize=10)
    ax.legend(fontsize=7)
    ax = axes[2]
    ax.plot(thick.thickness, thick.error, marker="o", color=style.BLUE,
            label="solver, true index, refraction corrected")
    ax.plot(thick.thickness, thick.vert_10_20, marker="o", color=style.ORANGE,
            label="no refraction correction, A-scans beyond 10 deg")
    ax.plot(thick.thickness, thick.refr_10_20, marker="o", color=style.AQUA,
            label="refraction corrected, A-scans beyond 10 deg")
    ax.axhline(0, color=style.TEXT_SECONDARY, linewidth=0.8)
    ax.set_xlabel("true thickness (um)")
    ax.set_title("Thickness error (um), 4.5 to 30 um", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    fig.text(0.995, 0.005, f"config {tag} · surfaces = truth + {BOUNDARY_NOISE_PX:g} px noise",
             ha="right", fontsize=7, color=style.TEXT_SECONDARY)
    fig.savefig(REPORTS_DIR / f"solver_validation_{tag}.png")

    at35 = index_table[index_table.snr == cfg.oct.snr_db].iloc[0]
    lines = [
        "# Solver validation", "",
        f"Config `{tag}` · {len(ok)} of {len(table)} scans solved · surfaces = truth + "
        f"{BOUNDARY_NOISE_PX:g} px localisation noise · true n = {n_true:g}", "",
        f"## Refractive index at SNR {cfg.oct.snr_db:g} dB", "",
        "| Method | Error in n |", "| --- | --- |",
        f"| A reflectance, pooled over {int(at35.pellets)} pellets | {at35.A_pooled_err:+.4f} |",
        f"| A reflectance, one pellet (mean absolute) | {at35.A_per_pellet_abs_err:.4f} |",
        f"| A with reflector calibration +10% / -10% | {at35.A_pooled_err_cal_plus10:+.4f} / "
        f"{at35.A_pooled_err_cal_minus10:+.4f} |",
        f"| A ratio variant (known core index), pooled | {at35.A_ratio_pooled_err:+.4f} |",
        f"| B camera-OCT fusion ({CAMERA_PELLETS} camera readings) | {at35.B_fusion_err:+.4f} |",
        f"| C at-line anchor ({len(anchor)} pellets, microscopy sd "
        f"{cfg.solve.microscopy_sigma_um:g} um) | {at35.C_anchor_err:+.4f} |", "",
        "## Thickness, true index, 4.5 to 30 um", "",
        f"Mean absolute error per pellet: {thick.abs_error.mean():.2f} um "
        f"(worst thickness: {thick.abs_error.max():.2f} um).  ",
        f"A-scans beyond 10 degrees: {thick.vert_10_20.abs().mean():.2f} um mean error without "
        f"the refraction correction, {thick.refr_10_20.abs().mean():.2f} um with it.  ",
        f"Assuming n = {cfg.solve.assumed_n:g} instead of {n_true:g}: thickness reads "
        f"{100 * assumed_index_error(n_true, cfg.solve.assumed_n):+.1f}% "
        f"(measured {100 * (thick.assumed_error / thick.thickness).mean():+.1f}%).", "",
        "Method A is precise per pellet but only as accurate as the reflector calibration; "
        "methods B and C carry no reflectance calibration.", "",
    ]
    text = "\n".join(lines)
    (REPORTS_DIR / f"solver_summary_{tag}.md").write_text(text)
    print(index_table.round(4).to_string(index=False))
    print(thick.round(3).to_string(index=False))
    print(text)


if __name__ == "__main__":
    main()
