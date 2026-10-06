"""Phase 2 validation: the m x k x gamma grid and the fault scenarios.

Writes CSVs, figures and a summary to reports/, each named with the config hash.
`--quick` runs a reduced grid that reproduces the headline in a few minutes.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from functools import partial
from multiprocessing import get_context

for _var in ("NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # one batch per worker process, no nested threads

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _common import REPORTS_DIR  # noqa: E402

from coatshield import style  # noqa: E402
from coatshield.config import load_config  # noqa: E402
from coatshield.estimate.controllers import CONTROLLERS  # noqa: E402
from coatshield.estimate.validate import (  # noqa: E402
    fault_cases,
    grid_cases,
    run_case,
    summarise_faults,
    summarise_grid,
)

CASE_KEYS = ("scenario", "gamma", "m", "k", "seed")


def _case_key(case: dict) -> str:
    return json.dumps([case[k] for k in CASE_KEYS])


def run_cases(cases: list[dict], workers: int, label: str, base_json: str,
              done_path) -> pd.DataFrame:
    """Run the cases in parallel; finished rows are appended to done_path so a run can resume."""
    rows = []
    if done_path.exists():
        rows = [json.loads(line) for line in done_path.read_text().splitlines() if line.strip()]
    finished = {_case_key(r) for r in rows}
    todo = [c for c in cases if _case_key(c) not in finished]
    if rows:
        print(f"  {label}: resuming, {len(rows)} batches already done", flush=True)
    start = time.perf_counter()
    with get_context("spawn").Pool(workers) as pool, open(done_path, "a") as out:
        job = partial(run_case, base_json=base_json)
        for i, row in enumerate(pool.imap_unordered(job, todo, chunksize=1), 1):
            rows.append(row)
            out.write(json.dumps(row) + "\n")
            out.flush()
            if i % 20 == 0 or i == len(todo):
                rate = (time.perf_counter() - start) / i
                print(f"  {label}: {len(rows)}/{len(cases)} batches, about "
                      f"{rate * (len(todo) - i) / 60:.1f} min left", flush=True)
    wanted = {_case_key(c) for c in cases}
    rows = [r for r in rows if _case_key(r) in wanted]
    return pd.DataFrame(rows).sort_values(list(CASE_KEYS)).reset_index(drop=True)


def _footer(fig, tag: str) -> None:
    fig.text(0.995, 0.005, f"config {tag}", ha="right", va="bottom", fontsize=7,
             color=style.TEXT_SECONDARY)


def plot_heatmap(cells: pd.DataFrame, spec: float, tag: str, path) -> None:
    """True % below spec when the raw-d10 rule stops, over m and k (gamma = 0)."""
    sub = cells[cells.gamma == 0.0]
    grid = sub.pivot(index="k", columns="m", values="C2_true_below_spec_pct").sort_index()
    ref = sub.pivot(index="k", columns="m", values="C3_true_below_spec_pct").sort_index()
    fig, ax = plt.subplots(figsize=(1.5 * grid.shape[1] + 2.4, 0.9 * grid.shape[0] + 2.0))
    ax.grid(False)
    image = ax.imshow(grid.values, cmap=style.sequential_cmap(), origin="lower", aspect="auto",
                      vmin=0, vmax=max(30.0, float(np.nanmax(grid.values))))
    ax.set_xticks(range(grid.shape[1]), [f"{m:g}" for m in grid.columns])
    ax.set_yticks(range(grid.shape[0]), [f"{k:g}" for k in grid.index])
    ax.set_xlabel("window size bias m (0 = none)")
    ax.set_ylabel("size-growth exponent k")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            value = grid.values[i, j]
            dark = value > 0.55 * image.get_clim()[1]
            ax.text(j, i, f"{value:.0f}%\n(CoatShield {ref.values[i, j]:.0f}%)", ha="center",
                    va="center", fontsize=8, color="white" if dark else style.TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Pellets truly below spec when the raw-d10 rule stops\n", fontsize=11)
    ax.text(0.0, 1.02, f"The rule believes 10%. Spec d10 >= {spec:g} um; mean over seeds.",
            transform=ax.transAxes, fontsize=8, color=style.TEXT_SECONDARY)
    fig.colorbar(image, ax=ax, label="% of pellets below spec", shrink=0.85)
    _footer(fig, tag)
    fig.savefig(path)
    plt.close(fig)


def plot_fault_bars(faults: pd.DataFrame, tag: str, path) -> None:
    """Two measures, two panels: out-of-spec share and excess coating, per scenario."""
    panels = [("true_below_spec_pct", "Pellets truly below spec at the stop (%)"),
              ("excess_coating_pct", "Coating sprayed beyond what spec needed (%)")]
    scenarios = list(faults.scenario)
    y = np.arange(len(scenarios))
    height = 0.19
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.62 * len(scenarios) + 2.2), sharey=True)
    for ax, (key, title) in zip(axes, panels, strict=True):
        for i, ctrl in enumerate(CONTROLLERS):
            values = faults[f"{ctrl}_{key}"].to_numpy()
            ax.barh(y + (i - 1.5) * height, values, height=height * 0.86,
                    color=style.CONTROLLER_COLORS[ctrl], label=f"{ctrl} {CONTROLLERS[ctrl]}")
            for yy, v in zip(y + (i - 1.5) * height, values, strict=True):
                if np.isfinite(v):
                    ax.text(v, yy, f" {v:.0f}", va="center", fontsize=7,
                            ha="left" if v >= 0 else "right", color=style.TEXT_SECONDARY)
        ax.axvline(0, color=style.TEXT_SECONDARY, linewidth=0.8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [s.replace("_", " ") for s in scenarios])
    axes[0].invert_yaxis()
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.08), ncol=4, fontsize=8)
    fig.suptitle("Four stopping rules on the same batches, by scenario", x=0.01, ha="left",
                 fontweight="bold")
    fig.text(0.01, 0.0, "No bar on the right: the true d10 never reached spec in the simulated "
             "spray time.", fontsize=8, color=style.TEXT_SECONDARY)
    _footer(fig, tag)
    fig.savefig(path)
    plt.close(fig)


def plot_gamma_panel(cells: pd.DataFrame, spec: float, tag: str, path) -> None:
    """Where the correction stops working: hidden selection the camera cannot see."""
    k_default = load_config().wurster.size_growth_exponent_k
    sub = cells[np.isclose(cells.k, k_default)]
    gammas = sorted(sub.gamma.unique())
    shades = [style.SEQUENTIAL[2], style.SEQUENTIAL[5]]
    panels = [("hybrid_d10_bias", "Corrected d10 minus true d10 (um)"),
              ("C3_true_below_spec_pct", "Below spec when CoatShield stops (%)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, (key, title) in zip(axes, panels, strict=True):
        for g, shade in zip(gammas, shades, strict=False):
            part = sub[sub.gamma == g].sort_values("m")
            ax.plot(part.m, part[key], marker="o", color=shade,
                    label=f"hidden selection gamma = {g:g}")
            last = part.iloc[-1]
            ax.annotate(f"{last[key]:.2f}" if "bias" in key else f"{last[key]:.1f}%",
                        (last.m, last[key]), textcoords="offset points", xytext=(6, 0),
                        fontsize=8, va="center", color=style.TEXT_SECONDARY)
        ax.set_xlabel("window size bias m")
        ax.set_title(title, fontsize=10)
        ax.margins(x=0.12)
    axes[0].axhline(0, color=style.TEXT_SECONDARY, linewidth=0.8)
    axes[1].axhline(10, color=style.TEXT_SECONDARY, linewidth=0.8, linestyle="--")
    axes[1].annotate("10% = on spec", (axes[1].get_xlim()[0], 10), fontsize=7,
                     xytext=(3, 3), textcoords="offset points", color=style.TEXT_SECONDARY)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.1, -0.2), ncol=2, fontsize=8)
    fig.suptitle(f"The correction only undoes selection by size (k = {k_default:g}, "
                 f"spec d10 >= {spec:g} um)", x=0.01, y=1.04, ha="left", fontweight="bold")
    _footer(fig, tag)
    fig.savefig(path)
    plt.close(fig)


def write_summary(cells, faults, cfg, tag, path, quick, n_grid, n_fault) -> str:
    def cell(m, k, g):
        hit = cells[np.isclose(cells.m, m) & np.isclose(cells.k, k) & np.isclose(cells.gamma, g)]
        return hit.iloc[0] if len(hit) else None

    m0, k0 = cfg.window.size_bias_m, cfg.wurster.size_growth_exponent_k
    head = cell(m0, k0, 0.0)
    g0 = cells[cells.gamma == 0.0]
    lines = [
        f"# Validation summary ({'quick' if quick else 'full'} run)",
        "",
        f"Config hash `{tag}` · {n_grid} grid batches and {n_fault} fault batches at "
        f"{cfg.validation.n_pellets:,} pellets · spec d10 >= {cfg.spec.d10_min_um:g} um",
        "",
        f"## Headline (window bias m = {m0:g}, size-growth exponent k = {k0:g}, "
        "no hidden selection)",
        "",
        "| Stopping rule | Stop (h) | True % below spec | True d10 (um) | Believed d10 (um) "
        "| Excess coating (%) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    if head is not None:
        for ctrl, label in CONTROLLERS.items():
            cols = ("stop_h", "true_below_spec_pct", "true_d10_um", "believed_d10_um",
                    "excess_coating_pct")
            fmt = (".2f", ".1f", ".2f", ".2f", ".1f")
            values = " | ".join(
                format(head[f"{ctrl}_{c}"], f) for c, f in zip(cols, fmt, strict=True)
            )
            lines.append(f"| {ctrl} {label} | {values} |")
        ratio = head.C2_true_below_spec_pct / head.C3_true_below_spec_pct
        lines += ["", f"The raw-d10 rule ships {ratio:.1f}x the out-of-spec pellets of the "
                  "CoatShield rule on the same batches."]
    lines += [
        "",
        "## Estimator error on d10 (gamma = 0, all grid cells, scored from "
        f"{cfg.validation.eval_from_h:g} h)",
        "",
        "| Estimator | Median absolute error (um), mean over cells | Worst cell | Mean bias (um) |",
        "| --- | --- | --- | --- |",
    ]
    for name in ("raw", "ipw", "model", "hybrid"):
        lines.append(f"| {name} | {g0[f'{name}_d10_mae'].mean():.2f} | "
                     f"{g0[f'{name}_d10_mae'].max():.2f} | {g0[f'{name}_d10_bias'].mean():+.2f} |")
    no_bias = g0[np.isclose(g0.m, 0.0) | np.isclose(g0.k, 0.0)]
    gap = (no_bias.C2_true_d10_um - no_bias.C3_true_d10_um)
    lines += [
        "",
        "## No false claim of bias (m = 0 or k = 0)",
        "",
        f"True d10 at the raw-d10 stop minus at the CoatShield stop: {gap.min():+.2f} to "
        f"{gap.max():+.2f} um across those cells. Where it is positive, the raw rule stops "
        "late, not early: measurement noise widens the raw sample, so its d10 reads low. "
        "The raw rule ships at most "
        f"{no_bias.C2_true_below_spec_pct.max():.1f}% below spec in these cells.",
        "",
        "## Hidden selection (gamma > 0): where the correction degrades",
        "",
        "| gamma | Corrected d10 bias (um), mean over cells | Worst cell | "
        "CoatShield true % below spec, mean | Worst cell |",
        "| --- | --- | --- | --- | --- |",
    ]
    for g, part in cells.groupby("gamma"):
        worst = part.loc[part.hybrid_d10_bias.abs().idxmax()]
        lines.append(f"| {g:g} | {part.hybrid_d10_bias.mean():+.2f} | "
                     f"{worst.hybrid_d10_bias:+.2f} (m={worst.m:g}, k={worst.k:g}) | "
                     f"{part.C3_true_below_spec_pct.mean():.1f} | "
                     f"{part.C3_true_below_spec_pct.max():.1f} |")
    lines += [
        "",
        "The correction assumes the window selects pellets by size only. With gamma > 0 the "
        "window also favours pellets that have made more passes than their size predicts; the "
        "camera cannot see that, so the corrected d10 reads high by the amount in the table and "
        "the CoatShield rule ships more than 10% below spec.",
        "",
        "## Fault scenarios (mean over seeds)",
        "",
        "| Scenario | " + " | ".join(f"{c} below spec %" for c in CONTROLLERS) + " | "
        + " | ".join(f"{c} mean (um)" for c in CONTROLLERS) + " | C3 stopped |",
        "| --- |" + " --- |" * (2 * len(CONTROLLERS) + 1),
    ]
    for _, r in faults.iterrows():
        lines.append(
            f"| {r.scenario} | "
            + " | ".join(f"{r[f'{c}_true_below_spec_pct']:.1f}" for c in CONTROLLERS) + " | "
            + " | ".join(f"{r[f'{c}_true_mean_um']:.2f}" for c in CONTROLLERS)
            + f" | {100 * r.C3_stopped:.0f}% of batches |"
        )
    lines += ["", "A rule that never stops within the simulated spray time is scored at the "
              "end of the batch. CoatShield holds on purpose while the undecided share is above "
              "its limit (window fouling).", ""]
    text = "\n".join(lines)
    path.write_text(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="reduced grid, a few minutes")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()

    cfg = load_config()
    tag = cfg.analysis_hash() + ("-quick" if args.quick else "")
    REPORTS_DIR.mkdir(exist_ok=True)
    style.apply_matplotlib()

    grid_list, fault_list = grid_cases(cfg, args.quick), fault_cases(cfg, args.quick)
    print(f"config {tag}: {len(grid_list)} grid + {len(fault_list)} fault batches, "
          f"{args.workers} workers")
    base_json = cfg.model_dump_json()
    progress = REPORTS_DIR / ".progress"
    progress.mkdir(exist_ok=True)
    grid = run_cases(grid_list, args.workers, "grid", base_json, progress / f"grid_{tag}.jsonl")
    grid.to_csv(REPORTS_DIR / f"validation_grid_{tag}.csv", index=False)
    faults_raw = run_cases(fault_list, args.workers, "faults", base_json,
                           progress / f"faults_{tag}.jsonl")
    faults_raw.to_csv(REPORTS_DIR / f"validation_faults_{tag}.csv", index=False)

    cells, faults = summarise_grid(grid), summarise_faults(faults_raw)
    cells.to_csv(REPORTS_DIR / f"validation_cells_{tag}.csv", index=False)
    faults.to_csv(REPORTS_DIR / f"validation_fault_means_{tag}.csv", index=False)
    spec = cfg.spec.d10_min_um
    plot_heatmap(cells, spec, tag, REPORTS_DIR / f"heatmap_raw_d10_m_k_{tag}.png")
    plot_fault_bars(faults, tag, REPORTS_DIR / f"controllers_by_scenario_{tag}.png")
    plot_gamma_panel(cells, spec, tag, REPORTS_DIR / f"gamma_panel_{tag}.png")
    print(write_summary(cells, faults, cfg, tag, REPORTS_DIR / f"validation_summary_{tag}.md",
                        args.quick, len(grid), len(faults_raw)))


if __name__ == "__main__":
    main()
