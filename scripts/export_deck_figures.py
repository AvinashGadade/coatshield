"""Export the deck figures as 1920-pixel PNGs into reports/deck/.

Three-curve plot, controller race, m x k heatmap, one pellet through the chain, and the gate
gallery, all from real output of the default scenario and the validation reports.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _common import REPO_ROOT, REPORTS_DIR  # noqa: E402

from coatshield import style  # noqa: E402
from coatshield.bundle import bundle_key, load_bundle  # noqa: E402
from coatshield.chain.pipeline import SampledObject, process_object  # noqa: E402
from coatshield.compliance.registry import verified_path  # noqa: E402
from coatshield.config import load_config  # noqa: E402
from coatshield.estimate.controllers import CONTROLLERS  # noqa: E402
from coatshield.gate.classical import classify  # noqa: E402
from coatshield.gate.silhouettes import CLASSES, render  # noqa: E402
from coatshield.seeds import rng  # noqa: E402
from coatshield.seg.infer import Segmenter  # noqa: E402
from coatshield.twin.population import SINGLE  # noqa: E402

DECK = REPORTS_DIR / "deck"
WIDTH_PX, DPI = 1920, 200
MERGE = 3  # thickness bins merged for the three-curve plot, as on the dashboard


def _save(fig, name: str, tag: str) -> None:
    fig.text(0.997, 0.995, f"config {tag}", ha="right", va="top", fontsize=6,
             color=style.TEXT_SECONDARY)
    with plt.rc_context({"savefig.bbox": "standard"}):  # keep the exact 1920-pixel width
        fig.savefig(DECK / name, dpi=DPI)
    plt.close(fig)
    print(f"  {name}")


def _figure(height_in: float):
    return plt.figure(figsize=(WIDTH_PX / DPI, height_in), layout="constrained")


def three_curves(bundle, cfg, tag) -> None:
    step = bundle.meta["stop_steps"]["C2"]
    j = int(np.argmin(np.abs(bundle.frame_steps - step)))
    edges = bundle.thickness_edges
    n = (edges.size - 1) // MERGE
    edges = edges[: n * MERGE + 1 : MERGE]
    centres, width = 0.5 * (edges[:-1] + edges[1:]), edges[1] - edges[0]

    def curve(dist):
        return dist[j][: n * MERGE].reshape(n, MERGE).sum(axis=1) / width

    truth, raw = curve(bundle.dist_truth), curve(bundle.dist_raw)
    cor = curve(bundle.dist_corrected)
    spec = cfg.spec.d10_min_um
    fig = _figure(4.6)
    ax = fig.add_subplot()
    ax.fill_between(centres, truth, color=style.NEUTRAL, alpha=0.2, linewidth=0)
    below = centres <= spec
    ax.fill_between(centres[below], truth[below], color=style.CRITICAL, alpha=0.45, linewidth=0,
                    label="true batch below spec")
    ax.plot(centres, truth, color=style.NEUTRAL, label="true batch")
    ax.plot(centres, raw, color=style.CURVE_COLORS["raw"], label="raw window sample")
    ax.plot(centres, cor, color=style.CURVE_COLORS["corrected"], label="corrected estimate")
    ax.axvline(spec, color=style.TEXT, linewidth=1, linestyle="--")
    ax.annotate(f"spec: d10 >= {spec:g} um", (spec, ax.get_ylim()[1]), xytext=(-6, -14),
                textcoords="offset points", ha="right", fontsize=9)
    shown = np.flatnonzero(truth + raw > 1e-4)
    ax.set_xlim(centres[shown[0]], centres[shown[-1]])
    ax.set_yticks([])
    ax.set_xlabel("coating thickness (um)")
    row_t, row_e = bundle.truth.iloc[step], bundle.est.iloc[step]
    ax.set_title(f"At {row_t.t_h:.2f} h the raw sample reads d10 = {row_e.raw_d10:.1f} um; the "
                 f"batch is at {row_t.d10_um:.1f} um (corrected {row_e.hybrid_d10:.1f} um)",
                 fontsize=10)
    ax.legend(loc="upper right")
    _save(fig, "three_curves.png", tag)


def controller_race(bundle, cfg, tag) -> None:
    truth, est = bundle.truth, bundle.est
    ctrl = bundle.controllers.set_index("controller")
    t = truth.t_h.to_numpy()
    late = t >= 0.5 * t[-1]
    fig = _figure(4.6)
    left, right = fig.subplots(1, 2, width_ratios=[3, 2])
    left.plot(t[late], truth.d10_um[late], color=style.NEUTRAL, linewidth=3, label="true d10")
    left.plot(t[late], est.raw_d10[late], color=style.CURVE_COLORS["raw"], label="raw d10")
    left.plot(t[late], est.hybrid_d10[late], color=style.CURVE_COLORS["corrected"],
              label="corrected d10")
    left.axhline(cfg.spec.d10_min_um, color=style.TEXT, linewidth=1, linestyle="--")
    for name, step in bundle.meta["stop_steps"].items():
        if step is not None:
            left.plot(t[step], truth.d10_um.iloc[step], marker="D", color=style.TEXT, markersize=7,
                      markeredgecolor=style.SURFACE)
            left.annotate(name, (t[step], truth.d10_um.iloc[step]), xytext=(0, -14),
                          textcoords="offset points", ha="center", fontsize=8)
    left.set_xlabel("batch time (h)")
    left.set_ylabel("d10 coating thickness (um)")
    left.set_title("Four rules stop the same batch (diamonds)")
    left.legend(loc="upper left")
    labels = [f"{n} {CONTROLLERS[n]}" for n in ctrl.index]
    y = np.arange(len(labels))
    right.barh(y, ctrl.true_below_spec_pct, height=0.5,
               color=[style.CONTROLLER_COLORS[n] for n in ctrl.index])
    for yy, v in zip(y, ctrl.true_below_spec_pct, strict=True):
        right.text(v, yy, f" {v:.0f}%", va="center", fontsize=9)
    right.axvline(10, color=style.TEXT, linewidth=1, linestyle="--")
    right.set_yticks(y, labels)
    right.invert_yaxis()
    right.grid(axis="y", visible=False)
    right.set_xlabel("% of pellets truly below spec at the stop")
    right.set_title("What each rule ships")
    _save(fig, "controller_race.png", tag)


def heatmap(cfg) -> None:
    tag = cfg.analysis_hash()
    path = REPORTS_DIR / f"validation_cells_{tag}.csv"
    if not path.exists():
        print("  m_k_heatmap.png skipped: no validation report for this config")
        return
    cells = pd.read_csv(path)
    sub = cells[cells.gamma == 0.0]
    raw = sub.pivot(index="k", columns="m", values="C2_true_below_spec_pct").sort_index()
    ours = sub.pivot(index="k", columns="m", values="C3_true_below_spec_pct").sort_index()
    fig = _figure(4.8)
    ax = fig.add_subplot()
    ax.grid(False)
    image = ax.imshow(raw.values, cmap=style.sequential_cmap(), origin="lower", aspect="auto",
                      vmin=0, vmax=max(30.0, float(raw.values.max())))
    ax.set_xticks(range(raw.shape[1]), [f"{m:g}" for m in raw.columns])
    ax.set_yticks(range(raw.shape[0]), [f"{k:g}" for k in raw.index])
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            v = raw.values[i, j]
            ax.text(j, i, f"{v:.0f}%\n(CoatShield {ours.values[i, j]:.0f}%)", ha="center",
                    va="center", fontsize=8,
                    color="white" if v > 0.55 * image.get_clim()[1] else style.TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("window size bias m (0 = none)")
    ax.set_ylabel("size-growth exponent k")
    ax.set_title("Pellets truly below spec when the raw-d10 rule stops (it believes 10%)")
    fig.colorbar(image, ax=ax, label="% of pellets below spec", shrink=0.85)
    _save(fig, "m_k_heatmap.png", tag)


def pellet_inspector(cfg) -> None:
    obj = SampledObject(true_class=SINGLE,
                        diameter_um=cfg.pellet.core_median_um + 2 * cfg.batch.target_mean_um,
                        thickness_um=cfg.batch.target_mean_um, n_coat=cfg.coating.n,
                        n_core=cfg.core.n, speed_m_s=float(np.mean(cfg.oct_dataset.speed_m_s)),
                        snr_db=cfg.chain.operating_snr_db, pigment=cfg.chain.operating_pigment,
                        seed=1)
    rec = process_object(obj, cfg, Segmenter(verified_path("unet"), cfg), cfg.coating.n)
    it = rec.intermediates
    valid = it["valid"]
    cols = np.arange(valid.size)
    rows = int(np.nanmax(np.where(valid, it["inner_px"], 0)) + 120)
    fig = _figure(4.6)
    axes = fig.subplots(1, 3, width_ratios=[1, 1.5, 1.5])
    axes[0].imshow(it["camera"], cmap="gray")
    axes[0].set_title(f"Camera: gate says {rec.gate_class}", fontsize=9)
    axes[1].imshow(it["scan"].T[:rows], cmap="gray", aspect="auto")
    axes[1].plot(cols[valid], it["outer_px"][valid], color=style.BLUE, label="air-coating")
    axes[1].plot(cols[valid], it["inner_px"][valid], color=style.YELLOW, label="coating-core")
    axes[1].legend(loc="lower center", ncol=2, fontsize=8, facecolor="white", frameon=True)
    axes[1].set_title(f"OCT scan: {rec.thickness_um:.1f} um (true {obj.thickness_um:.1f} um)",
                      fontsize=9)
    axes[2].imshow(it["prob"][1][:rows], cmap=style.sequential_cmap(), aspect="auto", vmin=0,
                   vmax=1)
    axes[2].set_title(f"Boundary finder: P(coating), confidence {rec.confidence:.2f}",
                      fontsize=9)
    for ax in axes:
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
    _save(fig, "pellet_inspector.png", cfg.hash(include=("seed", "oct", "seg", "solve", "chain")))


def gate_gallery(cfg) -> None:
    fig = _figure(5.2)
    axes = fig.subplots(2, 4)
    for ax, label in zip(axes.ravel(), range(len(CLASSES)), strict=True):
        image, _ = render(label, cfg, rng(f"deck.gate.{label}", cfg.seed))
        verdict = classify(image, cfg)
        passes = verdict.passes and verdict.confidence >= cfg.gate_vision.single_confidence_min
        ax.imshow(image, cmap="gray")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{CLASSES[label]}: gate says {CLASSES[verdict.label]}\n"
                     f"{'passes to OCT' if passes else 'held back'}", fontsize=9)
    _save(fig, "gate_gallery.png", cfg.hash(include=("seed", "gate_vision")))


def main() -> None:
    cfg = load_config()
    DECK.mkdir(parents=True, exist_ok=True)
    style.apply_matplotlib()
    full = cfg.with_overrides({"batch.n_pellets": cfg.app.n_pellets})
    bundle = load_bundle(bundle_key(full), REPO_ROOT / "app" / "assets" / "bundles")
    print(f"writing {DECK}")
    if bundle is None:
        print("  no precomputed default bundle: run scripts/precompute_scenarios.py")
    else:
        three_curves(bundle, cfg, bundle.meta["config_hash"])
        controller_race(bundle, cfg, bundle.meta["config_hash"])
    heatmap(cfg)
    pellet_inspector(cfg)
    gate_gallery(cfg)


if __name__ == "__main__":
    main()
