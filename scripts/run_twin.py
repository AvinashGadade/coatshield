"""Run one batch of the digital twin and print the Phase 1 "Done when" numbers."""

from __future__ import annotations

import argparse
import time

import numpy as np
from _common import REPO_ROOT  # noqa: F401  (puts the repo on sys.path)

from coatshield.config import FAULT_SCENARIOS, load_config
from coatshield.twin.batch import run_batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", action="append", default=[], help="preset name (repeatable)")
    parser.add_argument("--scenario", choices=FAULT_SCENARIOS, default="none")
    parser.add_argument("--hours", type=float, default=None, help="batch duration override")
    parser.add_argument("--pellets", type=int, default=None, help="simulated pellets override")
    args = parser.parse_args()

    overrides: dict = {"fault.scenario": args.scenario}
    if args.hours is not None:
        overrides["batch.duration_h"] = args.hours
    if args.pellets is not None:
        overrides["batch.n_pellets"] = args.pellets
    cfg = load_config(presets=args.preset, overrides=overrides)

    start = time.perf_counter()
    res = run_batch(cfg, cache=False)
    elapsed = time.perf_counter() - start
    truth, samples = res.truth, res.samples
    last = truth.iloc[-1]

    print(f"config hash {res.config_hash} | {res.meta['n_pellets']:,} simulated pellets for "
          f"{res.meta['n_real']:.3g} real | {len(truth)} steps in {elapsed:.1f} s")
    balance = abs(last.coating_volume_cm3 / last.deposited_volume_cm3 - 1)
    print(f"mass balance error            {100 * balance:.2e} %  (limit 0.1 %)")
    lo, hi = cfg.wurster.growth_ratio_target
    print(f"large-to-small growth ratio   {last.growth_ratio:.2f}  (published {lo}-{hi})")
    sel = truth[(truth.t_h >= 0.5) & (truth.t_h <= 3.0)]
    if len(sel) > 2:
        slope_w = np.polyfit(np.log(sel.t_h), np.log(sel.cv_within_size), 1)[0]
        slope_t = np.polyfit(np.log(sel.t_h), np.log(sel.cv), 1)[0]
        print(f"CV decay slope, 0.5-3 h       {slope_w:.2f} size-adjusted, {slope_t:.2f} total"
              "  (target -0.5 +/- 0.1)")
    target = cfg.batch.target_mean_um
    if truth.mean_um.max() >= target:
        row = truth.iloc[(truth.mean_um - target).abs().argmin()]
        print(f"weight gain at {target:g} um mean    {row.weight_gain_pct:.1f} %  "
              f"(reached at {row.t_h:.2f} h)")
    win = samples[(samples.t_s > last.t_s - cfg.estimator.window_min * 60) & samples.accepted]
    print(f"d10 at end: true {last.d10_um:.2f} um | raw window sample "
          f"{np.quantile(win.thickness_um, 0.1):.2f} um  (m = {cfg.window.size_bias_m:g})")
    print(f"at end: mean {last.mean_um:.2f} um, CV {100 * last.cv:.1f} %, below spec "
          f"{100 * last.frac_below_spec:.1f} %, agglomerates {last.agglomerate_pct:.2f} %, "
          f"fines {last.fines_per_min:g}/min, growth {last.growth_um_per_h:.2f} um/h")


if __name__ == "__main__":
    main()
