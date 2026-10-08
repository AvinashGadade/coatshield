"""Collect every number the deck uses into reports/final/, each with its source file.

Reads the reports already in reports/ (it runs nothing heavy) and writes
final_report.md and final_numbers.json. Source file names carry the config hash of the
run that produced them. Does not touch the locked test set.
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
from _common import REPO_ROOT, REPORTS_DIR

from coatshield.bundle import bundle_key, load_bundle
from coatshield.chain.error_model import load_error_model
from coatshield.compliance.registry import load_manifest
from coatshield.config import load_config
from coatshield.estimate.controllers import CONTROLLERS

FINAL = REPORTS_DIR / "final"
ERROR_MODEL = "models/error_model.json"


def latest(pattern: str):
    hits = sorted(REPORTS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def headline(cells_path, cfg) -> dict | None:
    if cells_path is None or not cells_path.exists():
        return None
    cells = pd.read_csv(cells_path)
    m, k = cfg.window.size_bias_m, cfg.wurster.size_growth_exponent_k
    hit = cells[np.isclose(cells.m, m) & np.isclose(cells.k, k) & np.isclose(cells.gamma, 0.0)]
    if hit.empty:
        return None
    row = hit.iloc[0]
    g0 = cells[cells.gamma == 0.0]
    hidden = cells[cells.gamma > 0.0]
    out = {
        "source": cells_path.name,
        "rules": {c: {"stop_h": float(row[f"{c}_stop_h"]),
                      "true_below_spec_pct": float(row[f"{c}_true_below_spec_pct"]),
                      "true_d10_um": float(row[f"{c}_true_d10_um"])} for c in CONTROLLERS},
        "raw_over_coatshield": float(row.C2_true_below_spec_pct / row.C3_true_below_spec_pct),
        "hybrid_d10_mae_um_mean": float(g0.hybrid_d10_mae.mean()),
        "hybrid_d10_mae_um_worst": float(g0.hybrid_d10_mae.max()),
        "raw_d10_mae_um_mean": float(g0.raw_d10_mae.mean()),
        "raw_below_spec_pct_at_m4_k1p5": float(g0.C2_true_below_spec_pct.max()),
    }
    if len(hidden):
        out["hidden_selection"] = {
            "gamma": float(hidden.gamma.iloc[0]),
            "hybrid_d10_bias_um_mean": float(hidden.hybrid_d10_bias.mean()),
            "coatshield_below_spec_pct_mean": float(hidden.C3_true_below_spec_pct.mean()),
            "coatshield_below_spec_pct_worst": float(hidden.C3_true_below_spec_pct.max()),
        }
    return out


def headline_lines(title: str, h: dict | None) -> list[str]:
    if h is None:
        return [f"## {title}", "", "Not run.", ""]
    lines = [f"## {title}", "", f"Source: `{h['source']}`", "",
             "| Stopping rule | Stops at (h) | Truly below spec | True d10 (um) |",
             "| --- | --- | --- | --- |"]
    for c, label in CONTROLLERS.items():
        r = h["rules"][c]
        lines.append(f"| {c} {label} | {r['stop_h']:.2f} | {r['true_below_spec_pct']:.1f}% | "
                     f"{r['true_d10_um']:.2f} |")
    lines += ["", f"- The raw-d10 rule ships **{h['raw_over_coatshield']:.1f}x** the out-of-spec "
              "pellets of the CoatShield rule.",
              f"- Corrected d10 error: {h['hybrid_d10_mae_um_mean']:.2f} um median absolute, "
              f"{h['hybrid_d10_mae_um_worst']:.2f} um in the worst grid cell "
              f"(raw sample: {h['raw_d10_mae_um_mean']:.2f} um).",
              f"- Strongest assumptions in the grid: the raw rule ships "
              f"{h['raw_below_spec_pct_at_m4_k1p5']:.0f}% below spec."]
    if "hidden_selection" in h:
        hs = h["hidden_selection"]
        lines.append(f"- Hidden selection (gamma = {hs['gamma']:g}): corrected d10 reads "
                     f"{hs['hybrid_d10_bias_um_mean']:+.2f} um; CoatShield ships "
                     f"{hs['coatshield_below_spec_pct_mean']:.1f}% below spec on average, "
                     f"{hs['coatshield_below_spec_pct_worst']:.1f}% in the worst cell.")
    return lines + [""]


def main() -> None:
    cfg = load_config()
    FINAL.mkdir(parents=True, exist_ok=True)
    numbers: dict = {"created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                     "config_hash": cfg.analysis_hash()}
    lines = ["# CoatShield final report", "",
             f"Built {numbers['created_utc']} from the files in `reports/`. Every number names "
             "its source file; file names carry the hash of the configuration that produced "
             "them. Demonstration on simulated and public data, not a validated GMP system.", ""]

    # --- headline, placeholder measurement and chain error model -------------------
    base = headline(REPORTS_DIR / f"validation_cells_{cfg.analysis_hash()}.csv", cfg)
    numbers["headline_placeholder"] = base
    lines += headline_lines("1. Headline with the placeholder measurement (1 um noise)", base)
    model = load_error_model(ERROR_MODEL) if (REPO_ROOT / ERROR_MODEL).exists() else None
    realistic = None
    if model is not None:
        em_cfg = cfg.with_overrides({
            "measurement.error_model_path": ERROR_MODEL,
            "measurement.sigma_um": model.operating_spread_um(cfg.batch.target_mean_um)})
        realistic = headline(REPORTS_DIR / f"validation_cells_{em_cfg.analysis_hash()}.csv", em_cfg)
    numbers["headline_chain_error_model"] = realistic
    lines += headline_lines("2. Headline with the measurement chain's fitted error model",
                            realistic)
    if base and realistic:
        a, b = base["raw_over_coatshield"], realistic["raw_over_coatshield"]
        lines += [f"The ratio is {a:.1f}x with the placeholder and {b:.1f}x with the chain's "
                  f"error model: the headline **{'holds' if b >= 1.5 else 'shrinks'}** under "
                  "the measurement error of the simulated chain.", ""]

    # --- twin -----------------------------------------------------------------------
    full = cfg.with_overrides({"batch.n_pellets": cfg.app.n_pellets})
    bundle = load_bundle(bundle_key(full), REPO_ROOT / "app" / "assets" / "bundles")
    lines += ["## 3. Batch twin (default scenario)", ""]
    if bundle is None:
        lines += ["Default bundle not built.", ""]
    else:
        t = bundle.truth
        sel = t[(t.t_h >= 0.5) & (t.t_h <= 3.0)]
        slope = float(np.polyfit(np.log(sel.t_h), np.log(sel.cv_within_size), 1)[0])
        at16 = t.iloc[(t.mean_um - cfg.batch.target_mean_um).abs().argmin()]
        stop = bundle.meta["stop_steps"]["C3"]
        twin = {"source": f"app/assets/bundles/{bundle.key}", "n_pellets": bundle.meta["n_pellets"],
                "n_real": bundle.meta["n_real"], "growth_ratio": float(t.growth_ratio.iloc[-1]),
                "cv_decay_slope": slope, "weight_gain_pct_at_target": float(at16.weight_gain_pct),
                "measured_share_pct": float(100 * bundle.measured_cum[stop]
                                            / bundle.meta["n_real"])}
        numbers["twin"] = twin
        lines += [f"Source: `{twin['source']}` ({twin['n_pellets']:,} simulated pellets for "
                  f"{twin['n_real'] / 1e6:.0f} million real ones)", "",
                  f"- Large-to-small growth ratio {twin['growth_ratio']:.2f} "
                  "(published 1.08-1.81).",
                  f"- Size-adjusted variability decays with slope {slope:.2f} on log-log axes "
                  "(the published law is -0.5).",
                  f"- Weight gain at {cfg.batch.target_mean_um:g} um mean coat: "
                  f"{twin['weight_gain_pct_at_target']:.1f}%.",
                  f"- Share of the batch the window has measured when CoatShield stops: "
                  f"{twin['measured_share_pct']:.2f}%.", ""]

    # --- measurement modules ----------------------------------------------------------
    manifest = load_manifest()["models"]
    lines += ["## 4. Measurement modules", ""]
    seg = latest("seg_evaluation_*.md")
    if seg and "unet" in manifest:
        m = manifest["unet"]["metrics"]
        numbers["boundary_finder"] = {"source": seg.name, **m,
                                      "parameters": manifest["unet"]["parameters"]}
        lines += [f"**Boundary finder** (`{seg.name}`, model `unet` "
                  f"{manifest['unet']['version']}, {manifest['unet']['parameters']:,} parameters): "
                  f"outer surface {m['outer_mae_px_clear_good_snr']:.2f} px and inner surface "
                  f"{m['inner_mae_px_clear_good_snr']:.2f} px mean error (median "
                  f"{m['inner_median_px_clear_good_snr']:.2f}) on clear coats at SNR >= "
                  f"{cfg.seg.eval_min_snr_db:g} dB; one pixel is "
                  f"{0.3637:.2f} um of optical path. Pigmented coats are not readable.", ""]
    solver = latest("solver_index_*.csv")
    if solver:
        s = pd.read_csv(solver)
        at = s[np.isclose(s.snr, cfg.oct.snr_db)].iloc[0]
        numbers["solver"] = {"source": solver.name, "A_pooled_err": float(at.A_pooled_err),
                             "B_fusion_err": float(at.B_fusion_err),
                             "C_anchor_err": float(at.C_anchor_err),
                             "A_err_cal_plus10": float(at.A_pooled_err_cal_plus10)}
        lines += [f"**Refractive index** (`{solver.name}`, SNR {cfg.oct.snr_db:g} dB, surfaces = "
                  f"truth + 1 px): error in n {at.A_pooled_err:+.3f} (A reflectance), "
                  f"{at.B_fusion_err:+.3f} (B camera-OCT fusion), {at.C_anchor_err:+.3f} "
                  f"(C at-line anchor); a 10% reflector calibration error moves method A by "
                  f"{at.A_pooled_err_cal_plus10 - at.A_pooled_err:+.3f}.", ""]
    gate = latest("gate_comparison_*.csv")
    if gate:
        g = pd.read_csv(gate).iloc[0]
        numbers["gate"] = {"source": gate.name, "twin_recall": float(g.twin_recall),
                           "false_twin_rate": float(g.false_twin_rate),
                           "single_pass_rate": float(g.single_pass_rate)}
        lines += [f"**Gate** (`{gate.name}`, classical method, held-out synthetic images): twin "
                  f"recall {100 * g.twin_recall:.1f}%, false twins {100 * g.false_twin_rate:.1f}%, "
                  f"{100 * g.single_pass_rate:.1f}% of single pellets pass.", ""]
    cal, consistency = latest("chain_calibration_*.md"), latest("chain_consistency_*.md")
    if model is not None and cal:
        op = {"undecided_clean": float(model.p_undecided(0.0, cfg.chain.operating_pigment,
                                                         cfg.chain.operating_snr_db)),
              "undecided_fouled": float(model.p_undecided(1.0, cfg.chain.operating_pigment,
                                                          cfg.chain.operating_snr_db)),
              "spread_um": model.operating_spread_um(cfg.batch.target_mean_um)}
        numbers["chain"] = {"source": cal.name, "error_model_sha256": model.sha256(),
                            "twin_leak": model.twin_leak, "single_reject": model.single_reject,
                            **op}
        file_hash = manifest.get("error_model", {}).get("sha256", model.sha256())
        lines += [f"**Full chain** (`{cal.name}`, error model file sha256 `{file_hash[:12]}`): "
                  f"reading spread {op['spread_um']:.2f} um at the operating point; undecided "
                  f"share {100 * op['undecided_clean']:.1f}% with a clean window and "
                  f"{100 * op['undecided_fouled']:.0f}% fully fouled; {100 * model.twin_leak:.1f}% "
                  f"of twins pass the gate; {100 * model.single_reject:.1f}% of single pellets "
                  "are held back.", ""]
    if consistency:
        verdict = [ln for ln in consistency.read_text().splitlines() if ln.startswith("Target:")]
        lines += [f"**Chain against error model** (`{consistency.name}`): {verdict[0]}", ""]

    # --- models and the locked evaluation -------------------------------------------
    lines += ["## 5. Models in force", "", "| Model | Version | SHA-256 | Status |",
              "| --- | --- | --- | --- |"]
    for name, e in manifest.items():
        lines.append(f"| {name} | {e['version']} | `{e['sha256']}` | {e['status']} |")
    numbers["models"] = {n: {"version": e["version"], "sha256": e["sha256"], "status": e["status"]}
                         for n, e in manifest.items()}
    locked = sorted(FINAL.glob("locked_eval_*.json"))
    lines += ["", "## 6. Locked test set", ""]
    if locked:
        result = json.loads(locked[-1].read_text())
        numbers["locked_evaluation"] = result
        lines += [f"`{locked[-1].name}`: outer {result['outer_mae_px']:.2f} px, inner "
                  f"{result['inner_mae_px']:.2f} px on {result['n_items']} scans, run by "
                  f"{result['who']} at {result['when_utc']}.", ""]
    else:
        numbers["locked_evaluation"] = None
        lines += ["Not run. Only the test owner builds the locked set "
                  "(`python scripts/make_locked_testset.py --confirm`) and evaluates on it once "
                  "(`python scripts/run_locked_eval.py --confirm`).", ""]

    (FINAL / "final_report.md").write_text("\n".join(lines))
    (FINAL / "final_numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
