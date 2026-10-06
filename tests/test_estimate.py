import numpy as np
import pandas as pd
import pytest

from coatshield.config import load_config
from coatshield.estimate import dissolution
from coatshield.estimate.controllers import (
    CONTROLLERS,
    evaluate_controllers,
    run_estimators,
    stop_steps,
)
from coatshield.estimate.diagnosis import RULES, diagnose, diagnosis_matrix, signals
from coatshield.estimate.hybrid import estimate_window
from coatshield.estimate.quantiles import raw_estimate, weighted_quantile
from coatshield.estimate.validate import fault_cases, grid_cases, run_case
from coatshield.estimate.weights import (
    Reference,
    atline_reference,
    classify,
    ipw_weights,
    kish_ess,
    oracle_reference,
)
from coatshield.seeds import rng
from coatshield.twin.batch import run_batch

SMALL = {"batch.n_pellets": 40000}


def cfg_small(**overrides):
    return load_config(overrides={**SMALL, **overrides})


def analysed(**overrides):
    cfg = cfg_small(**overrides)
    result = run_batch(cfg, cache=False)
    est = run_estimators(result, cfg, bootstrap="needed")
    return cfg, result, est


@pytest.fixture(scope="module")
def default():
    return analysed()


@pytest.fixture(scope="module")
def unbiased():
    return analysed(**{"window.size_bias_m": 0.0})


def scored(result, est, column):
    late = (result.truth.t_h >= 2.0).to_numpy()
    return (est[column] - result.truth.d10_um).to_numpy()[late]


# --- quantiles and weights ------------------------------------------------------


def test_weighted_quantile_matches_plain_quantile_for_equal_weights():
    x = rng("test.quantile").normal(10, 2, 5001)
    got = weighted_quantile(x, np.ones_like(x), [0.1, 0.5, 0.9])
    assert np.allclose(got, np.quantile(x, [0.1, 0.5, 0.9]), atol=0.01)


def test_weighted_quantile_follows_the_weights_and_is_order_independent():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    w = np.array([1.0, 1.0, 1.0, 9.0])
    assert weighted_quantile(x, w, 0.5) > weighted_quantile(x, np.ones(4), 0.5)
    perm = np.array([2, 0, 3, 1])
    assert weighted_quantile(x[perm], w[perm], 0.3) == weighted_quantile(x, w, 0.3)


def test_kish_effective_sample_size():
    assert kish_ess(np.ones(100)) == pytest.approx(100)
    assert kish_ess(np.array([100.0] + [1e-9] * 99)) == pytest.approx(1.0, abs=1e-3)


def synthetic_window(m=3.0, n=2500, noise=1.0, k=1.0, seed_name="test.window"):
    """A log-normal bed with thickness ~ size^k, seen through a size-biased window."""
    gen = rng(seed_name)
    size = 700 * np.exp(0.18 * gen.standard_normal(400_000))
    thickness = 15.0 * (size / 700) ** k * (1 + 0.02 * gen.standard_normal(size.size))
    p = size**m
    idx = gen.choice(size.size, n, p=p / p.sum())
    measured = thickness[idx] + noise * gen.standard_normal(n)
    ref = Reference(np.log(size[::40]), np.full(size[::40].size, 1 / size[::40].size), None)
    return size[idx], measured, ref, float(np.quantile(thickness, 0.1)), float(thickness.mean())


def test_ipw_weights_undo_the_size_bias():
    size, _, ref, _, _ = synthetic_window()
    classes = classify(np.log(size), ref, 30)
    assert classes.f_true.sum() == pytest.approx(1.0)
    w = ipw_weights(classes, 100.0)
    true_mean_log = np.sum(ref.weight * ref.log_size)
    assert np.mean(np.log(size)) > true_mean_log + 0.05
    assert np.sum(w * np.log(size)) / w.sum() == pytest.approx(true_mean_log, abs=0.01)
    assert ipw_weights(classes, 99.0).max() <= w.max()


# --- estimators on a window with known truth ------------------------------------


def test_raw_overstates_and_corrected_estimators_recover_d10():
    size, h, ref, true_d10, true_mean = synthetic_window()
    est = estimate_window(size, h, ref, load_config())
    assert est["raw_d10"] > true_d10 + 0.8
    assert est["hybrid_d10"] == pytest.approx(true_d10, abs=0.2)
    assert est["model_d10"] == pytest.approx(true_d10, abs=0.2)
    assert est["ipw_d10"] == pytest.approx(true_d10, abs=0.6)
    assert est["hybrid_mean"] == pytest.approx(true_mean, abs=0.2)
    assert est["fit_beta"] == pytest.approx(1.0, abs=0.1)
    assert 0 < est["ipw_ess"] < est["n"]


def test_no_false_claim_of_bias_when_growth_ignores_size():
    size, h, ref, true_d10, _ = synthetic_window(k=0.0, noise=0.0)
    est = estimate_window(size, h, ref, load_config(overrides={"measurement.sigma_um": 0.0}))
    assert est["raw_d10"] == pytest.approx(true_d10, abs=0.1)
    assert est["hybrid_d10"] == pytest.approx(est["raw_d10"], abs=0.1)


def test_known_measurement_noise_is_removed_from_the_spread():
    size, h, ref, true_d10, _ = synthetic_window(m=0.0, noise=2.0)
    on = estimate_window(size, h, ref, load_config())
    off = estimate_window(size, h, ref, load_config(overrides={
        "estimator.deconvolve_noise": False}))
    assert off["hybrid_d10"] < true_d10 - 0.2
    assert abs(on["hybrid_d10"] - true_d10) < abs(off["hybrid_d10"] - true_d10)


def test_bootstrap_is_seeded_and_brackets_the_point_estimate():
    size, h, ref, _, _ = synthetic_window()
    cfg = load_config()
    a = estimate_window(size, h, ref, cfg, boot_gen=rng("test.boot"))
    b = estimate_window(size, h, ref, cfg, boot_gen=rng("test.boot"))
    assert a == b
    assert a["d10_lo"] < a["hybrid_d10"] < a["d10_hi"]
    assert a["d10_hi"] - a["d10_lo"] < 1.0
    assert a["p_spec"] == 0.0  # true d10 is about 11.9 um, under the 12 um spec
    low = estimate_window(size, h, ref, cfg.with_overrides({"spec.d10_min_um": 6.0}),
                          boot_gen=rng("test.boot"))
    high = estimate_window(size, h, ref, cfg.with_overrides({"spec.d10_min_um": 20.0}),
                           boot_gen=rng("test.boot"))
    assert (low["p_spec"], high["p_spec"]) == (1.0, 0.0)
    skipped = estimate_window(size, h, ref, cfg, boot_gen=rng("test.boot"), boot_from_d10=99.0)
    assert np.isnan(skipped["p_spec"])


@pytest.mark.parametrize("method", ["raw", "ipw", "model", "hybrid"])
def test_bootstrap_supports_every_method(method):
    size, h, ref, _, _ = synthetic_window()
    cfg = load_config(overrides={"estimator.method": method})
    est = estimate_window(size, h, ref, cfg, boot_gen=rng("test.boot"))
    assert est["d10_lo"] <= est[f"{method}_d10"] + 0.3
    assert est["d10_hi"] >= est[f"{method}_d10"] - 0.3


# --- estimators and controllers on the twin ------------------------------------


def test_hybrid_d10_tracks_the_truth(default):
    _, result, est = default
    assert np.nanmedian(np.abs(scored(result, est, "hybrid_d10"))) < 0.3
    assert np.nanmedian(np.abs(scored(result, est, "raw_d10"))) > 0.4
    assert np.nanmean(scored(result, est, "raw_d10")) > 0.4


@pytest.mark.parametrize("reference", ["oracle", "atline", "coa"])
def test_every_reference_source_supports_the_correction(default, reference):
    _, result, _ = default
    cfg = cfg_small(**{"estimator.reference": reference})
    est = run_estimators(result, cfg, bootstrap="none")
    assert np.nanmedian(np.abs(scored(result, est, "hybrid_d10"))) < 0.3


def test_atline_reference_is_a_finite_noisy_sample(default):
    cfg, result, _ = default
    ref = atline_reference(result, 300, cfg)
    exact = oracle_reference(result, 300)
    assert ref.n_effective == cfg.estimator.atline_n == ref.log_size.size
    assert np.sum(ref.weight * ref.log_size) == pytest.approx(
        np.sum(exact.weight * exact.log_size), abs=0.01)
    assert np.array_equal(ref.log_size, atline_reference(result, 300, cfg).log_size)


def test_controllers_stop_and_coatshield_holds_the_spec(default):
    cfg, result, est = default
    out = evaluate_controllers(result, est, cfg)
    assert list(out.index) == list(CONTROLLERS)
    assert out.stopped.all()
    assert out.loc["C3", "true_below_spec_pct"] <= 11.0
    assert out.loc["C3", "true_d10_um"] == pytest.approx(cfg.spec.d10_min_um, abs=0.4)
    # The raw rules stop early and ship more out-of-spec pellets than they believe.
    assert out.loc["C2", "stop_h"] < out.loc["C3", "stop_h"]
    assert out.loc["C2", "true_below_spec_pct"] > 1.5 * out.loc["C3", "true_below_spec_pct"]
    assert out.loc["C2", "believed_d10_um"] >= cfg.spec.d10_min_um
    assert out.loc["C2", "true_d10_um"] < cfg.spec.d10_min_um - 0.5
    assert out.loc["C2", "excess_coating_pct"] < 0 < out.loc["C0", "excess_coating_pct"] + 20


def test_controllers_agree_without_window_bias(unbiased):
    cfg, result, est = unbiased
    out = evaluate_controllers(result, est, cfg)
    assert abs(out.loc["C2", "true_d10_um"] - out.loc["C3", "true_d10_um"]) < 0.4
    assert abs(out.loc["C2", "true_below_spec_pct"] - out.loc["C3", "true_below_spec_pct"]) < 4.0
    # Without window bias the raw d10 never overstates; measurement noise only widens it.
    assert np.nanmean(scored(result, est, "raw_d10")) < 0.05
    assert np.nanmedian(np.abs(scored(result, est, "raw_d10"))) < 0.5


def test_controllers_agree_without_size_dependent_growth():
    # Noise-free readings isolate the size-bias mechanism: with k = 0 there is none.
    cfg, result, est = analysed(**{"wurster.size_growth_exponent_k": 0.0,
                                   "measurement.sigma_um": 0.0})
    out = evaluate_controllers(result, est, cfg)
    assert abs(out.loc["C2", "true_d10_um"] - out.loc["C3", "true_d10_um"]) < 0.4
    assert np.nanmedian(np.abs(scored(result, est, "raw_d10"))) < 0.3


def test_raw_rule_is_only_conservative_without_size_dependent_growth():
    # With noisy readings and k = 0 the raw d10 reads low, so the raw rule stops late, not early.
    cfg, result, est = analysed(**{"wurster.size_growth_exponent_k": 0.0})
    out = evaluate_controllers(result, est, cfg)
    assert out.loc["C2", "true_d10_um"] >= out.loc["C3", "true_d10_um"] - 0.1
    assert out.loc["C2", "true_below_spec_pct"] <= 10.0


def test_estimates_are_deterministic(default):
    cfg, result, est = default
    again = run_estimators(result, cfg, bootstrap="needed")
    pd.testing.assert_frame_equal(est, again)


def test_substrate_shift_fools_gravimetric_but_not_coatshield():
    cfg, result, est = analysed(**{"fault.scenario": "substrate_shift"})
    out = evaluate_controllers(result, est, cfg)
    assert out.loc["C0", "true_mean_um"] > cfg.batch.target_mean_um + 1.0
    assert out.loc["C3", "true_d10_um"] == pytest.approx(cfg.spec.d10_min_um, abs=0.4)
    assert out.loc["C0", "excess_coating_pct"] > out.loc["C3", "excess_coating_pct"] + 5.0


def test_coatshield_holds_while_the_window_is_fouled():
    cfg, result, est = analysed(**{"fault.scenario": "window_fouling"})
    assert stop_steps(result, est, cfg)["C3"] is None
    out = evaluate_controllers(result, est, cfg)
    assert not out.loc["C3", "stopped"]
    assert est.undecided_rate.iloc[-1] > cfg.controller.undecided_limit


# --- diagnosis ------------------------------------------------------------------


def flagged_share(cfg, result, est, fault, after_h=5.0):
    matrix = diagnosis_matrix(signals(result, est, cfg), cfg)
    return float(matrix[fault][matrix.t_h > after_h].mean())


def test_healthy_batch_raises_no_process_fault(default):
    cfg, result, est = default
    for fault in ("over_wetting", "spray_drying", "nozzle_block", "window_fouling",
                  "substrate_shift"):
        assert flagged_share(cfg, result, est, fault, after_h=0.0) == 0.0


@pytest.mark.parametrize("fault", ["nozzle_block", "over_wetting", "spray_drying"])
def test_fault_is_diagnosed_with_its_action(fault):
    cfg, result, est = analysed(**{"fault.scenario": fault, "batch.duration_h": 7.0})
    assert flagged_share(cfg, result, est, fault) > 0.9
    sig = signals(result, est, cfg)
    assert diagnose(sig.iloc[10], cfg) == []  # warm-up: no verdict
    found = diagnose(sig.iloc[-1], cfg)
    assert fault in [r.fault for r in found]
    assert all(r.action for r in found)
    others = {"nozzle_block", "over_wetting", "spray_drying"} - {fault}
    assert not others & {r.fault for r in found}


def test_maldistribution_raises_the_spread_signal(default):
    cfg, result, est = default
    healthy = signals(result, est, cfg)
    cfg_f, result_f, est_f = analysed(**{"fault.scenario": "maldistribution"})
    faulty = signals(result_f, est_f, cfg_f)
    late = healthy.t_h > 5.0
    assert faulty.spread_ratio[late].mean() > 1.1 * healthy.spread_ratio[late].mean()
    assert healthy.spread_ratio[healthy.t_h < cfg.diagnosis.spread_from_h].isna().all()


def test_rules_table_covers_every_fault_scenario():
    from coatshield.config import FAULT_SCENARIOS

    assert {r.fault for r in RULES} == set(FAULT_SCENARIOS) - {"none"}


# --- dissolution ------------------------------------------------------------------


def test_dissolution_projection_is_labelled_and_range_limited():
    cfg = load_config()
    slope, _ = dissolution.fit_line(cfg)
    assert slope == pytest.approx(6.4, abs=0.2)
    assert dissolution.t63_minutes(13.2, cfg) == pytest.approx(50.0, abs=2.0)
    assert list(dissolution.in_supported_range([9.0, 12.0, 17.0], cfg)) == [False, True, False]
    proj = dissolution.project(np.array([8.0, 10.0, 12.0, 18.0]), np.array([1, 2, 1]), cfg)
    assert proj["share_outside_range"] == pytest.approx(0.25)  # the 9 um bin is below 9.8
    assert "llustrative" in proj["label"]


# --- validation harness -------------------------------------------------------


def test_validation_cases_cover_the_guide_grid():
    cfg = load_config()
    assert len(grid_cases(cfg)) == 5 * 4 * 2 * 50
    assert len(fault_cases(cfg)) == 7 * 20
    assert len(grid_cases(cfg, quick=True)) == 2 * 2 * 2 * cfg.validation.quick_n_seeds


def test_run_case_scores_estimators_and_controllers():
    case = {"kind": "grid", "scenario": "none", "m": 3.0, "k": 1.0, "gamma": 0.0, "seed": 5}
    base = load_config(overrides={"validation.n_pellets": 20000})
    row = run_case(case, base.model_dump_json())
    assert row["hybrid_d10_mae"] < 0.3 < row["raw_d10_mae"]
    assert row["C2_true_below_spec_pct"] > row["C3_true_below_spec_pct"]
    assert {"C0_stop_h", "C3_excess_coating_pct", "ideal_stop_h", "raw_d10_bias"} <= set(row)


def test_raw_estimate_is_the_plain_sample_summary():
    x = rng("test.raw").normal(12, 2, 1000)
    est = raw_estimate(x)
    assert est["mean"] == pytest.approx(x.mean())
    assert est["d10"] == pytest.approx(np.quantile(x, 0.1), abs=0.02)
