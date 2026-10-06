import time

import numpy as np
import pytest

from coatshield.config import load_config
from coatshield.twin.batch import run_batch, twin_hash
from coatshield.twin.faults import fault_state
from coatshield.twin.growth import make_recipe
from coatshield.twin.population import ABSORBED, SINGLE, TWIN, make_population

SMALL = {"batch.n_pellets": 40000}


def cfg_small(**overrides):
    return load_config(overrides={**SMALL, **overrides})


@pytest.fixture(scope="module")
def default_run():
    return run_batch(cfg_small(), cache=False)


def row_at(result, hours):
    return result.truth.iloc[(result.truth.t_h - hours).abs().argmin()]


# --- population ---------------------------------------------------------------


def test_population_is_lognormal_around_the_preset_median():
    pop = make_population(cfg_small())
    assert np.median(pop.core_um) == pytest.approx(700, rel=0.01)
    assert np.std(np.log(pop.core_um)) == pytest.approx(0.18, rel=0.03)
    assert (pop.kind == SINGLE).all()


def test_real_pellet_count_for_30kg_of_700um_cores():
    pop = make_population(cfg_small())
    assert 0.8e8 < pop.n_real < 1.3e8


# --- determinism --------------------------------------------------------------


def test_same_config_and_seed_give_identical_results(default_run):
    again = run_batch(cfg_small(), cache=False)
    assert default_run.truth.equals(again.truth)
    assert default_run.samples.equals(again.samples)
    assert np.array_equal(default_run.thickness_hist, again.thickness_hist)


def test_different_seed_gives_different_results(default_run):
    other = run_batch(cfg_small(seed=1), cache=False)
    assert not default_run.truth.d10_um.equals(other.truth.d10_um)


def test_cache_returns_the_same_result(tmp_path):
    cfg = cfg_small(**{"batch.duration_h": 0.5})
    first = run_batch(cfg, cache_dir=tmp_path)
    assert list(tmp_path.glob("*.pkl"))
    second = run_batch(cfg, cache_dir=tmp_path)
    assert first.truth.equals(second.truth)
    assert first.config_hash == second.config_hash == twin_hash(cfg)
    assert twin_hash(cfg.with_overrides({"estimator.method": "raw"})) == twin_hash(cfg)


# --- mass balance and growth --------------------------------------------------


@pytest.mark.parametrize("scenario", ["none", "over_wetting", "spray_drying", "nozzle_block"])
def test_mass_balance_within_0p1_percent(scenario):
    cfg = cfg_small(**{"fault.scenario": scenario, "batch.duration_h": 4.0})
    last = run_batch(cfg, cache=False).truth.iloc[-1]
    assert last.coating_volume_cm3 == pytest.approx(last.deposited_volume_cm3, rel=1e-3)
    assert last.deposited_volume_cm3 > 0


def test_growth_ratio_within_published_range(default_run):
    lo, hi = load_config().wurster.growth_ratio_target
    ratio = row_at(default_run, 3.0).growth_ratio
    assert lo <= ratio <= hi
    assert ratio == pytest.approx(1.6, abs=0.1)


def test_no_size_dependence_when_k_is_zero():
    res = run_batch(cfg_small(**{"wurster.size_growth_exponent_k": 0.0}), cache=False)
    assert row_at(res, 3.0).growth_ratio == pytest.approx(1.0, abs=0.02)


def _loglog_slope(truth, column):
    sel = truth[(truth.t_h >= 0.5) & (truth.t_h <= 3.0)]
    return np.polyfit(np.log(sel.t_h), np.log(sel[column]), 1)[0]


def test_cv_decays_as_sqrt_one_over_t(default_run):
    # Size-adjusted (pellet-to-pellet at equal size) variability follows the published law.
    assert _loglog_slope(default_run.truth, "cv_within_size") == pytest.approx(-0.5, abs=0.1)


def test_total_cv_decays_as_sqrt_one_over_t_without_size_effect():
    res = run_batch(cfg_small(**{"wurster.size_growth_exponent_k": 0.0}), cache=False)
    assert _loglog_slope(res.truth, "cv") == pytest.approx(-0.5, abs=0.1)


def test_weight_gain_about_11_percent_at_16um(default_run):
    truth = default_run.truth
    row = truth.iloc[(truth.mean_um - 16.0).abs().argmin()]
    assert row.mean_um == pytest.approx(16.0, abs=0.1)
    assert row.weight_gain_pct == pytest.approx(11.0, abs=1.5)


def test_recipe_rate_matches_deposit_per_pass(default_run):
    cfg = load_config()
    per_hour = cfg.wurster.deposit_nm_per_pass / 1000 / cfg.wurster.cycle_time_s * 3600
    assert row_at(default_run, 4.0).mean_um / 4.0 == pytest.approx(per_hour, rel=0.1)
    recipe = make_recipe(cfg)
    assert recipe.target_weight_gain_pct == pytest.approx(11.0, abs=1.0)
    assert recipe.target_solids_g > 0


# --- window sampling and measurement ------------------------------------------


def _window_d10(result, t_from_h=5.5, t_to_h=5.6):
    s = result.samples
    sel = s[(s.t_s > t_from_h * 3600) & (s.t_s <= t_to_h * 3600) & s.accepted]
    return np.quantile(sel.thickness_um, 0.1), row_at(result, t_to_h).d10_um


def test_raw_sample_overstates_d10_at_m3(default_run):
    raw, true = _window_d10(default_run)
    assert raw > true


def test_size_bias_matches_theory_without_measurement_noise():
    # A size-biased draw from a log-normal bed shifts log thickness by m * k * sigma^2.
    res = run_batch(cfg_small(**{"measurement.sigma_um": 0.0}), cache=False)
    raw, true = _window_d10(res)
    cfg = load_config()
    shift = (cfg.window.size_bias_m * cfg.wurster.size_growth_exponent_k
             * cfg.pellet.size_sigma_log**2)
    assert raw / true == pytest.approx(np.exp(shift), abs=0.03)


def test_no_bias_without_size_selection():
    res = run_batch(cfg_small(**{"window.size_bias_m": 0.0, "measurement.sigma_um": 0.0}),
                    cache=False)
    raw, true = _window_d10(res)
    assert raw == pytest.approx(true, abs=0.25)


def test_sampled_sizes_follow_the_size_bias():
    means = {}
    for m in (0.0, 3.0):
        res = run_batch(cfg_small(**{"window.size_bias_m": m, "batch.duration_h": 0.5}),
                        cache=False)
        s = res.samples
        means[m] = s[s.true_class == SINGLE].size_um.mean()
    sigma = load_config().pellet.size_sigma_log
    assert means[3.0] / means[0.0] == pytest.approx(np.exp(3 * sigma**2), rel=0.02)


def test_hidden_selection_favours_thicker_pellets_at_equal_size():
    kw = {"window.size_bias_m": 0.0, "measurement.sigma_um": 0.0}
    base = run_batch(cfg_small(**kw), cache=False)
    hidden = run_batch(cfg_small(**kw, **{"window.hidden_selection_gamma": 1.0}), cache=False)

    def mean_true(res):
        s = res.samples
        return s[(s.t_s > 3 * 3600) & (s.true_class == SINGLE)].true_thickness_um.mean()

    assert mean_true(hidden) > mean_true(base) * 1.01


def test_sample_records_have_the_expected_fields(default_run):
    cols = {"t_s", "size_um", "thickness_um", "gate_class", "confidence", "accepted",
            "true_class", "true_thickness_um", "true_size_um"}
    assert cols <= set(default_run.samples.columns)
    cfg = load_config()
    per_step = default_run.samples.groupby("t_s").size()
    expected = cfg.window.objects_per_min * cfg.batch.step_s / 60
    assert per_step.min() >= expected
    assert per_step.mean() == pytest.approx(expected + cfg.wurster.fines_per_min, rel=0.05)


def test_assumed_index_scales_the_measurement():
    kw = {"measurement.sigma_um": 0.0, "batch.duration_h": 1.0}
    res = run_batch(cfg_small(**kw, **{"measurement.n_assumed": 1.5, "coating.n": 1.3}),
                    cache=False)
    s = res.samples[res.samples.accepted & (res.samples.true_class == SINGLE)]
    assert np.allclose(s.thickness_um / s.true_thickness_um, 1.3 / 1.5)


def test_gate_recall_and_twin_misread():
    kw = {"wurster.fusion_rate_per_h": 0.05, "measurement.sigma_um": 0.0,
          "measurement.undecided_base": 0.0, "measurement.twin_misread_model": "thin"}
    s = run_batch(cfg_small(**kw), cache=False).samples
    twins = s[s.true_class == TWIN]
    assert len(twins) > 500
    recall = (twins.gate_class == TWIN).mean()
    assert recall == pytest.approx(load_config().gate.twin_recall_target, abs=0.03)
    leaked = twins[twins.accepted]
    assert np.allclose(leaked.thickness_um / leaked.true_thickness_um, 0.45)


# --- faults -------------------------------------------------------------------


def _fault_pair(scenario, hours=5.0):
    kw = {"batch.duration_h": hours}
    base = run_batch(cfg_small(**kw), cache=False)
    fault = run_batch(cfg_small(**kw, **{"fault.scenario": scenario}), cache=False)
    return base.truth.iloc[-1], fault.truth.iloc[-1], base, fault


def test_fault_state_switches_on_at_start_time():
    cfg = cfg_small(**{"fault.scenario": "nozzle_block", "fault.start_h": 2.0})
    assert fault_state(cfg, 1.9 * 3600).spray_factor == 1.0
    assert fault_state(cfg, 2.1 * 3600).spray_factor == pytest.approx(0.7)
    fouling = cfg_small(**{"fault.scenario": "window_fouling", "fault.start_h": 1.0})
    assert fault_state(fouling, 0.5 * 3600).fouling == 0.0
    assert 0.0 < fault_state(fouling, 3.0 * 3600).fouling < 1.0
    assert fault_state(fouling, 9.0 * 3600).fouling == 1.0


def test_substrate_shift_moves_core_median_and_thickens_coat():
    base, fault, _, _ = _fault_pair("substrate_shift")
    pop = make_population(cfg_small(**{"fault.scenario": "substrate_shift"}))
    assert np.median(pop.core_um) == pytest.approx(767, rel=0.01)
    assert fault.sprayed_solids_g == pytest.approx(base.sprayed_solids_g)
    assert fault.mean_um > base.mean_um * 1.05


def test_nozzle_block_slows_growth():
    base, fault, _, _ = _fault_pair("nozzle_block")
    assert fault.growth_um_per_h == pytest.approx(0.7 * base.growth_um_per_h, rel=0.08)
    assert fault.sprayed_solids_g < base.sprayed_solids_g


def test_over_wetting_raises_agglomerates():
    base, fault, _, _ = _fault_pair("over_wetting")
    assert fault.agglomerate_pct > 4 * base.agglomerate_pct
    assert fault.growth_um_per_h < base.growth_um_per_h


def test_spray_drying_raises_fines_and_slows_growth():
    base, fault, _, _ = _fault_pair("spray_drying")
    assert fault.fines_per_min == pytest.approx(10 * base.fines_per_min)
    assert fault.growth_um_per_h == pytest.approx(0.7 * base.growth_um_per_h, rel=0.08)


def test_maldistribution_raises_variability():
    base, fault, _, _ = _fault_pair("maldistribution")
    assert fault.cv_within_size > 1.15 * base.cv_within_size
    assert fault.cv > base.cv


def test_window_fouling_raises_undecided_share():
    _, _, base, fault = _fault_pair("window_fouling")

    def undecided(res):
        s = res.samples
        s = s[(s.t_s > 4 * 3600) & (s.gate_class == SINGLE)]
        return 1 - s.accepted.mean()

    assert undecided(fault) > undecided(base) + 0.15
    assert fault.truth.iloc[-1].fouling > 0.5


def test_agglomerates_are_flagged_and_leave_the_spec_population():
    cfg = cfg_small(**{"wurster.fusion_rate_per_h": 0.05, "batch.duration_h": 2.0})
    res = run_batch(cfg, cache=False)
    last = res.truth.iloc[-1]
    assert last.n_twin > 0
    assert last.n_single + 2 * last.n_twin == cfg.batch.n_pellets
    assert last.agglomerate_pct == pytest.approx(100 * last.n_twin / (last.n_single + last.n_twin))
    assert ABSORBED not in set(res.samples.true_class)


# --- speed --------------------------------------------------------------------


@pytest.mark.slow
def test_full_size_batch_runs_in_under_15_seconds():
    cfg = load_config(overrides={"batch.duration_h": 8.0, "batch.n_pellets": 400000})
    start = time.perf_counter()
    res = run_batch(cfg, cache=False)
    elapsed = time.perf_counter() - start
    assert len(res.truth) == 480
    assert elapsed < 15, f"took {elapsed:.1f} s"


# --- pass-draw kernel -----------------------------------------------------------

TAIL = 1e-10


def _passes(lam_value, sigma_ln, n=200_000, key=123, step=0):
    from coatshield.twin.kernels import build_alias, draw_passes, mixed_poisson_pmf

    prob, alias = build_alias(mixed_poisson_pmf(np.array([lam_value]), sigma_ln, TAIL))
    cum = np.zeros(n)
    weight = np.empty(n)
    bounds = np.linspace(0, n, 17).astype(np.int64)
    total = draw_passes(key, step, np.zeros(n, np.int32), prob, alias, np.ones(n), cum, weight,
                        bounds)
    assert total == weight.sum()
    return cum


def test_pass_table_matches_poisson_without_spread():
    from scipy.stats import poisson

    from coatshield.twin.kernels import mixed_poisson_pmf

    lam = np.array([0.5, 10.3, 60.0])
    pmf = mixed_poisson_pmf(lam, 0.0, TAIL)
    k = np.arange(pmf.shape[1])
    assert np.allclose(pmf, poisson.pmf(k[None, :], lam[:, None]), atol=1e-9)


@pytest.mark.parametrize("rsd", [0.19, 0.35, 0.63])
def test_pass_table_moments_with_lognormal_spread(rsd):
    from coatshield.twin.kernels import mixed_poisson_pmf

    lam = np.array([4.0, 10.3, 25.0])
    pmf = mixed_poisson_pmf(lam, np.sqrt(np.log1p(rsd**2)), TAIL)
    k = np.arange(pmf.shape[1])
    mean = pmf @ k
    var = pmf @ k**2 - mean**2
    assert np.allclose(mean, lam, rtol=1e-6)
    assert np.allclose(var, lam + (lam * rsd) ** 2, rtol=1e-4)


@pytest.mark.parametrize("lam", [0.5, 4.0, 10.3, 60.0])
def test_kernel_poisson_mean_and_variance(lam):
    p = _passes(lam, 0.0)
    assert p.mean() == pytest.approx(lam, rel=0.01)
    assert p.var() == pytest.approx(lam, rel=0.03)
    assert np.array_equal(p, np.round(p)) and p.min() >= 0


def test_kernel_lognormal_spread_adds_expected_variance():
    lam, rsd = 10.3, 0.35
    p = _passes(lam, np.sqrt(np.log1p(rsd**2)))
    assert p.mean() == pytest.approx(lam, rel=0.01)
    assert p.var() == pytest.approx(lam + (lam * rsd) ** 2, rel=0.05)


def test_kernel_streams_are_reproducible_and_distinct():
    a, b = _passes(10.3, 0.3), _passes(10.3, 0.3)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, _passes(10.3, 0.3, step=1))
    assert not np.array_equal(a, _passes(10.3, 0.3, key=124))
    assert abs(np.corrcoef(a, _passes(10.3, 0.3, step=1))[0, 1]) < 0.01
    assert abs(np.corrcoef(a[:-1], a[1:])[0, 1]) < 0.01


def test_expected_share_is_exact_despite_rate_classes():
    """Pass-rate classes must not leave a sawtooth in growth against size."""
    res = run_batch(cfg_small(**{"batch.duration_h": 3.0}), cache=False)
    cfg = load_config()
    k = cfg.wurster.size_growth_exponent_k
    assert res.truth.iloc[-1].growth_exponent == pytest.approx(k, abs=0.01)
