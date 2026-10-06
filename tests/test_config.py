import pytest
from pydantic import ValidationError

from coatshield.config import (
    PRODUCT_PRESETS,
    SCALE_PRESETS,
    list_presets,
    load_config,
)


def test_default_loads_with_guide_values():
    cfg = load_config()
    assert cfg.pellet.core_median_um == 700
    assert cfg.pellet.size_sigma_log == 0.18
    assert cfg.wurster.cycle_time_s == 5.8
    assert cfg.wurster.cycle_time_rsd == 0.35
    assert cfg.wurster.deposit_nm_per_pass == 3
    assert cfg.wurster.growth_ratio_target == (1.08, 1.81)
    assert cfg.window.size_bias_m == 3
    assert cfg.window.objects_per_min == 500
    assert cfg.spec.d10_min_um == 12
    assert (cfg.oct.center_nm, cfg.oct.fwhm_nm) == (830, 200)
    assert cfg.oct.lateral_spot_um == 18
    assert cfg.oct.ascan_rate_hz == 250000
    assert cfg.coating.n == 1.48
    assert cfg.core.n == 1.54
    assert cfg.measurement.sigma_um == 1.0
    assert cfg.gate.twin_recall_target == 0.95


def test_all_presets_exist_and_load():
    assert set(PRODUCT_PRESETS + SCALE_PRESETS) == set(list_presets())
    for name in list_presets():
        load_config(presets=name)


def test_preset_overrides_default_and_keeps_the_rest():
    cfg = load_config(presets=["taste_mask", "multilab_2kg"])
    assert cfg.pellet.core_median_um == 250
    assert cfg.batch.mass_kg == 2
    assert cfg.pellet.size_sigma_log == load_config().pellet.size_sigma_log


def test_later_preset_wins():
    assert load_config(presets=["gpcg30_30kg", "fbc125"]).batch.mass_kg == 125


def test_unknown_preset_raises():
    with pytest.raises(FileNotFoundError):
        load_config(presets="nope")


def test_overrides_dotted_and_nested_return_a_copy():
    base = load_config()
    a = base.with_overrides({"window.size_bias_m": 0.0})
    b = base.with_overrides({"window": {"size_bias_m": 0.0}})
    assert a == b
    assert a.window.size_bias_m == 0.0
    assert a.window.objects_per_min == base.window.objects_per_min
    assert base.window.size_bias_m == 3.0


def test_validation_rejects_bad_values_and_unknown_keys():
    with pytest.raises(ValidationError):
        load_config(overrides={"pellet.core_median_um": -1})
    with pytest.raises(ValidationError):
        load_config(overrides={"pellet.not_a_key": 1})
    with pytest.raises(ValidationError):
        load_config(overrides={"fault.scenario": "gremlins"})


def test_config_is_frozen():
    cfg = load_config()
    with pytest.raises(ValidationError):
        cfg.pellet.core_median_um = 1.0


def test_hash_is_stable_and_sensitive():
    a, b = load_config(), load_config()
    assert a.hash() == b.hash()
    assert len(a.hash()) == 12
    assert a.with_overrides({"window.size_bias_m": 2.0}).hash() != a.hash()
    assert a.with_overrides({"seed": 1}).hash(exclude=("seed",)) == a.hash(exclude=("seed",))


def test_analysis_hash_ignores_sections_of_later_modules():
    base = load_config()
    assert base.with_overrides({"app.frame_every": 5}).analysis_hash() == base.analysis_hash()
    assert base.with_overrides({"oct.center_nm": 1300}).analysis_hash() == base.analysis_hash()
    assert base.with_overrides({"spec.d10_min_um": 11}).analysis_hash() != base.analysis_hash()
