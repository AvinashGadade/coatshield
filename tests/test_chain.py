import numpy as np
import pytest

from coatshield.chain.pipeline import LabelSegmenter, SampledObject, process_object
from coatshield.compliance import confidence as conf
from coatshield.config import load_config
from coatshield.twin.population import FINES, SINGLE, TWIN


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def make(cfg, **kw):
    base = dict(true_class=SINGLE, diameter_um=732.0, thickness_um=16.0, n_coat=cfg.coating.n,
                n_core=cfg.core.n, speed_m_s=0.3, snr_db=35.0, seed=1)
    base.update(kw)
    return SampledObject(**base)


def run(cfg, **kw):
    return process_object(make(cfg, **kw), cfg, LabelSegmenter(cfg), cfg.coating.n)


def test_single_pellet_runs_the_whole_chain_and_keeps_every_intermediate(cfg):
    rec = run(cfg)
    assert rec.status == "measured" and rec.gate_class == "single"
    assert rec.thickness_um == pytest.approx(16.0, abs=0.5)
    assert rec.radius_um == pytest.approx(366.0, rel=0.05)
    assert 0 < rec.confidence <= 1 and rec.uncertainty_um > 0
    for key in ("camera", "gate_features", "spectrum", "scan", "outer_px", "inner_px", "valid",
                "circle", "used", "reflector_db"):
        assert key in rec.intermediates, key
    assert rec.intermediates["camera"].shape == (256, 256)
    assert rec.intermediates["scan"].shape == (128, 512)
    assert rec.intermediates["spectrum"].shape == (cfg.oct.n_pixels,)
    row = rec.summary()
    assert row["error_um"] == pytest.approx(rec.thickness_um - 16.0)
    assert row["true_thickness_um"] == 16.0 and "intermediates" not in row


def test_chain_is_deterministic(cfg):
    a, b = run(cfg, seed=5), run(cfg, seed=5)
    assert a.thickness_um == b.thickness_um and a.confidence == b.confidence
    assert np.array_equal(a.intermediates["scan"], b.intermediates["scan"])
    assert run(cfg, seed=6).thickness_um != a.thickness_um


@pytest.mark.parametrize("true_class, expected", [(TWIN, {"twin", "touching"}), (FINES, {"fines"})])
def test_non_singles_are_gated_before_oct(cfg, true_class, expected):
    rec = run(cfg, true_class=true_class, diameter_um=500.0 if true_class == TWIN else 60.0)
    assert rec.status == "gated" and rec.gate_class in expected
    assert np.isnan(rec.thickness_um)
    assert "scan" not in rec.intermediates and "camera" in rec.intermediates


def test_assumed_index_error_flows_through_the_chain(cfg):
    obj = make(cfg, n_coat=1.35, seed=3)
    right = process_object(obj, cfg, LabelSegmenter(cfg), 1.35)
    wrong = process_object(obj, cfg, LabelSegmenter(cfg), cfg.solve.assumed_n)
    assert right.thickness_um == pytest.approx(16.0, abs=0.5)
    assert wrong.thickness_um / right.thickness_um == pytest.approx(1.35 / 1.5, rel=0.01)


def test_keep_false_drops_the_arrays(cfg):
    rec = process_object(make(cfg), cfg, LabelSegmenter(cfg), cfg.coating.n, keep=False)
    assert rec.status == "measured" and rec.intermediates == {}


def test_confidence_fusion_and_undecided_state(cfg):
    good = conf.fit_confidence(0.2, 0.1, cfg)
    poor = conf.fit_confidence(8.0, 8.0, cfg)
    assert 0.9 < good <= 1.0 and poor < 0.05
    used = np.array([True, True, False])
    seg = conf.segmentation_confidence(np.array([0.9, 0.8, 0.0]), np.array([0.7, 0.9, 0.0]), used)
    assert seg == pytest.approx(0.75)
    assert conf.segmentation_confidence(np.ones(3), np.ones(3), np.zeros(3, bool)) == 0.0
    assert conf.fuse(seg, good) == pytest.approx(seg * good)
    assert conf.is_decided(cfg.chain.confidence_min, cfg)
    assert not conf.is_decided(cfg.chain.confidence_min - 0.01, cfg)


def test_threshold_tuning_bounds_the_share_of_large_errors(cfg):
    from coatshield.seeds import rng

    gen = rng("test.threshold")
    scores = gen.random(4000)
    # Low-confidence readings are the ones with large errors.
    error = np.where(scores < 0.3, gen.uniform(0, 6, 4000), gen.uniform(0, 1.5, 4000))
    tuned = conf.tune_threshold(scores, error, cfg)
    assert tuned["accepted_bad_share"] <= cfg.chain.max_error_share
    assert 0.2 < tuned["confidence_min"] < 0.35
    assert tuned["undecided_share"] == pytest.approx(0.28, abs=0.08)
    hopeless = conf.tune_threshold(scores, np.full(4000, 9.0), cfg)
    assert hopeless["undecided_share"] == 1.0


# --- error model ----------------------------------------------------------------


def calibration_rows(cfg, n=3000):
    """Synthetic calibration records with a known error structure."""
    import pandas as pd

    from coatshield.seeds import rng

    gen = rng("test.error_model")
    fouling = gen.random(n)
    snr = gen.uniform(15, 45, n)
    thickness = gen.uniform(2, 40, n)
    klass = np.where(gen.random(n) < 0.15, TWIN, SINGLE)
    gated = np.where(klass == TWIN, gen.random(n) < 0.97, gen.random(n) < 0.04)
    undecided = gen.random(n) < 1 / (1 + np.exp(-(-4 + 5 * fouling)))
    status = np.where(gated, "gated", np.where(undecided, "undecided", "measured"))
    measured = thickness + 0.3 + (0.4 + 1.2 * fouling) * gen.standard_normal(n)
    return pd.DataFrame({
        "status": status, "thickness_um": np.where(status == "gated", np.nan, measured),
        "true_true_class": klass, "true_thickness_um": thickness, "true_fouling": fouling,
        "true_snr_db": snr, "true_pigment": gen.random(n) * 0.2,
    })


@pytest.fixture(scope="module")
def error_model(cfg):
    from coatshield.chain.error_model import fit_error_model

    return fit_error_model(calibration_rows(cfg), cfg, {"segmenter": "test"})


def test_error_model_recovers_bias_spread_leak_and_undecided_curve(cfg, error_model):
    m = error_model
    bias, spread_clean = m.bias_spread(0.1, 30.0, 16.0)
    _, spread_fouled = m.bias_spread(0.9, 30.0, 16.0)
    assert bias == pytest.approx(0.3, abs=0.15)
    assert spread_clean == pytest.approx(0.52, abs=0.15)
    assert spread_fouled > spread_clean + 0.6
    assert m.twin_leak == pytest.approx(0.03, abs=0.02)
    assert m.single_reject == pytest.approx(0.04, abs=0.015)
    assert m.p_undecided(0.0, 0.05, 35.0) < 0.06 < 0.4 < m.p_undecided(1.0, 0.05, 35.0)
    assert m.meta["segmenter"] == "test" and m.meta["n_objects"] == 3000


def test_error_model_saves_with_a_hash_and_refuses_edits(cfg, error_model, tmp_path):
    import json

    from coatshield.chain.error_model import load_error_model

    path = tmp_path / "error_model.json"
    digest = error_model.save(path)
    assert len(digest) == 64 and load_error_model(path).sha256() == digest
    data = json.loads(path.read_text())
    data["twin_leak"] = 0.0
    edited = tmp_path / "edited.json"
    edited.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="does not match"):
        load_error_model(edited)


def test_twin_uses_the_error_model_when_configured(cfg, error_model, tmp_path):
    from coatshield.twin.batch import run_batch, twin_hash

    path = tmp_path / "error_model.json"
    error_model.save(path)
    small = {"batch.n_pellets": 20000, "batch.duration_h": 3.0, "fault.scenario": "window_fouling",
             "fault.start_h": 0.5, "fault.fouling_ramp_h": 2.0}
    plain = cfg.with_overrides(small)
    modelled = plain.with_overrides({"measurement.error_model_path": str(path)})
    assert twin_hash(plain) != twin_hash(modelled)
    res = run_batch(modelled, cache=False)
    s = res.samples
    ok = s.accepted & (s.true_class == SINGLE)
    early, late = ok & (s.t_s < 0.5 * 3600), ok & (s.t_s > 2.6 * 3600)
    err = s.thickness_um - s.true_thickness_um
    assert err[early].mean() == pytest.approx(0.3, abs=0.1)  # the model's bias
    assert err[late].std() > 2 * err[early].std()  # its spread grows with fouling
    scanned = s.gate_class == SINGLE
    assert s.accepted[scanned & (s.t_s > 2.6 * 3600)].mean() < 0.75  # more undecided when fouled
    assert res.samples.equals(run_batch(modelled, cache=False).samples)
