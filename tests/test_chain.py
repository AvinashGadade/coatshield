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
    good = conf.fit_confidence(0.2, 0.03, cfg)
    poor = conf.fit_confidence(3.0, 0.5, cfg)
    assert 0.9 < good <= 1.0 and poor < 0.05
    used = np.array([True, True, False])
    seg = conf.segmentation_confidence(np.array([0.9, 0.8, 0.0]), np.array([0.7, 0.9, 0.0]), used)
    assert seg == pytest.approx(0.75)
    assert conf.segmentation_confidence(np.ones(3), np.ones(3), np.zeros(3, bool)) == 0.0
    assert conf.fuse(1.0, seg, good) == pytest.approx(seg * good)
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
