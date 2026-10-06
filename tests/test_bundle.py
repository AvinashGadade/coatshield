import numpy as np
import pandas as pd
import pytest

from coatshield.bundle import build_bundle, bundle_key, load_bundle, save_bundle
from coatshield.config import load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config(overrides={"batch.n_pellets": 20000})


@pytest.fixture(scope="module")
def bundle(cfg):
    return build_bundle(cfg, bootstrap="needed")


def test_bundle_holds_every_view_the_dashboard_needs(bundle, cfg):
    assert bundle.key == bundle_key(cfg)
    assert bundle.meta["config_hash"] == cfg.analysis_hash()
    assert len(bundle.truth) == len(bundle.est) == len(bundle.signals) == bundle.measured_cum.size
    assert list(bundle.controllers.controller) == ["C0", "C1", "C2", "C3"]
    assert len(bundle.samples) <= 2 * cfg.app.sample_rows
    for dist in (bundle.dist_truth, bundle.dist_raw, bundle.dist_corrected):
        assert dist.shape == (bundle.frame_steps.size, bundle.thickness_edges.size - 1)
    assert np.all(np.diff(bundle.measured_cum) >= 0)
    stops = bundle.meta["stop_steps"]
    assert all(stops[c] in bundle.frame_steps for c in stops if stops[c] is not None)


def test_distributions_agree_with_the_headline_numbers(bundle, cfg):
    """d10 read off the stored curves matches the estimator table for the same step."""
    centres = 0.5 * (bundle.thickness_edges[:-1] + bundle.thickness_edges[1:])
    j = int(np.argmin(np.abs(bundle.frame_steps - bundle.meta["stop_steps"]["C3"])))
    step = bundle.frame_steps[j]

    def d10(shares):
        return float(np.interp(0.1, np.cumsum(shares), centres))

    assert bundle.dist_corrected[j].sum() == pytest.approx(1.0)
    assert d10(bundle.dist_corrected[j]) == pytest.approx(bundle.est.hybrid_d10[step], abs=0.2)
    assert d10(bundle.dist_raw[j]) == pytest.approx(bundle.est.raw_d10[step], abs=0.2)
    assert d10(bundle.dist_truth[j]) == pytest.approx(bundle.truth.d10_um[step], abs=0.2)
    assert d10(bundle.dist_raw[j]) > d10(bundle.dist_truth[j]) + 0.4


def test_bundle_round_trips_through_files(bundle, tmp_path):
    path = save_bundle(bundle, tmp_path)
    assert sum(f.stat().st_size for f in path.iterdir()) < 3_000_000
    loaded = load_bundle(bundle.key, tmp_path)
    pd.testing.assert_frame_equal(loaded.est, bundle.est)
    pd.testing.assert_frame_equal(loaded.controllers, bundle.controllers)
    assert np.allclose(loaded.dist_corrected, bundle.dist_corrected, atol=1e-6)
    assert loaded.meta["stop_steps"] == bundle.meta["stop_steps"]
    assert load_bundle("missing", tmp_path) is None
