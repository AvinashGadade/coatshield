import numpy as np
import pytest

from coatshield.config import load_config
from coatshield.oct import physics
from coatshield.oct.generator import default_params, labels, simulate
from coatshield.oct.preprocess import calibrated_dispersion, process
from coatshield.seeds import rng
from coatshield.solve.ellipse import fit_circle, fit_ellipse
from coatshield.solve.index import (
    assumed_index_error,
    index_from_anchor,
    index_from_fusion,
    index_from_ratio,
    index_from_reflectance,
    pool_index,
)
from coatshield.solve.refraction import thickness_along_normal, thickness_vertical
from coatshield.solve.thickness import measure_surfaces, pellet_thickness


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def solved(cfg, boundary_noise_px=0.0, calibration_error=0.0, **params):
    """Simulate a scan and run the solver on its surfaces (labels plus localisation noise)."""
    p = default_params(cfg, **params)
    raw = simulate(p, cfg)
    out = process(raw.spectra, raw.background, raw.x_um, cfg.oct,
                  dispersion=calibrated_dispersion(cfg))
    lab = labels(p, cfg, out.x_um, out.crop_start_px)
    gen = rng(f"test.solve.{p.seed}")
    outer = lab["outer_px"] + boundary_noise_px * gen.standard_normal(lab["outer_px"].size)
    inner = lab["inner_px"] + boundary_noise_px * gen.standard_normal(lab["inner_px"].size)
    surf = measure_surfaces(out.db, outer, inner, lab["valid"], out.x_um, out.depth_px_um,
                            out.crop_start_px, out.reflector_db, cfg, calibration_error)
    return p, surf


# --- surface fit ----------------------------------------------------------------


def test_circle_fit_recovers_an_arc():
    gen = rng("test.circle")
    angle = np.radians(np.linspace(-25, 25, 80))
    x = 12.0 + 366.0 * np.sin(angle) + 0.3 * gen.standard_normal(80)
    z = 250.0 + 366.0 * (1 - np.cos(angle)) + 0.3 * gen.standard_normal(80)
    c = fit_circle(x, z)
    assert c.radius == pytest.approx(366.0, rel=0.03)
    assert c.xc == pytest.approx(12.0, abs=1.0)
    assert c.zc == pytest.approx(250.0 + 366.0, rel=0.03)
    assert c.residual_rms == pytest.approx(0.3, abs=0.1)


def test_ellipse_fit_recovers_axes_and_orientation():
    gen = rng("test.ellipse")
    t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    a, b, phi = 400.0, 320.0, np.radians(20.0)
    x = 5 + a * np.cos(t) * np.cos(phi) - b * np.sin(t) * np.sin(phi)
    z = -8 + a * np.cos(t) * np.sin(phi) + b * np.sin(t) * np.cos(phi)
    e = fit_ellipse(x + 0.5 * gen.standard_normal(200), z + 0.5 * gen.standard_normal(200))
    assert (e.semi_major, e.semi_minor) == pytest.approx((a, b), rel=0.01)
    assert (e.xc, e.zc) == pytest.approx((5.0, -8.0), abs=0.5)
    assert e.aspect == pytest.approx(1.25, rel=0.01)
    assert np.tan(e.angle) == pytest.approx(np.tan(phi), abs=0.02)


# --- refraction -----------------------------------------------------------------


def test_refraction_correction_inverts_the_sphere_geometry():
    radius, thickness, n = 366.0, 16.0, 1.48
    x = np.linspace(-150, 150, 31)
    geo = physics.sphere_geometry(x, radius, thickness, n)
    optical = n * geo.path_um
    corrected = thickness_along_normal(optical, geo.theta, radius, n)
    assert np.allclose(corrected, thickness, atol=1e-9)
    naive = thickness_vertical(optical, n)
    beyond = np.abs(np.degrees(geo.theta)) > 10
    assert (naive[beyond] > thickness + 0.1).all()
    assert np.abs(naive[beyond] - thickness).mean() > 20 * np.abs(corrected[beyond]
                                                                  - thickness).mean()


def test_assumed_index_helper_reproduces_the_published_errors():
    assert 100 * assumed_index_error(1.45, 1.5) == pytest.approx(-3.3, abs=0.05)
    assert 100 * assumed_index_error(1.30, 1.5) == pytest.approx(-13.3, abs=0.05)
    assert 100 * assumed_index_error(1.60, 1.5) == pytest.approx(6.7, abs=0.05)


# --- refractive index methods ---------------------------------------------------


def test_reflectance_inversion_is_exact_on_ideal_levels(cfg):
    from coatshield.oct.preprocess import sensitivity_db

    oc = cfg.oct
    n, depth = 1.42, 250.0
    r = abs(physics.fresnel_amplitude(1.0, n))
    surface_db = 20 * np.log10(r) + sensitivity_db(oc, depth)
    reflector_db = 10 * np.log10(oc.reflector_reflectance) + sensitivity_db(oc, oc.reflector_um)
    assert index_from_reflectance(surface_db, reflector_db, depth, cfg) == pytest.approx(n)
    # A 10% error in the reflector calibration moves n by about 0.05.
    high = index_from_reflectance(surface_db, reflector_db, depth, cfg, calibration_error=0.10)
    low = index_from_reflectance(surface_db, reflector_db, depth, cfg, calibration_error=-0.10)
    assert 0.03 < high - n < 0.08 and 0.03 < n - low < 0.08
    # The calibrated roll-off falls with depth.
    assert sensitivity_db(oc, 400.0) < sensitivity_db(oc, 100.0)


def test_ratio_variant_needs_no_calibration(cfg):
    n = 1.45
    r_s = abs(physics.fresnel_amplitude(1.0, n))
    ratio = abs(physics.fresnel_amplitude(n, cfg.core.n)) * (1 - r_s**2) / r_s
    for offset_db in (0.0, 17.0):  # any common gain cancels
        got = index_from_ratio(20 * np.log10(ratio) + offset_db, offset_db, cfg)
        assert got == pytest.approx(n, abs=1e-6)
    assert np.isnan(index_from_ratio(40.0, 0.0, cfg))  # an impossible ratio gives no answer


def test_fusion_and_anchor_methods():
    # 16 um of coating at n = 1.44 on 700 um cores.
    assert index_from_fusion(16 * 1.44, 732.0, 700.0) == pytest.approx(1.44)
    assert np.isnan(index_from_fusion(10.0, 700.0, 700.0))
    gen = rng("test.anchor")
    true = gen.uniform(10, 20, 30)
    microscopy = true + 0.5 * gen.standard_normal(30)
    assert index_from_anchor(1.44 * true, microscopy) == pytest.approx(1.44, abs=0.02)


def test_pooling_tightens_with_the_number_of_pellets():
    gen = rng("test.pool")
    few = pool_index(1.48 + 0.03 * gen.standard_normal(25))
    many = pool_index(1.48 + 0.03 * gen.standard_normal(2500))
    assert many.n == pytest.approx(1.48, abs=0.005)
    assert many.standard_error < few.standard_error / 5
    assert pool_index(np.array([np.nan])).count == 0


# --- end to end on synthetic scans ----------------------------------------------


def test_solver_recovers_radius_and_thickness_with_the_true_index(cfg):
    p, surf = solved(cfg, boundary_noise_px=1.0, snr_db=35.0)
    result = pellet_thickness(surf, p.n_coat)
    assert result.radius_um == pytest.approx(p.radius_um, rel=0.03)
    assert result.thickness_um == pytest.approx(p.thickness_um, abs=0.3)
    assert result.fit_residual_um < 1.0
    assert result.n_ascans >= cfg.solve.min_ascans
    assert 0 < result.uncertainty_um < 0.3
    assert result.optical_um == pytest.approx(p.n_coat * p.thickness_um, rel=0.03)
    assert np.degrees(np.abs(surf.theta)).max() <= cfg.solve.max_angle_deg


def test_refraction_correction_helps_on_real_scan_geometry(cfg):
    p, surf = solved(cfg, radius_um=250.0, thickness_um=25.0, snr_db=40.0)
    with_correction = pellet_thickness(surf, p.n_coat).thickness_um
    without = pellet_thickness(surf, p.n_coat, refraction=False).thickness_um
    assert abs(with_correction - p.thickness_um) < abs(without - p.thickness_um)
    assert abs(with_correction - p.thickness_um) < 0.2


def test_pooled_reflectance_index_is_within_0p01_at_snr_35(cfg):
    n_true = 1.44
    values = [solved(cfg, n_coat=n_true, snr_db=35.0, seed=s)[1].n_reflectance for s in range(10)]
    pooled = pool_index(np.array(values))
    assert pooled.count == 10
    assert pooled.n == pytest.approx(n_true, abs=0.01)


def test_thickness_error_from_assuming_1p5(cfg):
    p, surf = solved(cfg, n_coat=1.30, snr_db=40.0)
    assumed = pellet_thickness(surf, cfg.solve.assumed_n).thickness_um
    assert assumed / p.thickness_um - 1 == pytest.approx(assumed_index_error(1.30, 1.5), abs=0.01)


def test_too_few_valid_ascans_gives_no_result(cfg):
    oc = cfg.oct
    nothing = np.zeros(oc.out_ascans, bool)
    rows = np.full(oc.out_ascans, 50.0)
    x = np.linspace(-250, 250, oc.out_ascans)
    db = np.zeros((oc.out_ascans, oc.depth_pixels))
    assert measure_surfaces(db, rows, rows + 60, nothing, x, 0.36, 600, 30.0, cfg) is None
