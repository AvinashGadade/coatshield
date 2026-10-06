import numpy as np
import pytest
from scipy.signal import find_peaks, hilbert

from coatshield.config import load_config
from coatshield.oct import physics
from coatshield.oct.generator import (
    ABOVE,
    COATING,
    CORE,
    default_params,
    labels,
    scan_axis,
    simulate,
)
from coatshield.oct.preprocess import (
    calibrated_dispersion,
    depth_profile,
    estimate_dispersion,
    process,
    to_wavenumber,
)

FLAT = 1e7  # radius (um) large enough that the surface is flat across the field


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def clean(cfg, **overrides):
    """A config without scatterers, for checks on the interfaces alone."""
    quiet = {"oct.core_scatter_per_um2": 0.0, "oct.coating_scatter_per_um2": 0.0}
    return cfg.with_overrides({**quiet, **overrides})


def fwhm_px(profile: np.ndarray) -> float:
    """Full width at half maximum of the main peak, with linear interpolation."""
    peak = int(np.argmax(profile))
    half = profile[peak] / 2.0
    left = peak
    while profile[left] > half:
        left -= 1
    right = peak
    while profile[right] > half:
        right += 1
    lo = left + (half - profile[left]) / (profile[left + 1] - profile[left])
    hi = right - (half - profile[right]) / (profile[right - 1] - profile[right])
    return hi - lo


def reflector_profile(oc, z_um=250.0, mismatch=(0.0, 0.0), correction=(0.0, 0.0)):
    spec = physics.spectrometer(oc)
    phase = physics.dispersion_phase(spec, spec.k, mismatch)
    fringe = spec.source * np.cos(2.0 * spec.k * z_um + phase)
    analytic = hilbert(to_wavenumber(fringe[None, :], spec), axis=-1)
    return np.sqrt(depth_profile(analytic, spec, oc, correction)[0]), spec, analytic


# --- physics --------------------------------------------------------------------


def test_axial_resolution_formula(cfg):
    assert physics.axial_resolution_um(cfg.oct.center_nm, cfg.oct.fwhm_nm) == pytest.approx(
        1.5, abs=0.05)


def test_single_reflector_peak_width_matches_theory(cfg):
    theory = physics.axial_resolution_um(cfg.oct.center_nm, cfg.oct.fwhm_nm)
    source_limited = cfg.with_overrides({"oct.window": "none"}).oct
    profile, spec, _ = reflector_profile(source_limited)
    assert fwhm_px(profile) * spec.depth_px_um == pytest.approx(theory, rel=0.10)
    assert np.argmax(profile) * spec.depth_px_um == pytest.approx(250.0, abs=spec.depth_px_um)
    # The Hann window trades some resolution for lower side lobes.
    hann, _, _ = reflector_profile(cfg.oct)
    assert theory < fwhm_px(hann) * spec.depth_px_um < 1.45 * theory


def test_fresnel_and_refraction():
    assert physics.fresnel_amplitude(1.0, 1.48) ** 2 == pytest.approx(0.0375, abs=0.001)
    geo = physics.sphere_geometry(np.array([0.0, 150.0, 400.0]), 366.0, 16.0, 1.48)
    assert geo.path_um[0] == pytest.approx(16.0)
    assert geo.path_um[1] > 16.0  # the refracted ray crosses the coating obliquely
    assert not geo.inside[2]
    assert geo.theta[1] == pytest.approx(np.arcsin(150 / 366))


def test_motion_washout_grows_with_speed_and_angle(cfg):
    spec = physics.spectrometer(cfg.oct)
    theta = np.radians([0.0, 10.0, 30.0])
    slow = physics.motion_washout(theta, 0.05, cfg.oct, spec.k0)
    fast = physics.motion_washout(theta, 0.5, cfg.oct, spec.k0)
    assert slow[0] == fast[0] == 1.0
    assert fast[2] < fast[1] < 1.0
    assert fast[2] < slow[2]


# --- processing chain -----------------------------------------------------------


def test_dispersion_compensation_restores_the_peak(cfg):
    oc = cfg.oct
    sharp, spec, _ = reflector_profile(oc)
    blurred, _, analytic = reflector_profile(oc, mismatch=oc.dispersion_rad)
    assert fwhm_px(blurred) > 1.2 * fwhm_px(sharp)
    assert blurred.max() < 0.92 * sharp.max()
    found = estimate_dispersion(analytic, spec, oc)
    assert found == pytest.approx(oc.dispersion_rad, abs=0.3)
    restored, _, _ = reflector_profile(oc, mismatch=oc.dispersion_rad, correction=found)
    assert fwhm_px(restored) == pytest.approx(fwhm_px(sharp), rel=0.10)
    assert restored.max() == pytest.approx(sharp.max(), rel=0.03)


def test_calibration_finds_the_instrument_dispersion(cfg):
    assert calibrated_dispersion(cfg) == pytest.approx(cfg.oct.dispersion_rad, abs=0.4)


@pytest.mark.parametrize("thickness, n_coat", [(4.5, 1.48), (16.0, 1.40), (30.0, 1.45)])
def test_flat_coating_peak_separation_is_n_times_thickness(cfg, thickness, n_coat):
    c = clean(cfg)
    p = default_params(c, thickness_um=thickness, n_coat=n_coat, radius_um=FLAT, snr_db=45.0,
                       speed_m_s=0.5)
    raw = simulate(p, c)
    out = process(raw.spectra, raw.background, raw.x_um, c.oct, dispersion=c.oct.dispersion_rad)
    col = out.db[c.oct.out_ascans // 2]
    peaks, props = find_peaks(col, prominence=8.0)
    first, second = sorted(peaks[np.argsort(props["prominences"])[-2:]])
    expected = n_coat * thickness / out.depth_px_um
    assert second - first == pytest.approx(expected, abs=1.0)
    lab = labels(p, c, out.x_um, out.crop_start_px)
    mid = c.oct.out_ascans // 2
    assert lab["outer_px"][mid] == pytest.approx(first, abs=1.0)
    assert lab["inner_px"][mid] == pytest.approx(second, abs=1.0)


@pytest.mark.parametrize("snr", [20.0, 35.0])
def test_output_snr_matches_the_request(cfg, snr):
    c = clean(cfg)
    p = default_params(c, radius_um=FLAT, snr_db=snr, speed_m_s=0.5)
    raw = simulate(p, c)
    out = process(raw.spectra, raw.background, raw.x_um, c.oct, dispersion=c.oct.dispersion_rad,
                  stages=True)
    power = out.stages["power"]
    first = int((c.oct.reflector_um + 30.0) / out.depth_px_um)
    measured = 10.0 * np.log10(power[:, first:].max(axis=1).mean() / out.noise_floor)
    assert measured == pytest.approx(snr, abs=2.0)


def test_processed_scan_has_the_stored_shape(cfg):
    p = default_params(cfg)
    raw = simulate(p, cfg)
    out = process(raw.spectra, raw.background, raw.x_um, cfg.oct,
                  dispersion=calibrated_dispersion(cfg))
    assert out.image.shape == (cfg.oct.out_ascans, cfg.oct.depth_pixels) == (128, 512)
    assert out.image.dtype == np.uint8
    assert out.x_um.size == 128
    # The apex found in the data sits one crop margin below the top of the image.
    lab = labels(p, cfg, out.x_um, out.crop_start_px)
    assert np.nanmin(lab["outer_px"]) == pytest.approx(cfg.oct.crop_margin_px, abs=10)


# --- generator ------------------------------------------------------------------


def test_simulation_is_deterministic_per_seed(cfg):
    p = default_params(cfg, seed=3)
    assert np.array_equal(simulate(p, cfg).spectra, simulate(p, cfg).spectra)
    other = simulate(default_params(cfg, seed=4), cfg).spectra
    assert not np.array_equal(simulate(p, cfg).spectra, other)


def test_ascan_spacing_follows_pellet_speed(cfg):
    oc = cfg.oct
    fast, scale_fast = scan_axis(default_params(cfg, speed_m_s=0.5), oc)
    slow, scale_slow = scan_axis(default_params(cfg, speed_m_s=0.05), oc)
    assert fast.size == int(np.ceil(oc.field_um / 2.0))  # 0.5 m/s at 250 kHz = 2 um apart
    assert scale_fast == 1.0
    assert slow.size == oc.max_raw_ascans and scale_slow < 1.0


def _processed(cfg, **params):
    p = default_params(cfg, **params)
    raw = simulate(p, cfg)
    out = process(raw.spectra, raw.background, raw.x_um, cfg.oct,
                  dispersion=cfg.oct.dispersion_rad, stages=True)
    return p, out, labels(p, cfg, out.x_um, out.crop_start_px)


def _inner_contrast(out, lab, cols=slice(56, 72)):
    """Mean dB at the labelled coating-core interface minus the mean inside the coating."""
    at_interface, inside = [], []
    for c in range(cols.start, cols.stop):
        o, i = int(round(lab["outer_px"][c])), int(round(lab["inner_px"][c]))
        at_interface.append(out.db[c, i - 1 : i + 2].max())
        inside.append(out.db[c, o + 12 : i - 6].mean())
    return float(np.mean(at_interface) - np.mean(inside))


def test_pigment_hides_the_coating_core_interface(cfg):
    _, clear, lab_clear = _processed(cfg, pigment=0.0, snr_db=40.0)
    _, opaque, lab_opaque = _processed(cfg, pigment=1.0, snr_db=40.0)
    assert _inner_contrast(clear, lab_clear) > 8.0
    assert _inner_contrast(opaque, lab_opaque) < 4.0


def test_core_shows_speckle_and_the_space_above_does_not(cfg):
    _, out, lab = _processed(cfg, snr_db=40.0)
    core = np.concatenate([out.db[c, int(lab["inner_px"][c]) + 8 : int(lab["inner_px"][c]) + 60]
                           for c in range(54, 74)])
    above = out.db[54:74, :20].ravel()
    assert core.mean() > above.mean() + 6.0


def test_fouling_dims_the_reference_reflector_and_the_pellet(cfg):
    def reflector_db(out):
        row = int(cfg.oct.reflector_um / out.depth_px_um)
        return out.stages["db_full"][:, row - 3 : row + 4].max(axis=1).mean()

    _, clean_scan, _ = _processed(cfg, fouling=0.0)
    _, fouled, _ = _processed(cfg, fouling=0.8)
    assert reflector_db(fouled) < reflector_db(clean_scan) - 6.0
    assert fouled.db.max() < clean_scan.db.max() - 6.0
    haze_rows = slice(int(cfg.oct.window_face_um / fouled.depth_px_um) + 8,
                      int((cfg.oct.window_face_um + cfg.oct.haze_thickness_um)
                          / fouled.depth_px_um) - 8)
    assert (fouled.stages["db_full"][:, haze_rows].mean()
            > clean_scan.stages["db_full"][:, haze_rows].mean() + 3.0)


def test_labels_come_from_parameters_and_match_the_image(cfg):
    p, out, lab = _processed(cfg, snr_db=40.0)
    assert set(np.unique(lab["mask"])) == {ABOVE, COATING, CORE}
    valid = np.flatnonzero(lab["valid"])
    assert 20 < valid.size < cfg.oct.out_ascans
    # Classes are ordered top to bottom in every column.
    assert (np.diff(lab["mask"].astype(int), axis=1) >= 0).all()
    brightest = out.db[valid].argmax(axis=1)
    assert np.median(np.abs(brightest - lab["outer_px"][valid])) <= 1.5
    # Along the normal at the apex the optical thickness is n x thickness.
    mid = cfg.oct.out_ascans // 2
    optical = (lab["inner_px"][mid] - lab["outer_px"][mid]) * lab["depth_px_um"]
    assert optical == pytest.approx(p.n_coat * p.thickness_um, rel=0.01)
    # Fewer columns are usable when the signal is weak.
    _, _, weak = _processed(cfg, snr_db=15.0)
    assert weak["valid"].sum() < lab["valid"].sum()


# --- dataset --------------------------------------------------------------------


def test_dataset_parameters_follow_the_configured_ranges(cfg):
    from coatshield.oct.dataset import sample_params

    dc = cfg.oct_dataset
    ps = [sample_params(cfg, "train", i) for i in range(400)]
    thickness = np.array([p.thickness_um for p in ps])
    assert dc.thickness_um[0] <= thickness.min() and thickness.max() <= dc.thickness_um[1]
    assert np.median(thickness) == pytest.approx(np.sqrt(2 * 40), rel=0.2)  # log-uniform
    assert all(dc.n_coat[0] <= p.n_coat <= dc.n_coat[1] for p in ps)
    assert all(dc.radius_um[0] <= p.radius_um <= dc.radius_um[1] for p in ps)
    pigment = np.array([p.pigment for p in ps])
    assert np.mean(pigment >= dc.pigment_high[0]) == pytest.approx(dc.pigment_high_share, abs=0.06)
    assert np.median([p.fouling for p in ps]) < 0.2  # fouling mostly low
    assert sample_params(cfg, "train", 7) == sample_params(cfg, "train", 7)


def test_splits_never_share_a_seed(cfg):
    from coatshield.oct.dataset import SPLITS, sample_params

    dc = cfg.oct_dataset
    sizes = {"train": dc.n_train, "val": dc.n_val, "locked": dc.n_locked}
    ranges = sorted((dc.seed_offsets[s], dc.seed_offsets[s] + sizes[s]) for s in SPLITS)
    assert all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:], strict=False))
    assert sample_params(cfg, "train", 0).seed != sample_params(cfg, "val", 0).seed


def test_make_scan_stores_image_and_parameter_derived_labels(cfg):
    from coatshield.oct.dataset import make_scan, masks_from_rows

    scan = make_scan(cfg, "val", 3)
    assert scan["image"].shape == (128, 512) and scan["image"].dtype == np.uint8
    assert scan["outer_px"].shape == scan["inner_px"].shape == scan["valid"].shape == (128,)
    assert np.array_equal(scan["image"], make_scan(cfg, "val", 3)["image"])
    mask = masks_from_rows(scan["outer_px"], scan["inner_px"], 512)
    assert mask.shape == (128, 512) and set(np.unique(mask)) <= {ABOVE, COATING, CORE}
    inside = np.isfinite(scan["outer_px"])
    assert (scan["inner_px"][inside] > scan["outer_px"][inside]).all()
    assert (mask[~inside] == ABOVE).all()


def test_only_the_two_owner_scripts_mention_the_locked_test_set():
    """Training and product code must never reference data/test_locked."""
    from coatshield.config import REPO_ROOT

    allowed = {"make_locked_testset.py", "run_locked_eval.py"}
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for folder in ("coatshield", "scripts", "app", "notebooks")
        for path in (REPO_ROOT / folder).rglob("*")
        if path.suffix in (".py", ".ipynb") and path.name not in allowed
        and "test_locked" in path.read_text(errors="ignore")
    ]
    assert offenders == []
