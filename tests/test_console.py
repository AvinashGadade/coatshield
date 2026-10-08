"""Live Console: the replay exporter, the scan library, the built data and the Python side."""

import base64
import json
import re

import numpy as np
import pandas as pd
import pytest

from coatshield import console_export as ce
from coatshield.chain.pipeline import LabelSegmenter
from coatshield.compliance import drift, registry
from coatshield.compliance.audit import AuditTrail
from coatshield.config import REPO_ROOT, load_config
from coatshield.estimate.controllers import evaluate_controllers, run_estimators
from coatshield.twin.batch import run_batch

CONSOLE = REPO_ROOT / "web" / "console"
DATA = CONSOLE / "data"
OPERATOR_PIN = "2468"  # demonstration PIN of operator1 (see README)
EVENT_FIELDS = ("t_s", "size_um", "gate_class", "gate_conf", "accepted", "undecided",
                "thickness_um", "n_est", "conf", "scan_id")
TIMELINE_FIELDS = ("t_s", "raw_mean", "raw_d10", "corr_d10", "corr_d10_lo", "corr_d10_hi",
                   "corr_d50", "corr_d90", "p_d10_ge_spec", "cv", "n_accepted",
                   "n_rejected_agglom", "n_undecided", "true_d10")
built = pytest.mark.skipif(not (DATA / "scenarios.js").exists(),
                           reason="run scripts/make_console_data.py first")


@pytest.fixture(scope="module")
def cfg():
    return load_config(overrides={"app.n_pellets": 8000})


@pytest.fixture(scope="module")
def replay(cfg):
    return ce.build_scenario(cfg, "default")


@pytest.fixture(scope="module")
def reference(cfg):
    """The same batch run directly, to check the exported numbers against."""
    scen = ce.scenario_config(cfg, "default")
    result = run_batch(scen, cache=False)
    est = run_estimators(result, scen, bootstrap="all")
    return scen, result, est


def nums(values):
    return np.array([np.nan if v is None else v for v in values], dtype=float)


# --- exporter -------------------------------------------------------------------------------

def test_events_have_every_field_and_a_sensible_count(replay, cfg):
    events = replay["events"]
    assert set(EVENT_FIELDS) <= set(events)
    n = len(events["t_s"])
    assert all(len(events[k]) == n for k in EVENT_FIELDS)
    assert 3000 <= n <= cfg.console.max_events + 1 <= 6001
    assert events["t_s"] == sorted(events["t_s"])
    assert set(events["gate_class"]) <= set(ce.GATE_LABELS)


def test_event_states_are_consistent(replay):
    e = pd.DataFrame(replay["events"])
    assert not (e.accepted & e.undecided).any()
    assert e.thickness_um[e.accepted].notna().all()
    assert e.n_est[e.accepted].notna().all()
    assert e.thickness_um[~e.accepted].isna().all()  # nothing is reported for a held-back object
    assert (e.gate_class[e.accepted | e.undecided] == "single").all()
    assert e.gate_conf.between(0, 1).all()
    assert e.conf[e.accepted].between(0, 1).all()
    assert (e.scan_id == -1).all()  # no scan library was given


def test_timeline_is_the_estimators_output(replay, reference):
    scen, result, est = reference
    tl = replay["timeline"]
    assert set(TIMELINE_FIELDS) <= set(tl)
    method = scen.estimator.method
    for key, column in (("raw_mean", est.raw_mean), ("raw_d10", est.raw_d10),
                        ("corr_d10", est[f"{method}_d10"]), ("corr_d10_lo", est.d10_lo),
                        ("corr_d10_hi", est.d10_hi), ("corr_d50", est[f"{method}_d50"]),
                        ("corr_d90", est[f"{method}_d90"]), ("p_d10_ge_spec", est.p_spec),
                        ("cv", est[f"{method}_cv"])):
        np.testing.assert_allclose(nums(tl[key]), column.to_numpy(), atol=6e-5, equal_nan=True)
    np.testing.assert_allclose(nums(tl["true_d10"]), result.truth.d10_um, atol=6e-5)
    np.testing.assert_allclose(tl["t_s"], result.truth.t_s)


def test_timeline_counts_add_up(replay, reference):
    _, result, _ = reference
    tl, s = replay["timeline"], result.samples
    for key in ("n_accepted", "n_rejected_agglom", "n_undecided", "n_fines", "n_held_back"):
        assert np.all(np.diff(tl[key]) >= 0)
    assert tl["n_accepted"][-1] == int(s.accepted.sum())
    total = sum(tl[k][-1] for k in ("n_accepted", "n_rejected_agglom", "n_undecided", "n_fines",
                                    "n_held_back"))
    assert total == len(s)
    n = nums(tl["n_pooled"])
    assert abs(np.nanmean(n) - replay["meta"]["true_n"]) < 0.03


def test_interval_brackets_the_estimate(replay):
    tl = replay["timeline"]
    d10, lo, hi = nums(tl["corr_d10"]), nums(tl["corr_d10_lo"]), nums(tl["corr_d10_hi"])
    ok = np.isfinite(lo) & np.isfinite(d10)
    assert ok.sum() > 100
    assert np.mean((lo[ok] <= d10[ok] + 1e-3) & (d10[ok] <= hi[ok] + 1e-3)) > 0.95


def test_stops_match_the_controllers(replay, reference):
    scen, result, est = reference
    outcome = evaluate_controllers(result, est, scen)
    stops = {s["rule"]: s for s in replay["stops"]}
    assert set(stops) == {"gravimetric", "raw_mean", "raw_d10", "coatshield"}
    for code, name in ce.RULE_NAMES.items():
        assert stops[name]["true_below_spec_pct"] == round(
            float(outcome.loc[code].true_below_spec_pct), 2)
    tl, spec = replay["timeline"], replay["meta"]["spec_d10_min_um"]
    cs = stops["coatshield"]
    assert cs["stopped"]
    step = tl["t_s"].index(cs["t_s"])
    assert tl["p_d10_ge_spec"][step] > replay["meta"]["p_stop"]
    raw = stops["raw_d10"]
    assert tl["raw_d10"][tl["t_s"].index(raw["t_s"])] >= spec
    # The claim, in this one batch: the raw rule stops earlier and ships more below spec.
    assert raw["t_s"] < cs["t_s"]
    assert raw["true_below_spec_pct"] > cs["true_below_spec_pct"]


def test_time_to_spec_projection(replay):
    tl, spec = replay["timeline"], replay["meta"]["spec_d10_min_um"]
    d10, eta = nums(tl["corr_d10"]), nums(tl["eta_to_spec_s"])
    assert np.all(eta[d10 >= spec] == 0)
    rising = np.isfinite(eta) & (d10 < spec)
    assert rising.sum() > 50 and np.all(eta[rising] > 0)
    # Half-way to the stop, the projection is within an hour of what then happened.
    stop = next(s for s in replay["stops"] if s["rule"] == "coatshield")["t_s"]
    i = int(np.searchsorted(tl["t_s"], 0.6 * stop))
    assert abs(tl["t_s"][i] + eta[i] - stop) < 3600


def test_meta_states_what_this_is(replay, cfg):
    meta = replay["meta"]
    assert "Simulated replay" in meta["note"]
    assert meta["model_hash"] == registry.combined_hash()
    assert meta["models"]["unet"]["status"] == "locked"
    assert meta["config_hash"] == ce.scenario_config(cfg, "default").analysis_hash()
    assert "700 um cores" in meta["pellet_assumption"]
    assert 0 < meta["window_share_pct"] < 100
    assert meta["sources"]["headline"] == ce.HEADLINE_SOURCE
    assert (REPO_ROOT / ce.HEADLINE_SOURCE).exists()


def test_same_seed_gives_the_same_replay(cfg, replay):
    again = ce.build_scenario(cfg, "default")
    assert json.dumps(again, sort_keys=True) == json.dumps(replay, sort_keys=True)


def test_another_seed_gives_another_replay(replay):
    other = ce.build_scenario(load_config(overrides={"app.n_pellets": 8000, "seed": 7}),
                              "default")
    assert other["events"]["thickness_um"] != replay["events"]["thickness_um"]


def test_batch_frames(replay):
    batch, spec = replay["batch"], replay["meta"]["spec_d10_min_um"]
    x = np.array(batch["thickness_um"])
    full = [f for f in batch["frames"] if "f_obs" in f]
    assert len(full) > len(batch["frames"]) // 2
    for f in full:
        for key in ("f_obs", "f_true", "raw", "corrected", "truth"):
            assert abs(np.nansum(nums(f[key])) - 1.0) < 2e-3, key
        f_obs, f_true, w = nums(f["f_obs"]), nums(f["f_true"]), nums(f["weight"])
        seen = f_obs > 0.01  # shares are stored to four decimals
        np.testing.assert_allclose(w[seen], f_true[seen] / f_obs[seen], rtol=0.02, atol=0.02)
    # The correction's point: the corrected batch has more below spec than the raw sample.
    mid = full[len(full) // 2]
    below = x <= spec
    assert nums(mid["corrected"])[below].sum() > nums(mid["raw"])[below].sum()


def test_drift_states_follow_the_monitor_limits(cfg):
    dc = cfg.drift
    rates = np.array([0.0, np.nan, dc.undecided_warning, dc.undecided_alarm, 1.0])
    assert ce.drift_states(rates, cfg) == [drift.OK, drift.OK, drift.WARNING, drift.ALARM,
                                           drift.ALARM]


def test_fouling_scenario_turns_scans_undecided_and_never_stops(cfg):
    data = ce.build_scenario(cfg, "window_fouling")
    tl = data["timeline"]
    assert max(tl["fouling"]) > 0.5
    assert max(nums(tl["undecided_rate"])) > data["meta"]["undecided_limit"]
    assert drift.ALARM in tl["drift_state"]
    assert not next(s for s in data["stops"] if s["rule"] == "coatshield")["stopped"]


def test_write_scenario_files(replay, tmp_path):
    ce.write_scenario(replay, tmp_path / "default")
    for key in ("events", "timeline", "stops", "meta", "batch"):
        assert json.loads((tmp_path / "default" / f"{key}.json").read_text()) == replay[key]
    script = (tmp_path / "default" / "data.js").read_text()
    body = script[script.index('["default"]=') + len('["default"]='):].rstrip(";")
    assert json.loads(body) == replay


# --- representative scans -------------------------------------------------------------------

def test_scan_grid_is_about_300_scans(cfg):
    jobs = ce.scan_grid(cfg)
    cc = cfg.console
    assert len(jobs) == (cc.scan_thickness_steps * len(cc.scan_diameters_um)
                         * len(cc.scan_fouling) * cc.scan_repeats)
    assert 250 <= len(jobs) <= 350  # Brief: about 300
    assert [j["scan_id"] for j in jobs] == list(range(len(jobs)))
    assert min(j["thickness_um"] for j in jobs) == pytest.approx(cc.scan_thickness_um[0])


def test_nearest_scan_matches_thickness_size_and_fouling(cfg):
    scans = pd.DataFrame({"scan_id": [10, 11, 12, 13],
                          "thickness_um": [5.0, 15.0, 15.0, 15.0],
                          "diameter_um": [700.0, 700.0, 800.0, 700.0],
                          "fouling": [0.0, 0.0, 0.0, 0.85]})
    measured = np.array([5.5, 14.0, 16.0, np.nan])
    truth = np.array([5.0, 15.0, 15.0, 15.0])
    size = np.array([690.0, 705.0, 810.0, 700.0])
    fouling = np.array([0.0, 0.0, 0.0, 0.9])
    got = ce.nearest_scan(scans, measured, truth, size, fouling, cfg.console.scan_match_weights)
    assert got.tolist() == [10, 11, 12, 13]


def test_draw_scan_marks_both_surfaces(cfg):
    scan = np.full((128, 512), 40, np.uint8)
    outer, inner = np.full(128, 100), np.full(128, 140)
    valid = np.zeros(128, bool)
    valid[20:100] = True
    picture = ce.draw_scan(scan, outer, inner, valid, cfg.console.scan_rows)
    assert picture.shape == (cfg.console.scan_rows, 128 * 3, 3)
    top, bottom = picture[100, 180].astype(int), picture[140, 180].astype(int)
    assert top[2] > top[0] + 60  # red (BGR) on the outer surface
    assert bottom[0] > bottom[2] + 60  # cyan on the inner surface
    assert picture[100, 10].tolist() == [40, 40, 40]  # nothing drawn where there is no signal


def test_make_scan_writes_pictures_and_a_consistent_signal(cfg, tmp_path):
    job = {"scan_id": 3, "thickness_um": 14.0, "diameter_um": 700.0, "fouling": 0.0}
    row = ce.make_scan(job, cfg, LabelSegmenter(cfg), tmp_path)
    assert row["has_scan"] and row["status"] in ("measured", "undecided")
    for name in ("scan_0003.jpg", "scan_0003_camera.jpg", "scan_0003_input.npy"):
        assert (tmp_path / name).stat().st_size > 0
    signal = row["signal"]
    assert len(signal["spectrum"]) == len(signal["background"]) == len(signal["fringes_k"]) == 512
    assert len(signal["ascan_db"]) == cfg.console.scan_rows
    assert cfg.oct.center_nm - 100 < signal["wavelength_nm"][0] < cfg.oct.center_nm
    # The Signal tab's sum: the gap between the two surfaces, divided by n, is the thickness.
    dz = (signal["inner_row"] - signal["outer_row"]) * signal["depth_px_um"]
    assert dz / cfg.coating.n == pytest.approx(job["thickness_um"], abs=1.5)
    assert row["measured_um"] == pytest.approx(job["thickness_um"], abs=1.0)
    again = ce.make_scan(job, cfg, LabelSegmenter(cfg), tmp_path)
    assert again == row


# --- the built data and the page --------------------------------------------------------------

@built
@pytest.mark.parametrize("name", list(ce.SCENARIOS))
def test_built_scenario_is_complete_and_current(name):
    folder = DATA / name
    data = {k: json.loads((folder / f"{k}.json").read_text())
            for k in ("events", "timeline", "stops", "meta", "batch")}
    assert 3000 <= len(data["events"]["t_s"]) <= 6000  # Brief
    assert data["meta"]["model_hash"] == registry.combined_hash()
    assert data["meta"]["config_hash"] == ce.scenario_config(load_config(), name).analysis_hash()
    script = (folder / "data.js").read_text()
    assert json.loads(script[script.index(f'["{name}"]=') + len(name) + 5:].rstrip(";")) == data
    library = json.loads((DATA / "scans" / "index.json").read_text())
    known = {s["scan_id"] for s in library["scans"] if s["has_scan"]}
    used = {i for i in data["events"]["scan_id"] if i >= 0}
    assert used and used <= known
    signals = json.loads((DATA / "scans" / "signals.json").read_text())
    for sid in used:
        assert (DATA / "scans" / f"scan_{sid:04d}.jpg").exists()
        assert (DATA / "scans" / f"scan_{sid:04d}_camera.jpg").exists()
        assert str(sid) in signals


@built
def test_built_replays_tell_the_story_the_console_shows():
    stops = {name: {s["rule"]: s for s in json.loads((DATA / name / "stops.json").read_text())}
             for name in ce.SCENARIOS}
    default = stops["default"]
    assert default["raw_d10"]["true_below_spec_pct"] > 2 * default["coatshield"][
        "true_below_spec_pct"]
    assert default["raw_d10"]["t_s"] < default["coatshield"]["t_s"]
    no_bias = stops["no_bias"]  # without window bias the raw rule is nearly as good
    assert no_bias["raw_d10"]["true_below_spec_pct"] < 0.6 * default["raw_d10"][
        "true_below_spec_pct"]
    assert not stops["window_fouling"]["coatshield"]["stopped"]
    assert stops["spray_drying"]["gravimetric"]["true_below_spec_pct"] > 40
    # The averages quoted in the footer come from the final report: 23.1 %, 9.6 %, 2.4x.
    meta = json.loads((DATA / "default" / "meta.json").read_text())
    assert meta["headline"] == ce.headline()
    assert (meta["headline"]["raw_d10_below_spec_pct"], meta["headline"][
        "coatshield_below_spec_pct"], meta["headline"]["ratio"]) == (23.1, 9.6, 2.4)


@built
def test_scan_library_has_thumbnails_and_index_methods(cfg):
    library = json.loads((DATA / "scans" / "index.json").read_text())
    assert len(library["scans"]) == len(ce.scan_grid(cfg))
    assert library["note"] == "representative simulated scans"
    for name in ce.GATE_LABELS:
        assert (DATA / "scans" / f"camera_{name}.jpg").exists()
    methods = library["index_methods"]
    assert (REPO_ROOT / methods["source"]).exists()
    for key in ("reflectance", "fusion", "anchor"):
        assert abs(methods[key] - methods["true_n"]) < 0.02


def test_page_says_simulated_and_works_offline():
    html = (CONSOLE / "index.html").read_text()
    assert "SIMULATED REPLAY" in html
    assert 'id="fullBtn"' in html
    assert "requestFullscreen" in (CONSOLE / "console_tabs.js").read_text()
    for tab in ("monitoring", "process", "image", "signal", "batch"):
        assert f'data-tab="{tab}"' in html and f'id="tab-{tab}"' in html
    own = [html] + [(CONSOLE / f).read_text() for f in ("console.js", "console_tabs.js",
                                                        "console.css")]
    for text in own:
        assert not re.search(r"https?://", text)  # nothing is fetched from the network
    assert (CONSOLE / "vendor" / "plotly.min.js").exists()
    css, js = own[3].lower(), own[1].lower()
    for colour in ("#e8a33d", "#3b8ed0", "#3fa35b", "#8a63c9", "#c62839"):  # Brief's palette
        assert colour in css and colour in js


# --- the Python side of the embedded page ---------------------------------------------------

@pytest.fixture
def console(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "app"))
    from components import console as module

    return module


@pytest.fixture
def trail(tmp_path):
    return AuditTrail(tmp_path / "audit.jsonl")


def request_for(**changes):
    return {"nonce": "n1", "scenario": "default", "user": "operator1", "pin": OPERATOR_PIN,
            "meaning": "approve_stop", "reason": "target_reached", "comment": "",
            "stop_h": 9.22, **changes}


def test_signing_writes_a_valid_audit_entry(console, trail):
    full = load_config()
    result = console.handle_sign(request_for(), full, trail)
    assert result["ok"] and result["nonce"] == "n1" and result["user"] == "operator1"
    entries = trail.entries()
    assert [e["action"] for e in entries] == ["recommendation", "approve_stop"]
    signed = entries[-1]
    assert signed["hash"] == result["hash"]
    assert signed["signature"]["signed_by"] == "operator1"
    assert signed["object"] == console.batch_id(full, "default")
    assert signed["new"] == "stop at 9.22 h"
    assert signed["model_hash"] == registry.combined_hash()
    assert trail.verify_chain().ok
    assert OPERATOR_PIN not in trail.path.read_text()
    # A second signature for the same batch does not log the recommendation again.
    console.handle_sign(request_for(nonce="n2", user="qa1", pin="1357"), full, trail)
    assert [e["action"] for e in trail.entries()].count("recommendation") == 1
    assert trail.verify_chain().ok


def test_wrong_pin_is_refused_and_logged(console, trail):
    result = console.handle_sign(request_for(pin="0000"), load_config(), trail)
    assert not result["ok"] and "PIN" in result["problem"]
    assert trail.entries()[-1]["action"] == "signature_refused"
    assert trail.verify_chain().ok
    bad = console.handle_sign(request_for(scenario="nope"), load_config(), trail)
    assert not bad["ok"]


def test_audit_view_shows_the_last_ten(console, trail):
    for i in range(14):
        trail.append("system", "system", "note", f"object {i}")
    view = console.audit_view(trail)
    assert view["ok"] and view["count"] == 14
    assert [e["seq"] for e in view["entries"]] == list(range(5, 15))
    assert "signature" not in view["entries"][0]


def test_registry_view_checks_the_model_files(console):
    view = console.registry_view()
    assert view["unet"]["ok"] and view["error_model"]["ok"]
    assert view["unet"]["sha256"] == registry.active_versions()["unet"]["sha256"]


@built
def test_batch_record_pdf_comes_from_the_existing_exporter(console, trail):
    full = load_config()
    console.handle_sign(request_for(), full, trail)
    out = console.record_pdf({"nonce": "r1", "scenario": "default"}, full, trail)
    assert out["nonce"] == "r1" and out["name"] == f"{console.batch_id(full, 'default')}.pdf"
    assert base64.b64decode(out["pdf_b64"]).startswith(b"%PDF")
