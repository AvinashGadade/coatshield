import hashlib
import json

import pytest

from coatshield.compliance.audit import GENESIS, AuditTrail
from coatshield.compliance.registry import (
    ModelIntegrityError,
    active_versions,
    combined_hash,
    verified_path,
)
from coatshield.compliance.signature import SignatureError, hash_pin, sign
from coatshield.config import load_config

OPERATOR_PIN = "2468"  # demonstration PIN of operator1 (see README)


@pytest.fixture()
def cfg():
    return load_config()


@pytest.fixture()
def trail(tmp_path):
    t = AuditTrail(tmp_path / "audit.jsonl")
    t.append("system", "system", "batch_started", "batch B-001", config_hash="abc")
    t.append("system", "system", "recommendation", "spray stop", new="stop at 9.2 h",
             reason="P(d10 >= spec) > 0.95", model_hash="m1", config_hash="abc")
    t.append("operator1", "operator", "note", "batch B-001", reason="shift handover")
    return t


def rewrite(trail, entries):
    trail.path.write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in entries))


# --- audit trail ----------------------------------------------------------------


def test_chain_verifies_and_links_entries(trail):
    entries = trail.entries()
    assert [e["seq"] for e in entries] == [1, 2, 3]
    assert entries[0]["prev_hash"] == GENESIS
    assert entries[1]["prev_hash"] == entries[0]["hash"]
    assert {"time_utc", "user", "role", "action", "object", "old", "new", "reason",
            "model_hash", "config_hash"} <= set(entries[1])
    check = trail.verify_chain()
    assert check.ok and check.entries == 3
    assert trail.latest_hash() == entries[-1]["hash"]


@pytest.mark.parametrize("field, value", [("reason", "edited"), ("new", "stop at 8.0 h"),
                                          ("user", "someone"), ("time_utc", "2026-01-01")])
def test_editing_any_entry_breaks_the_chain(trail, field, value):
    entries = trail.entries()
    entries[1][field] = value
    rewrite(trail, entries)
    check = trail.verify_chain()
    assert not check.ok and check.first_bad_seq == 2


def test_deleting_or_reordering_entries_breaks_the_chain(trail):
    entries = trail.entries()
    rewrite(trail, [entries[0], entries[2]])
    assert not trail.verify_chain().ok
    rewrite(trail, [entries[1], entries[0], entries[2]])
    assert not trail.verify_chain().ok


def test_truncation_is_caught_against_a_recorded_hash(trail):
    latest = trail.latest_hash()
    entries = trail.entries()
    rewrite(trail, entries[:2])
    assert trail.verify_chain().ok  # the shortened chain is consistent in itself
    assert not trail.verify_chain(expected_latest=latest).ok


def test_recomputing_one_hash_is_not_enough_to_hide_an_edit(trail):
    entries = trail.entries()
    entries[1]["reason"] = "edited"
    body = {k: v for k, v in entries[1].items() if k != "hash"}
    entries[1]["hash"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    rewrite(trail, entries)
    assert not trail.verify_chain().ok  # the next entry still points at the old hash


# --- signature ------------------------------------------------------------------


def test_signed_stop_is_recorded_with_name_time_and_meaning(trail, cfg):
    entry = sign(trail, cfg, "operator1", OPERATOR_PIN, "approve_stop", "target_reached",
                 "spray stop", old="spraying", new="stopped", model_hash="m1",
                 config_hash="abc")
    assert entry["signature"] == {"signed_by": "operator1", "role": "operator",
                                  "meaning": "approve_stop", "reason_code": "target_reached"}
    assert entry["time_utc"] and entry["old"] == "spraying" and entry["new"] == "stopped"
    assert trail.verify_chain().ok
    assert OPERATOR_PIN not in trail.path.read_text()


@pytest.mark.parametrize("user, pin, meaning, reason", [
    ("operator1", "0000", "approve_stop", "target_reached"),
    ("nobody", OPERATOR_PIN, "approve_stop", "target_reached"),
    ("operator1", OPERATOR_PIN, "approve_stop", "because"),
    ("operator1", OPERATOR_PIN, "do_anything", "target_reached"),
])
def test_bad_signature_is_refused_and_logged(trail, cfg, user, pin, meaning, reason):
    with pytest.raises(SignatureError):
        sign(trail, cfg, user, pin, meaning, reason, "spray stop")
    last = trail.entries()[-1]
    assert last["action"] == "signature_refused" and last["signature"] is None
    assert trail.verify_chain().ok


def test_pin_hashes_are_salted_per_user(cfg):
    iterations = cfg.compliance.pin_iterations
    assert hash_pin("operator1", "2468", iterations) != hash_pin("qa1", "2468", iterations)
    assert hash_pin("operator1", OPERATOR_PIN, iterations) == cfg.compliance.users[0].pin_hash


# --- model registry -----------------------------------------------------------


@pytest.fixture()
def models(tmp_path):
    (tmp_path / "unet.onnx").write_bytes(b"model weights v1")
    digest = hashlib.sha256(b"model weights v1").hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"models": {"unet": {"file": "unet.onnx", "version": "0.1.0", "sha256": digest,
                             "status": "locked"}}}))
    return tmp_path


def test_registered_model_loads_and_reports_its_version(models):
    assert verified_path("unet", models).name == "unet.onnx"
    versions = active_versions(models)
    assert versions["unet"]["version"] == "0.1.0" and len(versions["unet"]["short_hash"]) == 12
    assert len(combined_hash(models)) == 64


def test_tampered_model_file_is_refused(models):
    (models / "unet.onnx").write_bytes(b"model weights v1 with a backdoor")
    with pytest.raises(ModelIntegrityError, match="does not match"):
        verified_path("unet", models)


def test_missing_or_unlisted_model_is_refused(models):
    with pytest.raises(ModelIntegrityError, match="not in the manifest"):
        verified_path("gate", models)
    (models / "unet.onnx").unlink()
    with pytest.raises(ModelIntegrityError, match="missing"):
        verified_path("unet", models)
    assert combined_hash(models / "nowhere") == ""


# --- batch record ---------------------------------------------------------------


def test_batch_record_exports_json_and_pdf(trail, cfg, models, tmp_path):
    from coatshield.bundle import build_bundle
    from coatshield.compliance.record import build_record, save_json, save_pdf

    small = cfg.with_overrides({"batch.n_pellets": 20000})
    bundle = build_bundle(small, bootstrap="needed")
    sign(trail, cfg, "operator1", OPERATOR_PIN, "approve_stop", "target_reached", "spray stop",
         old="spraying", new="stopped", config_hash=bundle.meta["config_hash"])
    record = build_record("B-001", bundle, trail, active_versions(models), preset="enteric")
    assert record["config_hash"] == bundle.meta["config_hash"]
    assert record["models"]["unet"]["sha256"]
    assert len(record["signed_actions"]) == 1
    assert record["audit"]["chain_ok"] and record["audit"]["latest_hash"] == trail.latest_hash()
    assert record["final_distribution"]["d10_um"] > 0
    assert record["recommendation"]["recommended_stop_h"] is not None
    loaded = json.loads(save_json(record, tmp_path / "record.json").read_text())
    assert loaded["batch_id"] == "B-001"
    pdf = save_pdf(record, tmp_path / "record.pdf")
    data = pdf.read_bytes()
    assert data[:5] == b"%PDF-" and len(data) > 1500


# --- drift monitor --------------------------------------------------------------


def drift_features(n, gen, reflector_db=22.0, shift=0.0):
    """Feature vectors scattered around a clean operating point."""
    import numpy as np

    from coatshield.compliance.drift import FEATURES

    centre = np.array([6.0, -30.0, 35.0, reflector_db, 0.9, 0.25, 0.95])
    spread = np.array([0.3, 0.2, 1.0, 0.3, 0.05, 0.03, 0.02])
    x = centre + spread * gen.standard_normal((n, len(FEATURES)))
    x[:, 2] -= shift  # falling surface SNR
    return x


def test_clean_scans_stay_ok_and_respect_the_control_limit(cfg):
    import numpy as np

    from coatshield.compliance import drift
    from coatshield.seeds import rng

    gen = rng("test.drift")
    baseline = drift.fit_baseline(drift_features(400, gen), cfg)
    fresh = drift_features(2000, gen)
    t2 = drift.hotelling_t2(fresh, baseline)
    assert (t2 > baseline.t2_limit).mean() == pytest.approx(0.01, abs=0.012)
    out = drift.monitor(fresh, np.zeros(2000, bool), baseline, cfg)
    assert (out["state"] == drift.OK).mean() > 0.98
    with pytest.raises(ValueError):
        drift.fit_baseline(drift_features(10, gen), cfg)


def test_fouling_ramp_raises_warning_then_alarm(cfg):
    import numpy as np

    from coatshield.compliance import drift
    from coatshield.seeds import rng

    gen = rng("test.drift.ramp")
    baseline = drift.fit_baseline(drift_features(400, gen), cfg)
    n = 600
    fouling = np.linspace(0, 1, n)
    x = drift_features(n, gen)
    x[:, drift.FEATURES.index("reflector_db")] -= 12.0 * fouling  # the reflector dims
    x[:, drift.FEATURES.index("surface_snr_db")] -= 12.0 * fouling
    undecided = gen.random(n) < 0.02 + 0.6 * fouling
    out = drift.monitor(x, undecided, baseline, cfg)
    states = list(out["state"])
    first_warning, first_alarm = states.index(drift.WARNING), states.index(drift.ALARM)
    assert states[0] == drift.OK and first_warning < first_alarm < n // 2
    assert out["reflector_drop_db"][-1] > cfg.drift.reflector_alarm_db
    assert drift.worst(states) == drift.ALARM and drift.worst([]) == drift.OK


def test_products_outside_the_training_range_are_flagged(cfg):
    from coatshield.compliance.drift import out_of_training_range

    assert out_of_training_range(cfg, thickness_um=16.0, n_coat=1.48, radius_um=366.0) == []
    problems = out_of_training_range(cfg, thickness_um=60.0, n_coat=1.25)
    assert len(problems) == 2 and "thickness_um" in problems[0]


def test_scan_features_have_the_documented_order(cfg):
    import numpy as np

    from coatshield.compliance.drift import FEATURES, scan_features

    db = np.full((128, 512), 3.0)
    db[:, 100] = 30.0
    core = np.zeros((128, 512), bool)
    core[:, 150:200] = True
    prob = np.full((3, 512, 128), 1 / 3)
    x = scan_features(db, 1e-3, 21.0, prob, core, 0.9)
    assert x.shape == (len(FEATURES),)
    named = dict(zip(FEATURES, x, strict=True))
    assert named["surface_snr_db"] == 30.0 and named["reflector_db"] == 21.0
    assert named["noise_floor_db"] == pytest.approx(-30.0)
    assert named["segmentation_entropy"] == pytest.approx(np.log(3), rel=1e-3)
    assert named["gate_confidence"] == 0.9 and named["core_contrast"] == pytest.approx(0.0)
