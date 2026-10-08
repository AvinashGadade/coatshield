"""The Live Console's Python side: the embedded page, signatures, audit view, batch record.

The console itself is a static page in web/console/. Embedded in the dashboard it sends
requests (a signature, a batch record) and this module answers them with the existing
compliance code.
"""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path

import streamlit.components.v1 as components

from coatshield import console_export
from coatshield.compliance import registry
from coatshield.compliance.audit import AuditTrail
from coatshield.compliance.record import build_record, save_pdf
from coatshield.compliance.signature import SignatureError, sign
from coatshield.config import Config
from coatshield.estimate.controllers import CONTROLLERS
from components import common

CONSOLE_DIR = common.REPO_ROOT / "web" / "console"
AUDIT_ROWS = 10  # Brief: the drawer shows the last 10 audit entries

_component = components.declare_component("coatshield_live_console", path=str(CONSOLE_DIR))


def is_built() -> bool:
    return (CONSOLE_DIR / "data" / "scenarios.js").exists()


def render(**args):
    """Draw the console; returns its latest request (or None)."""
    return _component(key="live_console", default=None, **args)


def batch_id(cfg: Config, scenario: str) -> str:
    """Same naming as the Audit page: B- plus the start of the scenario's config hash."""
    return f"B-{console_export.scenario_config(cfg, scenario).analysis_hash()[:8]}"


def handle_sign(request: dict, cfg: Config, trail: AuditTrail) -> dict:
    """Check a signature request from the console and write it to the audit trail."""
    scenario = request.get("scenario", "")
    if scenario not in console_export.SCENARIOS:
        return {"nonce": request.get("nonce"), "ok": False, "problem": "unknown scenario"}
    scen = console_export.scenario_config(cfg, scenario)
    obj = batch_id(cfg, scenario)
    model_hash, config_hash = registry.combined_hash(), scen.analysis_hash()
    stop_h = request.get("stop_h")
    recommended = f"stop at {stop_h:.2f} h" if stop_h is not None else "no stop recommended"
    if not any(e["action"] == "recommendation" and e["object"] == obj for e in trail.entries()):
        trail.append("system", "system", "recommendation", obj, new=recommended,
                     reason=f"{CONTROLLERS['C3']} rule: P(d10 >= spec) > "
                            f"{cfg.controller.p_d10_min:g} (simulated replay)",
                     model_hash=model_hash, config_hash=config_hash)
    meaning = request.get("meaning", "")
    try:
        entry = sign(trail, cfg, request.get("user", ""), request.get("pin", ""), meaning,
                     request.get("reason", ""), obj, old="spraying",
                     new=recommended if meaning == "approve_stop" else None,
                     comment=request.get("comment", ""), model_hash=model_hash,
                     config_hash=config_hash)
    except SignatureError as err:
        return {"nonce": request.get("nonce"), "ok": False, "problem": str(err)}
    return {"nonce": request.get("nonce"), "ok": True, "user": entry["user"], "meaning": meaning,
            "time_utc": entry["time_utc"], "hash": entry["hash"], "seq": entry["seq"]}


def audit_view(trail: AuditTrail) -> dict:
    """The last entries of the trail and its chain check, without signatures' details."""
    entries = trail.entries()
    check = trail.verify_chain()
    keep = ("seq", "time_utc", "user", "action", "object", "hash")
    return {"ok": check.ok, "problem": check.problem, "count": len(entries),
            "entries": [{k: e.get(k) for k in keep} for e in entries[-AUDIT_ROWS:]]}


def registry_view() -> dict:
    """Each registered model with whether its file still matches the registered hash."""
    out = {}
    for name, info in registry.active_versions().items():
        try:
            registry.verified_path(name)
            ok = True
        except registry.ModelIntegrityError:
            ok = False
        out[name] = {"ok": ok, "version": info["version"], "sha256": info["sha256"]}
    return out


def record_pdf(request: dict, cfg: Config, trail: AuditTrail) -> dict:
    """The scenario's batch record as a PDF (base64), from the existing exporter."""
    scenario = request.get("scenario", "")
    if scenario not in console_export.SCENARIOS:
        scenario = next(iter(console_export.SCENARIOS))
    bundle, _ = common.get_bundle(console_export.scenario_config(cfg, scenario))
    obj = batch_id(cfg, scenario)
    record = build_record(obj, bundle, trail, registry.active_versions(),
                          preset=f"Live Console replay: {scenario}")
    with tempfile.TemporaryDirectory() as tmp:
        pdf = save_pdf(record, Path(tmp) / "record.pdf").read_bytes()
    return {"nonce": request.get("nonce"), "name": f"{obj}.pdf",
            "pdf_b64": base64.b64encode(pdf).decode()}
