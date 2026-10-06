"""Batch record export: one JSON document and a PDF rendering of it."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from coatshield.bundle import Bundle
from coatshield.compliance.audit import AuditTrail
from coatshield.estimate.controllers import CONTROLLERS

NOTE = "Demonstration on simulated data. Not a validated GMP system."


def build_record(batch_id: str, bundle: Bundle, trail: AuditTrail, models: dict[str, dict],
                 preset: str = "", drift_alarms: list[dict] | None = None) -> dict:
    """Everything a reviewer needs about one batch, traceable to hashes."""
    cfg = bundle.meta["config"]
    method = cfg["estimator"]["method"]
    stops = bundle.meta["stop_steps"]
    step = stops.get("C3")
    at = bundle.est.iloc[step if step is not None else -1]
    truth_t = bundle.truth.t_h
    entries = trail.entries()
    check = trail.verify_chain()
    faults = [c for c in bundle.signals.columns
              if bundle.signals[c].dtype == bool and c != "ready" and bundle.signals[c].any()]
    return {
        "note": NOTE,
        "batch_id": batch_id,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "preset": preset,
        "scenario": cfg["fault"]["scenario"],
        "config_hash": bundle.meta["config_hash"],
        "models": models,
        "timeline": {
            "duration_h": float(truth_t.iloc[-1]),
            "stops_h": {name: (float(truth_t.iloc[s]) if s is not None else None)
                        for name, s in stops.items()},
        },
        "recommendation": {
            "rule": f"C3 {CONTROLLERS['C3']}",
            "recommended_stop_h": float(truth_t.iloc[step]) if step is not None else None,
            "status": "stop recommended" if step is not None else "no stop recommended",
        },
        "final_distribution": {
            "estimator": method,
            "at_h": float(at.t_h),
            "mean_um": float(at[f"{method}_mean"]),
            "d10_um": float(at[f"{method}_d10"]),
            "d50_um": float(at[f"{method}_d50"]),
            "d90_um": float(at[f"{method}_d90"]),
            "d10_interval_um": [float(at.d10_lo), float(at.d10_hi)],
            "p_d10_meets_spec": float(at.p_spec),
            "spec_d10_min_um": cfg["spec"]["d10_min_um"],
        },
        "undecided_rate": float(at.undecided_rate),
        "diagnosed_faults": faults,
        "drift_alarms": drift_alarms or [],
        "signed_actions": [e for e in entries if e.get("signature")],
        "audit": {"entries": len(entries), "chain_ok": check.ok,
                  "latest_hash": trail.latest_hash()},
    }


def save_json(record: dict, path: Path) -> Path:
    Path(path).write_text(json.dumps(record, indent=1, default=str))
    return Path(path)


def save_pdf(record: dict, path: Path) -> Path:
    """Render the record as a short PDF (fpdf2, core fonts, no network)."""
    from fpdf import FPDF

    def clean(text) -> str:  # core fonts are Latin-1
        return str(text).encode("latin-1", "replace").decode("latin-1")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, clean(f"Batch record {record['batch_id']}"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "I", 9)
    pdf.cell(0, 6, clean(record["note"]), new_x="LMARGIN", new_y="NEXT")

    def section(title: str) -> None:
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, clean(title), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)

    def line(label: str, value) -> None:
        pdf.multi_cell(0, 5.5, clean(f"{label}: {value}"), new_x="LMARGIN", new_y="NEXT")

    def number(value, digits=2) -> str:
        return "n/a" if value is None or value != value else f"{value:.{digits}f}"

    section("Batch")
    line("Created (UTC)", record["created_utc"])
    line("Preset", record["preset"] or "default")
    line("Scenario", record["scenario"])
    line("Config hash", record["config_hash"])
    section("Models in force")
    if not record["models"]:
        line("Models", "none registered")
    for name, m in record["models"].items():
        line(name, f"version {m['version']}, sha256 {m['sha256']}, status {m['status']}")
    section("Timeline and recommendation")
    line("Simulated spray time (h)", number(record["timeline"]["duration_h"]))
    for name, hours in record["timeline"]["stops_h"].items():
        line(f"{name} {CONTROLLERS[name]} would stop at (h)", number(hours))
    rec = record["recommendation"]
    line("Recommendation",
         f"{rec['status']} ({rec['rule']}, at {number(rec['recommended_stop_h'])} h)")
    section("Final distribution")
    d = record["final_distribution"]
    line("Estimator", f"{d['estimator']}, at {number(d['at_h'])} h")
    line("Mean / d10 / d50 / d90 (um)", " / ".join(number(d[k]) for k in
                                                   ("mean_um", "d10_um", "d50_um", "d90_um")))
    line("d10 interval (um)", " to ".join(number(v) for v in d["d10_interval_um"]))
    line("P(d10 meets spec)", f"{number(d['p_d10_meets_spec'])} (spec d10 >= "
         f"{d['spec_d10_min_um']} um)")
    line("Undecided rate", number(record["undecided_rate"], 3))
    line("Diagnosed faults", ", ".join(record["diagnosed_faults"]) or "none")
    line("Drift alarms", len(record["drift_alarms"]))
    section("Signed operator actions")
    if not record["signed_actions"]:
        line("Actions", "none")
    for e in record["signed_actions"]:
        line(f"#{e['seq']} {e['time_utc']}",
             f"{e['signature']['signed_by']} ({e['signature']['role']}) - "
             f"{e['signature']['meaning']} - {e['object']} - reason: {e['reason']}")
    section("Audit trail")
    line("Entries", record["audit"]["entries"])
    line("Chain check", "passed" if record["audit"]["chain_ok"] else "FAILED")
    line("Latest hash", record["audit"]["latest_hash"])
    pdf.output(str(path))
    return Path(path)
