"""Demo step 9: the audit trail, the operator's signature and the batch record."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from components import common, sidebar, story

from coatshield.compliance import registry
from coatshield.compliance.audit import AuditTrail
from coatshield.compliance.record import build_record, save_pdf
from coatshield.compliance.signature import SignatureError, sign
from coatshield.estimate.controllers import CONTROLLERS

common.page("Audit")
story.banner()
cfg = sidebar.render()
bundle, precomputed = common.get_bundle(cfg)

RUNTIME = common.REPO_ROOT / "app" / ".runtime"
trail = AuditTrail(RUNTIME / "audit.jsonl")
batch_id = f"B-{bundle.meta['config_hash'][:8]}"
model_hash = registry.combined_hash()
config_hash = bundle.meta["config_hash"]

st.title("Audit trail and operator sign-off")
st.write("The system recommends; an operator decides and signs. Every recommendation and every "
         "decision is written to an append-only log in which each entry carries the hash of the "
         "one before it.")

# The recommendation for this batch is logged once, by the system.
stop = bundle.meta["stop_steps"].get("C3")
recommended = (f"stop at {bundle.truth.t_h.iloc[stop]:.2f} h" if stop is not None
               else "no stop recommended")
already = any(e["action"] == "recommendation" and e["object"] == batch_id
              for e in trail.entries())
if not already:
    trail.append("system", "system", "recommendation", batch_id, new=recommended,
                 reason=f"{CONTROLLERS['C3']} rule: P(d10 >= spec) > {cfg.controller.p_d10_min:g}",
                 model_hash=model_hash, config_hash=config_hash)

left, right = st.columns([3, 2])
with right:
    st.subheader("Sign a decision")
    st.caption(f"Batch {batch_id} · recommendation: **{recommended}**")
    with st.form("signature", clear_on_submit=True):
        user = st.selectbox("User", [u.name for u in cfg.compliance.users])
        pin = st.text_input("Re-enter PIN", type="password")
        meaning = st.selectbox("Meaning of the signature", cfg.compliance.meanings,
                               format_func=lambda m: m.replace("_", " "))
        reason = st.selectbox("Reason code", cfg.compliance.reason_codes,
                              format_func=lambda r: r.replace("_", " "))
        comment = st.text_input("Comment (optional)")
        if st.form_submit_button("Sign", type="primary"):
            try:
                entry = sign(trail, cfg, user, pin, meaning, reason, batch_id, old="spraying",
                             new=recommended if meaning == "approve_stop" else None,
                             comment=comment, model_hash=model_hash, config_hash=config_hash)
                st.success(f"Signed by {user} at {entry['time_utc']} "
                           f"({meaning.replace('_', ' ')}).")
            except SignatureError as err:
                st.error(f"Signature refused: {err}. The attempt has been logged.")
    st.caption("Demonstration accounts; PINs are in the README.")

with left:
    st.subheader("Audit log")
    check = trail.verify_chain()
    if check.ok:
        st.success(f"Chain check passed: {check.entries} entries, each linked to the last.")
    else:
        st.error(f"Chain check FAILED at entry {check.first_bad_seq}: {check.problem}.")
    entries = trail.entries()
    table = pd.DataFrame(entries)
    if len(table):
        table["signed"] = ["yes" if s else "" for s in table.signature]
        table["hash"] = table.hash.str[:12]
        st.dataframe(table[["seq", "time_utc", "user", "role", "action", "object", "new",
                            "reason", "signed", "hash"]].iloc[::-1],
                     hide_index=True, use_container_width=True, height=320)
    a, b = st.columns(2)
    if a.button("Show what tampering looks like", disabled=len(entries) < 2,
                help="Edits one stored entry, as an intruder would. The chain check then fails."):
        entries[-2]["reason"] = "edited after the fact"
        trail.path.write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in entries))
        st.rerun()
    if b.button("Start a fresh log"):
        trail.path.unlink(missing_ok=True)
        st.rerun()

st.subheader("Models in force")
versions = registry.active_versions()
if versions:
    st.dataframe(pd.DataFrame(versions).T[["version", "sha256", "status"]],
                 use_container_width=True)
    st.caption("The app checks each file against this hash before loading it and refuses to "
               "run on a mismatch.")
else:
    st.info("No trained model is registered yet (`models/manifest.json` is empty). The gate in "
            "use is the classical, rule-based one.")

st.subheader("Locked test set")
access = common.REPORTS / "locked_access_log.jsonl"
if access.exists():
    st.dataframe(pd.DataFrame([json.loads(line) for line in access.read_text().splitlines()]),
                 hide_index=True, use_container_width=True)
else:
    st.info("The locked test set has not been built or evaluated yet. Only its owner builds it "
            "(`scripts/make_locked_testset.py --confirm`); every evaluation is logged with who, "
            "when, which model hash and how many items.")

st.subheader("Batch record")
record = build_record(batch_id, bundle, trail, versions,
                      preset=f"{st.session_state['product']} / {st.session_state['scale']}")
with tempfile.TemporaryDirectory() as tmp:
    pdf_bytes = save_pdf(record, Path(tmp) / "record.pdf").read_bytes()
c1, c2 = st.columns(2)
c1.download_button("Download batch record (PDF)", pdf_bytes, f"{batch_id}.pdf",
                   "application/pdf", use_container_width=True)
c2.download_button("Download batch record (JSON)", json.dumps(record, indent=1, default=str),
                   f"{batch_id}.json", "application/json", use_container_width=True)
st.caption(f"The record carries the config hash, the model hashes, "
           f"{len(record['signed_actions'])} signed action(s) and the audit chain's latest hash "
           f"{record['audit']['latest_hash'][:12]}…")
common.footer(bundle, precomputed)
