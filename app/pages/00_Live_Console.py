"""Live Console: a simulated batch replayed the way an in-line operator screen shows it."""

import streamlit as st
from components import common, console, story

from coatshield.compliance.audit import AuditTrail
from coatshield.config import load_config

common.page("Live Console")
story.banner()
cfg = load_config()
trail = AuditTrail(common.REPO_ROOT / "app" / ".runtime" / "audit.jsonl")

st.title("Live Console")
st.caption("Simulated replay of one batch: per-pellet readings, the gate, the size-bias "
           "correction, the d10 stop rule and the operator's signature. No claim of real-pellet "
           "accuracy.")
if not console.is_built():
    st.info("Run `make console-data` (or `python scripts/make_console_data.py`) to build the "
            "replay data for this page.")
    common.footer()
    st.stop()

state = st.session_state
request = console.render(
    users=[u.name for u in cfg.compliance.users], meanings=list(cfg.compliance.meanings),
    reasons=list(cfg.compliance.reason_codes), audit=console.audit_view(trail),
    registry=console.registry_view(), sign_result=state.get("console_sign"),
    record=state.get("console_record"))

# The console posts one request at a time; each carries a nonce so it is answered once.
if request:
    for kind, handler, slot in (("sign", console.handle_sign, "console_sign"),
                                ("record", console.record_pdf, "console_record")):
        body = request.get(kind)
        if body and body.get("nonce") != (state.get(slot) or {}).get("nonce"):
            state[slot] = handler(body, cfg, trail)
            st.rerun()

st.caption("Signatures use the demonstration accounts (PINs in the README) and are written to "
           "the same audit trail as the Audit page. The console also opens on its own, without "
           "the dashboard: `web/console/index.html`.")
common.footer()
