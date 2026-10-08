"""CoatShield dashboard: home and story mode."""

import json

import streamlit as st
import streamlit.components.v1 as components
from components import common, process, sidebar, story

common.page("Home")
story.follow_redirect()
sidebar.render()

st.title("CoatShield")
st.subheader("A raw in-line sample overstates coating quality. Correct the size bias and "
             "stop on d10, not the mean.")

go, text = st.columns([1, 3], vertical_alignment="center")
go.button("▶  Start the demo", type="primary", on_click=story.start, use_container_width=True)
text.write(f"Story mode walks through {len(story.STEPS)} screens. Each **Next step** sets the "
           "sliders and opens the right page; nothing else needs touching.")

st.markdown("##### How it works, from the coater to the decision")
components.html(process.process_html(), height=265, scrolling=False)

# Headline numbers, read from the final report so they always match it.
numbers_path = common.REPORTS / "final" / "final_numbers.json"
if numbers_path.exists():
    numbers = json.loads(numbers_path.read_text())
    head = numbers.get("headline_chain_error_model") or numbers.get("headline_placeholder")
    if head:
        raw = head["rules"]["C2"]["true_below_spec_pct"]
        ours = head["rules"]["C3"]["true_below_spec_pct"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Raw rule: pellets below spec", f"{raw:.0f} %")
        c1.caption("it believes 10 %")
        c2.metric("CoatShield rule: below spec", f"{ours:.0f} %")
        c2.caption("on the same batches")
        c3.metric("Out-of-spec pellets, raw vs corrected", f"{head['raw_over_coatshield']:.1f}×")
        c3.caption("2,140 simulated batches")
        c4.metric("Error of the corrected d10", f"{head['hybrid_d10_mae_um_mean']:.2f} µm")
        c4.caption(f"raw sample: {head['raw_d10_mae_um_mean']:.2f} µm")

st.write(
    "The measurement window of a Wurster coater sees big pellets more often, and big pellets "
    "coat faster. A stop rule that trusts the raw sample therefore stops early. CoatShield "
    "weights each measured pellet by how under-represented its size is, and stops only when "
    "the thinnest tenth of the whole batch meets the spec."
)

with st.expander("The demo script"):
    for i, step in enumerate(story.STEPS, 1):
        st.markdown(f"{i}. **{step.title}** (demo step {step.demo_step}). {step.say}")

st.markdown("##### All pages")
links = (
    ("pages/00_Live_Console.py", "Live Console: one batch replayed as an operator sees it"),
    ("pages/1_Batch.py", "Batch: the bed and the window"),
    ("pages/2_Distributions.py", "Distributions: truth, raw sample, corrected"),
    ("pages/3_Controllers.py", "Controllers: four stopping rules on one batch"),
    ("pages/4_Sensitivity.py", "Sensitivity: what the claim rests on"),
    ("pages/5_Pellet_inspector.py", "Pellet inspector: one pellet through the chain"),
    ("pages/6_Refractive_index.py", "Refractive index: thickness without assuming n"),
    ("pages/7_Gate.py", "Gate: fused pellets are held back"),
    ("pages/8_Drift.py", "Drift and fouling"),
    ("pages/9_Faults.py", "Faults and diagnosis"),
    ("pages/10_Dissolution.py", "Dissolution (illustrative)"),
    ("pages/11_Audit.py", "Audit trail, sign-off and batch record"),
    ("pages/12_Validation.py", "Validation reports"),
    ("pages/13_Impact.py", "Impact calculator"),
    ("pages/14_Limits.py", "Limits and sources"),
)
columns = st.columns(3)
for i, (path, label) in enumerate(links):
    columns[i % 3].page_link(path, label=label)
common.footer()
