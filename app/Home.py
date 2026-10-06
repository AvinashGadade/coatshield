"""CoatShield dashboard: home and story mode."""

import streamlit as st
from components import common, sidebar, story

common.page("Home")
story.follow_redirect()
sidebar.render()

st.title("CoatShield")
st.subheader("A raw in-line sample overstates coating quality. Correct the size bias and "
             "stop on d10, not the mean.")
st.write(
    "The measurement window of a Wurster coater sees big pellets more often, and big pellets "
    "coat faster. A stop rule that trusts the raw sample therefore stops early. CoatShield "
    "weights each measured pellet by how under-represented its size is, and stops only when "
    "the thinnest tenth of the whole batch meets the spec."
)

left, right = st.columns([1, 2], vertical_alignment="center")
left.button("Start the demo", type="primary", on_click=story.start, use_container_width=True)
right.write(f"Story mode walks through {len(story.STEPS)} screens. Each **Next step** sets the "
            "sliders and opens the right page; nothing else needs touching.")

st.divider()
st.markdown("##### The demo script")
for i, step in enumerate(story.STEPS, 1):
    st.markdown(f"{i}. **{step.title}** (demo step {step.demo_step}). {step.say}")

st.divider()
st.page_link("pages/1_Batch.py", label="Batch: the bed and the window")
st.page_link("pages/2_Distributions.py", label="Distributions: truth, raw sample, corrected")
st.page_link("pages/3_Controllers.py", label="Controllers: four stopping rules on one batch")
st.page_link("pages/4_Sensitivity.py", label="Sensitivity: what the claim rests on")
common.footer()
