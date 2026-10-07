"""Faults and diagnosis: an injected fault, what the signals show, and the recommended action."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from components import common, sidebar, story

from coatshield import style
from coatshield.estimate.diagnosis import RULES

common.page("Faults")
story.banner()
cfg = sidebar.render()
bundle, precomputed = common.get_bundle(cfg)

st.title("Faults and diagnosis")
scenario = cfg.fault.scenario
if scenario == "none":
    st.info("No fault is injected. Choose a fault scenario in the sidebar to see how it shows "
            "up; this page then names it from the signals alone.")
else:
    st.write(f"Injected fault: **{sidebar.LABELS[scenario]}**"
             + ("" if scenario == "substrate_shift" else f", from {cfg.fault.start_h:g} h")
             + ". The diagnosis below uses only what the window can see.")

sig = bundle.signals
t = sig.t_h.to_numpy()
last = sig.iloc[-1]
found = [r for r in RULES if bool(last[r.fault])]
if found:
    for rule in found:
        first = float(t[np.argmax(sig[rule.fault].to_numpy())])
        st.warning(f"**{rule.label}** diagnosed (first flagged at {first:.2f} h). "
                 f"Signature: {rule.signature.lower()}.  \n**Recommended action:** {rule.action}")
else:
    st.success("No fault diagnosed at the end of the batch.")
if scenario not in ("none", *[r.fault for r in found]):
    st.info(f"The injected fault ({sidebar.LABELS[scenario]}) is not diagnosed at the end of "
               "this batch. Maldistribution in particular changes the size-adjusted spread by "
               "only about a quarter, which is hard to see through 1 µm of measurement noise.")

st.subheader("Diagnosis matrix over the batch")
z = np.array([sig[r.fault].to_numpy().astype(float) for r in RULES])
fig = go.Figure(go.Heatmap(
    z=z, x=t, y=[r.label for r in RULES], zmin=0, zmax=1, showscale=False, ygap=3,
    colorscale=[[0, style.GRID], [1, style.CRITICAL]],
    hovertemplate="%{y}<br>%{x:.2f} h<extra></extra>"))
if scenario not in ("none", "substrate_shift"):
    fig.add_vline(x=cfg.fault.start_h, line=dict(color=style.TEXT, width=1, dash="dash"),
                  annotation_text="fault starts", annotation_position="top")
common.layout(fig, "Red: the rule's signature is present", height=320)
fig.update_xaxes(title="Batch time (h)", showgrid=False)
fig.update_yaxes(autorange="reversed", showgrid=False)
st.plotly_chart(fig, use_container_width=True)
st.caption(f"No verdict is given in the first {2 * cfg.diagnosis.baseline_h:g} h, while the "
           "baselines are being set.")

panels = (("growth_ratio", "Growth rate against the recipe", cfg.diagnosis.growth_low_factor,
           "low below"),
          ("agglomerate_pct", "Gate twin share (%)", cfg.diagnosis.agglomerate_high_pct,
           "high above"),
          ("fines_ratio", "Fines rate against the first hour", cfg.diagnosis.fines_high_factor,
           "high above"),
          ("undecided_rate", "Undecided share of scans", cfg.diagnosis.undecided_high,
           "high above"))
cols = st.columns(2)
for i, (key, title, limit, word) in enumerate(panels):
    fig = go.Figure(go.Scatter(x=t, y=sig[key], mode="lines",
                               line=dict(color=style.BLUE, width=2), name=title,
                               hovertemplate="%{x:.2f} h<br>%{y:.2f}<extra></extra>"))
    fig.add_hline(y=limit, line=dict(color=style.TEXT, width=1, dash="dash"),
                  annotation_text=f"{word} {limit:g}", annotation_position="top left")
    common.layout(fig, title, height=260, showlegend=False)
    fig.update_xaxes(title="Batch time (h)")
    cols[i % 2].plotly_chart(fig, use_container_width=True)

st.subheader("Coating variability against time")
truth = bundle.truth
keep = truth.t_h >= 0.25
fig = go.Figure()
fig.add_trace(go.Scatter(x=truth.t_h[keep], y=100 * truth.cv_within_size[keep], mode="lines",
                         name="True spread at equal size", line=dict(color=style.BLUE, width=2),
                         hovertemplate="%{x:.2f} h<br>%{y:.2f} %<extra></extra>"))
anchor = truth[keep].iloc[0]
law = 100 * anchor.cv_within_size * np.sqrt(anchor.t_h / truth.t_h[keep])
fig.add_trace(go.Scatter(x=truth.t_h[keep], y=law, mode="lines", name="√(1/t) law",
                         line=dict(color=style.NEUTRAL, width=2, dash="dash"),
                         hoverinfo="skip"))
common.layout(fig, "Pellet-to-pellet spread at equal size falls as √(1/t) in a healthy batch",
              height=340)
fig.update_xaxes(title="Batch time (h)", type="log")
fig.update_yaxes(title="Spread (CV, %)", type="log")
st.plotly_chart(fig, use_container_width=True)
st.caption("Simulation truth, shown for reference: a line above the dashed law means the bed "
           "is not mixing evenly. The total spread of the batch is larger and levels off, "
           "because bigger pellets coat faster.")
common.footer(bundle, precomputed)
