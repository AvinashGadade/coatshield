"""Demo step 6: the error from assuming n = 1.5, and three ways to find n instead."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from components import common, sidebar, story

from coatshield import style
from coatshield.oct import physics
from coatshield.oct.preprocess import sensitivity_db
from coatshield.solve.index import assumed_index_error, index_from_reflectance

common.page("Refractive index")
story.banner()
cfg = sidebar.render()

st.title("Thickness without assuming a refractive index")
st.write("OCT measures optical path: thickness times refractive index. Assume the wrong index "
         "and every thickness is wrong by the same factor.")

lo, hi = cfg.oct_dataset.n_coat
left, right = st.columns([2, 3])
with left:
    n_true = st.slider("True refractive index of the coating", lo, hi, cfg.coating.n, step=0.01)
    assumed = cfg.solve.assumed_n
    err = assumed_index_error(n_true, assumed)
    target = cfg.batch.target_mean_um
    st.metric(f"Thickness error from assuming n = {assumed:g}", f"{100 * err:+.1f} %")
    st.caption(f"A {target:g} µm coat would read {target * (1 + err):.1f} µm.")
with right:
    grid = np.linspace(lo, hi, 61)
    fig = go.Figure(go.Scatter(x=grid, y=100 * (grid / assumed - 1), mode="lines",
                               line=dict(color=style.BLUE, width=2), name="thickness error",
                               hovertemplate="n = %{x:.2f}<br>%{y:+.1f} %<extra></extra>"))
    fig.add_trace(go.Scatter(x=[n_true], y=[100 * err], mode="markers", showlegend=False,
                             marker=dict(size=12, color=style.TEXT,
                                         line=dict(width=2, color=style.SURFACE)),
                             hoverinfo="skip"))
    fig.add_hline(y=0, line=dict(color=style.TEXT_SECONDARY, width=1))
    common.layout(fig, f"Thickness error when n = {assumed:g} is assumed", height=300,
                  showlegend=False)
    fig.update_xaxes(title="True refractive index")
    fig.update_yaxes(title="Thickness error (%)")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Three independent ways to find n")
st.markdown(
    "- **A · Reflectance.** The air-coating reflection, calibrated with a small reflector on "
    "the window, inverted through the Fresnel equation. Precise per pellet, but only as "
    "accurate as the calibration.\n"
    "- **B · Camera-OCT fusion.** Mean optical thickness from OCT over mean physical thickness "
    "from the growth of the camera's mean diameter. No reflectance calibration at all.\n"
    "- **C · At-line anchor.** A one-time comparison with microscopy cross-sections of about "
    f"{cfg.solve.anchor_pellets} pellets."
)

path = common.latest_report("solver_index_*.csv")
if path is None:
    st.info("Run `python scripts/validate_solver.py` to fill this section.")
else:
    table = pd.read_csv(path)
    series = (("A_pooled_err", "A reflectance, pooled", style.BLUE),
              ("B_fusion_err", "B camera-OCT fusion", style.ORANGE),
              ("C_anchor_err", "C at-line anchor", style.AQUA))
    fig = go.Figure()
    for key, label, color in series:
        fig.add_trace(go.Scatter(x=table.snr, y=table[key].abs(), name=label,
                                 mode="lines+markers", line=dict(color=color, width=2),
                                 marker=dict(size=8),
                                 hovertemplate="SNR %{x:g} dB<br>|error| %{y:.4f}"
                                               f"<extra>{label}</extra>"))
    fig.add_hline(y=0.01, line=dict(color=style.TEXT, width=1, dash="dash"),
                  annotation_text="target 0.01", annotation_position="top left")
    common.layout(fig, "Error in the pooled refractive index on synthetic scans (true n known)",
                  height=380)
    fig.update_xaxes(title="Surface signal-to-noise ratio (dB)")
    fig.update_yaxes(title="|error| in n", type="log")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"From {path.name}. Surfaces are the simulation's own plus 1 pixel of noise; the "
               "trained boundary finder is not in this result yet.")

st.subheader("How much method A depends on its calibration")
cal = st.slider("Error in the reflector calibration (%)", -15, 15, 0, step=1)
oc = cfg.oct
depth = 0.5 * sum(oc.standoff_um)
surface_db = (20 * np.log10(abs(physics.fresnel_amplitude(1.0, n_true)))
              + sensitivity_db(oc, depth))
reflector_db = 10 * np.log10(oc.reflector_reflectance) + sensitivity_db(oc, oc.reflector_um)
n_read = index_from_reflectance(float(surface_db), float(reflector_db), depth, cfg, cal / 100.0)
c1, c2, c3 = st.columns(3)
c1.metric("True n", f"{n_true:.3f}")
c2.metric("Method A reads", f"{n_read:.3f}")
c3.metric("Thickness error that causes", f"{100 * (n_true / n_read - 1):+.1f} %")
st.caption("That is why methods B and C, which need no reflectance calibration, are built "
           "alongside it and the three are compared.")
common.footer()
