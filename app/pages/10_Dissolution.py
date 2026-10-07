"""Illustrative dissolution projection from the corrected thickness distribution."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from components import common, sidebar, story

from coatshield import style
from coatshield.estimate import dissolution

common.page("Dissolution")
story.banner()
cfg = sidebar.render()
bundle, precomputed = common.get_bundle(cfg)

st.title("What the thickness distribution could mean for release")
st.warning("Illustrative only. This is a straight line through three published data points "
           "for coated pellets, not a dissolution model, and it says nothing about this product.")

frames = bundle.frame_steps
valid = [i for i in range(len(frames)) if bundle.dist_corrected[i].sum() > 0]
hours = {round(float(bundle.truth.t_h.iloc[frames[i]]), 2): i for i in valid}
stop = bundle.meta["stop_steps"].get("C3")
default = min(hours, key=lambda h: abs(hours[h] - (np.searchsorted(frames, stop)
                                                   if stop is not None else valid[-1])))
t_h = st.select_slider("Batch time (h)", options=list(hours), value=default)
j = hours[t_h]

lo, hi = dissolution.supported_range(cfg)
window = st.slider("Thickness window shown (µm)", 0.0, float(bundle.thickness_edges[-1]),
                   (max(lo - 4.0, 0.0), hi + 4.0), step=0.5)
proj = dissolution.project(bundle.thickness_edges, bundle.dist_corrected[j], cfg)
centres = 0.5 * (bundle.thickness_edges[:-1] + bundle.thickness_edges[1:])
shown = (centres >= window[0]) & (centres <= window[1])
slope, intercept = dissolution.fit_line(cfg)

c1, c2, c3 = st.columns(3)
mean_t63 = float(np.sum(proj["share"] * proj["t63_min"]))
c1.metric("Mean release time t63", f"{mean_t63:.0f} min")
c2.metric("Slope of the published data", f"{slope:.1f} min per µm")
c3.metric("Share of pellets outside the data", f"{100 * proj['share_outside_range']:.0f} %")

fig = go.Figure()
for inside, color, name in ((True, style.BLUE, "Inside the published range"),
                            (False, style.NEUTRAL, "Outside it (extrapolated)")):
    sel = shown & (proj["inside"] == inside)
    fig.add_trace(go.Bar(x=proj["t63_min"][sel], y=100 * proj["share"][sel], name=name,
                         marker=dict(color=color), width=slope * (centres[1] - centres[0]) * 0.9,
                         hovertemplate="t63 %{x:.0f} min<br>%{y:.2f} % of pellets"
                                       f"<extra>{name}</extra>"))
common.layout(fig, f"Projected release time of the batch at {t_h:g} h (illustrative)",
              height=400, barmode="overlay")
fig.update_xaxes(title="Time to 63 % release, t63 (min)")
fig.update_yaxes(title="% of pellets")
st.plotly_chart(fig, use_container_width=True)
st.caption(f"Published points: t63 of {', '.join(f'{t:g}' for t in cfg.dissolution.t63_min)} "
           f"min at {', '.join(f'{t:g}' for t in cfg.dissolution.thickness_um)} µm "
           f"(Pharmaceutics 16:1307). Supported thickness range: {lo:g} to {hi:g} µm. "
           "The corrected (not the raw) thickness distribution is used.")
common.footer(bundle, precomputed)
