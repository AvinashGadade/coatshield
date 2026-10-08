"""Demo step 2: true distribution, raw sample and corrected estimate on one axis."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from components import common, sidebar, story

from coatshield import style

common.page("Distributions")
story.banner()
cfg = sidebar.render()
bundle, precomputed = common.get_bundle(cfg)
common.require_estimates(bundle, cfg)

st.title("Truth, raw sample and corrected estimate")
truth, est = bundle.truth, bundle.est
frames = bundle.frame_steps
valid = [int(i) for i, s in enumerate(frames) if bundle.dist_raw[i].sum() > 0]
frame_hours = {round(float(truth.t_h.iloc[frames[i]]), 2): i for i in valid}
stops = bundle.meta["stop_steps"]


def frame_time(step):
    j = min(valid, key=lambda i: abs(int(frames[i]) - step))
    return round(float(truth.t_h.iloc[frames[j]]), 2)


wanted = story.view("time")
if wanted and stops.get(wanted) is not None:
    st.session_state["dist_time"] = frame_time(stops[wanted])
if st.session_state.get("dist_time") not in frame_hours:
    st.session_state["dist_time"] = frame_time(stops.get("C2") or int(frames[valid[-1]]))
t_h = st.select_slider("Batch time (h)", options=list(frame_hours), key="dist_time",
                       help="Scrub through the batch. Opens at the moment the raw-d10 rule stops.")
j = frame_hours[t_h]
step = int(frames[j])
spec = cfg.spec.d10_min_um

row_t, row_e = truth.iloc[step], est.iloc[step]
c1, c2, c3, c4 = st.columns(4)
c1.metric("True d10", f"{row_t.d10_um:.1f} µm")
c2.metric("Raw sample d10", f"{row_e.raw_d10:.1f} µm")
c2.caption(f"{row_e.raw_d10 - row_t.d10_um:+.1f} µm against the truth")
c3.metric("Corrected d10", f"{row_e.hybrid_d10:.1f} µm")
c3.caption(f"{row_e.hybrid_d10 - row_t.d10_um:+.1f} µm against the truth")
c4.metric("Truly below spec", f"{100 * row_t.frac_below_spec:.0f} %")

# The window holds a few thousand pellets, so draw the curves on merged thickness bins.
MERGE = 3
edges = bundle.thickness_edges
n_merged = (edges.size - 1) // MERGE
edges = edges[: n_merged * MERGE + 1 : MERGE]
centres = 0.5 * (edges[:-1] + edges[1:])
width = float(edges[1] - edges[0])
curves = {
    name: dist[j][: n_merged * MERGE].reshape(n_merged, MERGE).sum(axis=1) / width
    for name, dist in (("truth", bundle.dist_truth), ("raw", bundle.dist_raw),
                       ("corrected", bundle.dist_corrected))
}
shown = np.flatnonzero(sum(curves.values()) > 1e-4)
lo, hi = centres[max(shown[0] - 3, 0)], centres[min(shown[-1] + 3, centres.size - 1)]

fig = go.Figure()
hover = "%{x:.1f} µm<extra>%{fullData.name}</extra>"
fig.add_trace(go.Scatter(x=centres, y=curves["truth"], name="True batch", mode="lines",
                         line=dict(color=style.NEUTRAL, width=2, shape="spline"), fill="tozeroy",
                         fillcolor="rgba(138,137,131,0.18)", hovertemplate=hover))
below = centres <= spec
fig.add_trace(go.Scatter(x=centres[below], y=curves["truth"][below], mode="lines",
                         name="True batch below spec", line=dict(color=style.CRITICAL, width=0),
                         fill="tozeroy", fillcolor="rgba(208,59,59,0.45)", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=centres, y=curves["raw"], name="Raw window sample", mode="lines",
                         line=dict(color=style.CURVE_COLORS["raw"], width=2, shape="spline"),
                         hovertemplate=hover))
fig.add_trace(go.Scatter(x=centres, y=curves["corrected"], name="Corrected estimate",
                         mode="lines",
                         line=dict(color=style.CURVE_COLORS["corrected"], width=2, shape="spline"),
                         hovertemplate=hover))
fig.add_vline(x=spec, line=dict(color=style.TEXT, width=1, dash="dash"),
              annotation_text=f"spec: d10 ≥ {spec:g} µm", annotation_position="top left")
common.layout(fig, f"Coating thickness distribution at {t_h:g} h", height=470,
              hovermode="x unified")
fig.update_xaxes(title="Coating thickness (µm)", range=[lo, hi])
fig.update_yaxes(title="Share of pellets per µm", showticklabels=False)
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    f"At {t_h:g} h the raw sample puts d10 at **{row_e.raw_d10:.1f} µm**; the batch's true d10 "
    f"is **{row_t.d10_um:.1f} µm** and the corrected estimate reads "
    f"**{row_e.hybrid_d10:.1f} µm**. The shaded tail is the "
    f"**{100 * row_t.frac_below_spec:.0f} %** of pellets that are truly below spec."
)
with st.expander("Numbers behind the chart"):
    st.dataframe(
        {
            "": ["True batch", "Raw window sample", "IPW", "Model-based", "Corrected (hybrid)"],
            "d10 (µm)": [row_t.d10_um, row_e.raw_d10, row_e.ipw_d10, row_e.model_d10,
                         row_e.hybrid_d10],
            "Median (µm)": [row_t.d50_um, row_e.raw_d50, row_e.ipw_d50, row_e.model_d50,
                            row_e.hybrid_d50],
            "Mean (µm)": [row_t.mean_um, row_e.raw_mean, row_e.ipw_mean, row_e.model_mean,
                          row_e.hybrid_mean],
        },
        hide_index=True, use_container_width=True,
        column_config={c: st.column_config.NumberColumn(format="%.2f")
                       for c in ("d10 (µm)", "Median (µm)", "Mean (µm)")},
    )
    st.caption(f"Window: last {cfg.estimator.window_min:g} min, {int(row_e.n_window):,} accepted "
               f"single pellets. Reference size distribution: "
               f"{sidebar.LABELS[cfg.estimator.reference]}.")
common.footer(bundle, precomputed)
