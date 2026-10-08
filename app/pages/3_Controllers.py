"""Demo step 3: four stopping rules on the same batch."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from components import common, sidebar, story

from coatshield import style
from coatshield.estimate.controllers import CONTROLLERS

common.page("Controllers")
story.banner()
cfg = sidebar.render()
bundle, precomputed = common.get_bundle(cfg)
common.require_estimates(bundle, cfg)

st.title("Four stopping rules, one batch")
truth, est, ctrl = bundle.truth, bundle.est, bundle.controllers.set_index("controller")
spec = cfg.spec.d10_min_um
stops = bundle.meta["stop_steps"]

cols = st.columns(len(CONTROLLERS))
for col, (name, label) in zip(cols, CONTROLLERS.items(), strict=True):
    r = ctrl.loc[name]
    col.metric(f"{name} · {label}", f"{r.true_below_spec_pct:.0f} %")
    col.caption("truly below spec · "
                + (f"stops at {r.stop_h:.2f} h" if r.stopped else "does not stop"))

t = truth.t_h.to_numpy()
fig = go.Figure()
if est.d10_lo.notna().any():
    band = est.d10_lo.notna().to_numpy()
    fig.add_trace(go.Scatter(x=np.r_[t[band], t[band][::-1]],
                             y=np.r_[est.d10_hi[band], est.d10_lo[band][::-1]],
                             fill="toself", fillcolor="rgba(42,120,214,0.18)", line=dict(width=0),
                             name=f"{100 * cfg.estimator.interval:.0f} % interval",
                             hoverinfo="skip"))
hover = "%{y:.2f} µm<extra>%{fullData.name}</extra>"
fig.add_trace(go.Scatter(x=t, y=truth.d10_um, name="True d10", mode="lines",
                         line=dict(color=style.NEUTRAL, width=3), hovertemplate=hover))
fig.add_trace(go.Scatter(x=t, y=est.raw_d10, name="Raw d10", mode="lines",
                         line=dict(color=style.CURVE_COLORS["raw"], width=2),
                         hovertemplate=hover))
fig.add_trace(go.Scatter(x=t, y=est.hybrid_d10, name="Corrected d10", mode="lines",
                         line=dict(color=style.CURVE_COLORS["corrected"], width=2),
                         hovertemplate=hover))
fig.add_hline(y=spec, line=dict(color=style.TEXT, width=1, dash="dash"),
              annotation_text=f"spec {spec:g} µm", annotation_position="bottom right")
stopped = [(name, s) for name, s in stops.items() if s is not None]
fig.add_trace(go.Scatter(
    x=[t[s] for _, s in stopped], y=[truth.d10_um.iloc[s] for _, s in stopped],
    mode="markers+text", text=[name for name, _ in stopped], textposition="bottom center",
    marker=dict(size=11, color=style.TEXT, symbol="diamond",
                line=dict(width=2, color=style.SURFACE)),
    name="Rule stops here (true d10)",
    customdata=[[CONTROLLERS[name]] for name, _ in stopped],
    hovertemplate="%{text} %{customdata[0]}<br>stops at %{x:.2f} h<br>"
                  "true d10 %{y:.2f} µm<extra></extra>"))
common.layout(fig, "d10 of the batch: truth and the two estimates", height=460,
              hovermode="x unified")
late = t >= 0.5 * t[-1]
y_lo = float(np.nanmin(truth.d10_um[late])) - 0.5
y_hi = float(np.nanmax(np.r_[est.raw_d10[late], truth.d10_um[late]])) + 0.5
fig.update_xaxes(title="Batch time (h)", range=[t[late][0], t[-1]])
fig.update_yaxes(title="d10 coating thickness (µm)", range=[y_lo, y_hi])

left, right = st.columns([3, 2])
left.plotly_chart(fig, use_container_width=True)

bars = go.Figure(go.Bar(
    x=ctrl.true_below_spec_pct, y=[f"{n} {CONTROLLERS[n]}" for n in ctrl.index], orientation="h",
    marker=dict(color=[style.CONTROLLER_COLORS[n] for n in ctrl.index], cornerradius=4),
    text=[f"{v:.0f} %" for v in ctrl.true_below_spec_pct], textposition="outside",
    textfont=dict(color=style.TEXT), width=0.4,
    hovertemplate="%{y}<br>%{x:.1f} % truly below spec<extra></extra>"))
bars.add_vline(x=10, line=dict(color=style.TEXT, width=1, dash="dash"),
               annotation_text="10 % = on spec", annotation_position="bottom right")
common.layout(bars, "Pellets truly below spec at the stop", height=460)
bars.update_yaxes(autorange="reversed")
bars.update_xaxes(title="% of pellets", range=[0, max(30.0, 1.25 * ctrl.true_below_spec_pct.max())])
right.plotly_chart(bars, use_container_width=True)

table = ctrl.assign(rule=[f"{n} {CONTROLLERS[n]}" for n in ctrl.index])
table["believed"] = [f"{v:.2f}" if v == v else "not measured" for v in ctrl.believed_d10_um]
st.dataframe(
    table[["rule", "stop_h", "true_below_spec_pct", "true_d10_um", "believed",
           "excess_coating_pct", "agglomerate_pct"]],
    hide_index=True, use_container_width=True,
    column_config={
        "rule": "Stopping rule",
        "stop_h": st.column_config.NumberColumn("Stop time (h)", format="%.2f"),
        "true_below_spec_pct": st.column_config.NumberColumn("True % below spec", format="%.1f"),
        "true_d10_um": st.column_config.NumberColumn("True d10 (µm)", format="%.2f"),
        "believed": st.column_config.TextColumn("d10 it believed (µm)"),
        "excess_coating_pct": st.column_config.NumberColumn(
            "Excess coating (%)", format="%.1f",
            help="Sprayed solids beyond what the spec needed. Negative: stopped too early."),
        "agglomerate_pct": st.column_config.NumberColumn("Agglomerates (%)", format="%.2f"),
    },
)
c2, c3 = ctrl.loc["C2"], ctrl.loc["C3"]
if c2.stopped and c3.stopped and c3.true_below_spec_pct > 0:
    ratio = c2.true_below_spec_pct / c3.true_below_spec_pct
    if abs(c2.true_d10_um - c3.true_d10_um) < 0.4:
        st.success("With these settings the raw and corrected rules agree: they stop at "
                   f"true d10 {c2.true_d10_um:.1f} and {c3.true_d10_um:.1f} µm.")
    elif c2.true_d10_um < c3.true_d10_um:
        st.warning(f"The raw-d10 rule believed d10 = {c2.believed_d10_um:.1f} µm and stopped at "
                   f"{c2.stop_h:.2f} h. The true d10 was {c2.true_d10_um:.1f} µm: "
                   f"{c2.true_below_spec_pct:.0f} % of pellets below spec, {ratio:.1f}× the "
                   f"corrected rule's {c3.true_below_spec_pct:.0f} %.")
    else:
        st.info("With these settings the raw-d10 rule stops later than the corrected rule: "
                "measurement noise widens the raw sample, so its d10 reads low.")
if not c3.stopped:
    st.info("CoatShield did not stop in the simulated spray time. It holds while the batch has "
            "not reached spec with 95 % confidence or while too many scans are undecided.")
common.footer(bundle, precomputed)
