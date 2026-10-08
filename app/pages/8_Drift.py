"""Demo step 7: window fouling, the undecided state and the drift alarm."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from components import common, sidebar, story

from coatshield import style
from coatshield.compliance import drift

common.page("Drift and fouling")
story.banner()
cfg = sidebar.render()

st.title("Drift and fouling")
st.write("As the window fouls, the signal fades. The monitor compares every scan with a clean "
         "baseline and raises an alarm; scans the chain is unsure about turn undecided instead "
         "of sending a wrong thickness to the controller.")

path = common.ASSETS / "drift.npz"
if not path.exists():
    st.info("Run `python scripts/build_gallery.py` to build the scans for this page.")
    st.stop()


@st.cache_data(show_spinner=False)
def load(file: str) -> dict:
    data = np.load(file)
    return {k: data[k] for k in data.files}


d = load(str(path))
ramp = ~d["baseline"]
baseline = drift.fit_baseline(d["features"][d["baseline"]], cfg)

hours = st.slider("Hours for the window to foul completely", 1.0, 8.0, 4.0, step=0.5,
                  help="A slower ramp gives more scans per fouling level; the scans are real "
                       "chain results replayed in order of fouling.")
levels = np.unique(d["level"][ramp])
per_level = max(1, int(round(hours / 4.0 * 12)))
order = np.concatenate([np.flatnonzero(ramp & (d["level"] == lv))[:per_level] for lv in levels])
out = drift.monitor(d["features"][order], d["undecided"][order], baseline, cfg)
fouling = d["level"][order]
t = np.linspace(0, hours, order.size)
error = np.abs(d["error_um"][order])
accepted = ~d["undecided"][order]

states = list(out["state"])


def first(state):
    return t[states.index(state)] if state in states else None


bad = accepted & (error > cfg.chain.max_error_um)
first_bad = t[np.argmax(bad)] if bad.any() else None
c1, c2, c3, c4 = st.columns(4)
c1.metric("State at the end", drift.worst(states[-10:]))
c2.metric("First warning", "none" if first(drift.WARNING) is None
          else f"{first(drift.WARNING):.2f} h")
c3.metric("First alarm", "none" if first(drift.ALARM) is None else f"{first(drift.ALARM):.2f} h")
c4.metric(f"First accepted reading off by > {cfg.chain.max_error_um:g} µm",
          "none" if first_bad is None else f"{first_bad:.2f} h")
alarm, warned = first(drift.ALARM), first(drift.WARNING)
if first_bad is None:
    st.success("No accepted reading was off by more than "
               f"{cfg.chain.max_error_um:g} µm at any fouling level: unsure scans turned "
               "undecided first.")
elif alarm is not None and alarm < first_bad:
    st.success(f"The alarm fired at {alarm:.2f} h, before the first badly wrong accepted "
               f"reading at {first_bad:.2f} h.")
else:
    st.warning(f"An accepted reading was off by more than {cfg.chain.max_error_um:g} µm at "
               f"{first_bad:.2f} h, before the alarm. This is reported as found.")


def chart(y, title, limit=None, limit_text="", color=style.BLUE, log=False):
    fig = go.Figure(go.Scatter(x=t, y=y, mode="lines", line=dict(color=color, width=2),
                               hovertemplate="%{x:.2f} h<br>%{y:.2f}<extra></extra>"))
    if limit is not None:
        fig.add_hline(y=limit, line=dict(color=style.TEXT, width=1, dash="dash"),
                      annotation_text=limit_text, annotation_position="top left")
    for when, label in ((warned, "warning"), (alarm, "alarm")):
        if when is not None:
            fig.add_vline(x=when, line=dict(color=style.CRITICAL if label == "alarm"
                                            else style.YELLOW, width=1.5),
                          annotation_text=label, annotation_position="bottom right")
    common.layout(fig, title, height=260, showlegend=False)
    fig.update_xaxes(title="Time since fouling began (h)")
    if log:
        fig.update_yaxes(type="log")
    return fig


a, b = st.columns(2)
a.plotly_chart(chart(fouling, "Window fouling (0 clean, 1 fully fouled)", color=style.NEUTRAL),
               use_container_width=True)
b.plotly_chart(chart(out["undecided_rate"], "Undecided share of scans (moving average)",
                     cfg.drift.undecided_alarm, "alarm level"), use_container_width=True)
a.plotly_chart(chart(np.maximum(out["t2"], 0.1), "Distance from the clean baseline (Hotelling T²)",
                     baseline.t2_limit, "99 % control limit", log=True),
               use_container_width=True)
b.plotly_chart(chart(out["reflector_drop_db"], "Drop of the reference reflector (dB)",
                     cfg.drift.reflector_alarm_db, "alarm level"), use_container_width=True)

st.caption(f"{order.size} scans through the full chain at rising fouling, against a baseline of "
           f"{int(d['baseline'].sum())} clean scans. Features per scan: "
           + ", ".join(f.replace("_", " ") for f in drift.FEATURES) + ".")
problems = drift.out_of_training_range(
    cfg, thickness_um=cfg.batch.target_mean_um, n_coat=cfg.coating.n, n_core=cfg.core.n,
    radius_um=cfg.pellet.core_median_um / 2 + cfg.batch.target_mean_um)
if problems:
    st.warning("This product is outside the ranges the boundary finder was trained on: "
               + "; ".join(problems) + ".")
else:
    st.caption("Product check: thickness, indices and size are inside the ranges the boundary "
               "finder was trained on.")
common.footer()
