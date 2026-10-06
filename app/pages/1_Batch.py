"""Demo step 1: the bed, the window and how little of the batch it measures."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from components import common, sidebar, story

from coatshield import style
from coatshield.seeds import rng

common.page("Batch")
story.banner()
cfg = sidebar.render()
bundle, precomputed = common.get_bundle(cfg)

st.title("The batch and the window")
truth = bundle.truth
stops = bundle.meta["stop_steps"]
default_step = stops.get("C3") or len(truth) - 1
t_h = st.slider("Batch time (h)", float(truth.t_h.iloc[0]), float(truth.t_h.iloc[-1]),
                float(truth.t_h.iloc[default_step]), step=0.25)
step = int(np.argmin(np.abs(truth.t_h.to_numpy() - t_h)))

n_real = bundle.meta["n_real"]
measured = int(bundle.measured_cum[step])
c1, c2, c3 = st.columns(3)
c1.metric("Pellets in the batch", f"{n_real / 1e6:,.0f} million")
c2.metric("Pellets measured so far", f"{measured:,}")
c3.metric("Share of the batch measured", f"{100 * measured / n_real:.2f} %")


@st.cache_data(show_spinner=False)
def bed_animation(n: int, n_frames: int, m: float, sigma: float, seed: int) -> go.Figure:
    """Schematic Wurster bed: pellets rise through the draft tube and fall in the annulus."""
    gen = rng("app.bed", seed)
    size = np.exp(sigma * gen.standard_normal(n))  # relative to the median
    phase0 = gen.random(n)
    lane = gen.random(n)
    side = np.where(gen.random(n) < 0.5, -1.0, 1.0)
    # The window favours big pellets: chance of lighting up ~ size^m.
    chance = size**m
    chance = 0.9 * chance / chance.max()
    lit_draw = gen.random((n_frames, n))

    def positions(phase):
        up = phase < 0.18  # fast rise in the tube
        arc = (phase >= 0.18) & (phase < 0.32)  # fountain above the tube
        u = np.where(up, phase / 0.18, np.where(arc, (phase - 0.18) / 0.14, (phase - 0.32) / 0.68))
        x = np.where(up, side * 0.10 * lane,
                     np.where(arc, side * (0.10 * lane + u * (0.22 + 0.5 * lane)),
                              side * (0.32 + 0.5 * lane)))
        y = np.where(up, 0.12 + 1.15 * u,
                     np.where(arc, 1.27 + 0.35 * np.sin(np.pi * u) * (0.4 + 0.6 * lane),
                              1.27 - 1.15 * u))
        return x, y

    window = dict(x0=0.84, x1=0.9, y0=0.55, y1=0.8)

    def frame_traces(i):
        x, y = positions((phase0 + i / n_frames) % 1.0)
        near = (x > 0.62) & (y > window["y0"]) & (y < window["y1"])
        lit = near & (lit_draw[i] < chance)
        return [
            go.Scattergl(x=x[~lit], y=y[~lit], mode="markers", hoverinfo="skip",
                         marker=dict(size=4 * size[~lit], color=style.NEUTRAL, opacity=0.45),
                         name="Pellets"),
            go.Scattergl(x=x[lit], y=y[lit], mode="markers", hoverinfo="skip",
                         marker=dict(size=6 * size[lit], color=style.BLUE,
                                     line=dict(width=1, color=style.SURFACE)),
                         name="Measured by the window"),
        ]

    fig = go.Figure(data=frame_traces(0),
                    frames=[go.Frame(data=frame_traces(i), name=str(i)) for i in range(n_frames)])
    outline = dict(color=style.TEXT_SECONDARY, width=2)
    fig.add_shape(type="path", line=outline,
                  path="M -0.55 0.05 L -0.9 0.45 L -0.9 1.9 L 0.9 1.9 L 0.9 0.45 L 0.55 0.05 Z")
    for x in (-0.2, 0.2):  # draft tube
        fig.add_shape(type="line", x0=x, x1=x, y0=0.18, y1=1.2, line=outline)
    fig.add_shape(type="rect", **window, fillcolor=style.BLUE, opacity=0.9, line_width=0)
    fig.add_annotation(x=0.95, y=0.675, text="window", showarrow=False, xanchor="left",
                       font=dict(color=style.TEXT, size=13))
    fig.add_annotation(x=0, y=0.0, text="spray nozzle", showarrow=False, yanchor="top",
                       font=dict(color=style.TEXT_SECONDARY, size=12))
    common.layout(fig, height=560)
    fig.update_xaxes(visible=False, range=[-1.25, 1.45], fixedrange=True)
    fig.update_yaxes(visible=False, range=[-0.12, 2.0], scaleanchor="x", fixedrange=True)
    play = dict(frame=dict(duration=90, redraw=True), fromcurrent=True, transition=dict(duration=0))
    fig.update_layout(updatemenus=[dict(
        type="buttons", showactive=False, x=0.0, y=0.0, xanchor="left", yanchor="bottom",
        buttons=[dict(label="▶ Play", method="animate", args=[None, play]),
                 dict(label="❚❚ Pause", method="animate",
                      args=[[None], dict(mode="immediate", frame=dict(duration=0))])])])
    return fig


left, right = st.columns([3, 2])
with left:
    fig = bed_animation(cfg.app.animation_pellets, cfg.app.animation_frames,
                        cfg.window.size_bias_m, cfg.pellet.size_sigma_log, cfg.seed)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Schematic of {cfg.app.animation_pellets:,} pellets; positions are illustrative, "
               "the counts above come from the simulated batch.")
with right:
    row = truth.iloc[step]
    st.markdown("##### What the whole bed truly looks like now")
    a, b = st.columns(2)
    a.metric("Mean coating", f"{row.mean_um:.1f} µm")
    b.metric("d10 (thinnest tenth)", f"{row.d10_um:.1f} µm")
    a.metric("Below spec", f"{100 * row.frac_below_spec:.0f} %")
    b.metric("Weight gain", f"{row.weight_gain_pct:.1f} %")
    st.markdown(
        f"The window records about **{cfg.window.objects_per_min:.0f} objects a minute** out of "
        f"**{n_real / 1e6:,.0f} million** pellets. With window bias m = "
        f"{cfg.window.size_bias_m:g}, a pellet 20 % larger than the median is "
        f"**{1.2 ** cfg.window.size_bias_m:.1f}×** as likely to be measured, and large pellets "
        f"carry thicker coats."
    )
common.footer(bundle, precomputed)
