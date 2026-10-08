"""Demo step 4: the m x k heatmap from the validation report, and where the correction fails."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from components import common, sidebar, story

from coatshield import style
from coatshield.config import load_config

common.page("Sensitivity")
story.banner()
cfg = sidebar.render()

st.title("What the claim rests on")
ERROR_MODEL = "models/error_model.json"
base = load_config()
tag = base.analysis_hash()
realistic = None
if (common.REPO_ROOT / ERROR_MODEL).exists():
    from coatshield.chain.error_model import config_with_error_model

    realistic_tag = config_with_error_model(base, ERROR_MODEL).analysis_hash()
    if (common.REPORTS / f"validation_cells_{realistic_tag}.csv").exists():
        realistic = realistic_tag
if realistic:
    choice = st.radio("Measurement error used in the validation runs",
                      ("Fitted from the full measurement chain", "Placeholder (1 µm noise)"),
                      horizontal=True)
    if choice.startswith("Fitted"):
        tag = realistic
path = common.REPORTS / f"validation_cells_{tag}.csv"
if not path.exists():
    quick = common.REPORTS / f"validation_cells_{tag}-quick.csv"
    if quick.exists():
        path, tag = quick, f"{tag}-quick"
    else:
        st.error("No validation report for this configuration. "
                 "Run `python scripts/run_validation.py` first.")
        st.stop()


@st.cache_data(show_spinner=False)
def load(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


cells = load(str(path))
gammas = sorted(cells.gamma.unique())
hidden = st.toggle(f"Add hidden selection (γ = {gammas[-1]:g}): the window also favours pellets "
                   "the camera cannot tell apart", value=False)
part = cells[cells.gamma == (gammas[-1] if hidden else gammas[0])]
raw = part.pivot(index="k", columns="m", values="C2_true_below_spec_pct").sort_index()
ours = part.pivot(index="k", columns="m", values="C3_true_below_spec_pct").sort_index()


def heatmap(grid: pd.DataFrame, title: str, zmax: float) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=grid.values, x=[f"{m:g}" for m in grid.columns], y=[f"{k:g}" for k in grid.index],
        colorscale=[[i / (len(style.SEQUENTIAL) - 1), c] for i, c in enumerate(style.SEQUENTIAL)],
        zmin=0, zmax=zmax, xgap=2, ygap=2, text=grid.round(0).values, texttemplate="%{text:.0f} %",
        colorbar=dict(title="% below spec", thickness=12),
        hovertemplate="window bias m = %{x}<br>size-growth k = %{y}<br>"
                      "%{z:.1f} % truly below spec<extra></extra>"))
    common.layout(fig, title, height=400)
    fig.update_xaxes(title="Window size bias m (0 = none)", type="category", showgrid=False)
    fig.update_yaxes(title="Size-growth exponent k", type="category", showgrid=False)
    return fig


zmax = max(30.0, float(raw.values.max()), float(ours.values.max()))
left, right = st.columns(2)
left.plotly_chart(heatmap(raw, "Raw-d10 rule: pellets truly below spec at its stop", zmax),
                  use_container_width=True)
right.plotly_chart(heatmap(ours, "CoatShield rule: pellets truly below spec at its stop", zmax),
                   use_container_width=True)
st.caption("Both rules aim to stop with 10 % of pellets below spec. Each cell is the mean over "
           "the validation seeds.")

m_here, k_here = cfg.window.size_bias_m, cfg.wurster.size_growth_exponent_k
if m_here in raw.columns and k_here in raw.index:
    st.markdown(
        f"At the sidebar's settings (m = {m_here:g}, k = {k_here:g}) the raw-d10 rule ships "
        f"**{raw.loc[k_here, m_here]:.0f} %** below spec and the CoatShield rule "
        f"**{ours.loc[k_here, m_here]:.0f} %**."
    )
st.markdown(
    "- With **no window bias (m = 0)** or **no size-dependent growth (k = 0)** the raw rule does "
    "not ship extra out-of-spec pellets. The claim needs both.\n"
    "- The window bias has never been measured on a real coater. Measuring it is the first "
    "question for the plant visit."
)
if hidden:
    bias = part.hybrid_d10_bias
    st.warning(
        "With hidden selection the correction degrades: the corrected d10 reads "
        f"{bias.mean():+.2f} µm off on average (worst cell {bias.loc[bias.abs().idxmax()]:+.2f} "
        f"µm) and the CoatShield rule ships up to {ours.values.max():.0f} % below spec. "
        "The correction can only undo selection by something the camera sees."
    )
with st.expander("Estimator error across the grid"):
    cols = ["raw_d10_mae", "ipw_d10_mae", "model_d10_mae", "hybrid_d10_mae"]
    st.dataframe(
        part[["m", "k", *cols]].rename(columns={c: c.split("_")[0] for c in cols}),
        hide_index=True, use_container_width=True,
        column_config={c.split("_")[0]: st.column_config.NumberColumn(
            f"{c.split('_')[0]} |d10 error| (µm)", format="%.2f") for c in cols},
    )
st.caption(f"{common.NOTE} · validation report {tag}, read from reports/ (not recomputed).")
