"""Tables and figures from reports/, for questions after the demo."""

import pandas as pd
import streamlit as st
from components import common, sidebar, story

common.page("Validation")
story.banner()
sidebar.render()

st.title("Validation reports")
st.write("Everything here is read from the `reports/` folder. Each file name carries the hash "
         "of the configuration that produced it, so any number can be traced to a run.")

sections = (
    ("Bias correction and stopping rules", "validation_summary_*.md",
     ("heatmap_raw_d10_m_k_*.png", "controllers_by_scenario_*.png", "gamma_panel_*.png"),
     "python scripts/run_validation.py"),
    ("Refractive index and thickness solver", "solver_summary_*.md",
     ("solver_validation_*.png",), "python scripts/validate_solver.py"),
    ("Agglomerate gate", "gate_comparison_*.md", (), "python scripts/tune_gate.py"),
    ("Boundary finder", "seg_evaluation_*.md", (), "python scripts/evaluate_seg.py"),
)
for title, summary, figures, command in sections:
    st.header(title)
    path = common.latest_report(summary)
    if path is None:
        st.info(f"Not run yet. Command: `{command}`")
        continue
    st.caption(path.name)
    st.markdown(path.read_text())
    for pattern in figures:
        figure = common.latest_report(pattern)
        if figure is not None:
            st.image(str(figure), caption=figure.name)

st.header("Synthetic OCT against real OCT")
csv, png = common.latest_report("oct_realism_*.csv"), common.latest_report("oct_realism_*.png")
if csv is None:
    st.info("Not run yet. Command: `python scripts/check_oct_realism.py`")
else:
    table = pd.read_csv(csv)
    stats = ["skewness", "excess_kurtosis", "grain_axial_px", "grain_lateral_px"]
    st.dataframe(table.groupby("source")[stats].median().round(2), use_container_width=True)
    st.caption(f"{csv.name}: medians per source. Only the shape statistics (skewness, kurtosis) "
               "are comparable across instruments; grain size depends on pixel size.")
    if png is not None:
        st.image(str(png), caption=png.name)
common.footer()
