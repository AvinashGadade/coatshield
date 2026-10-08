"""Demo step 5: one pellet through the whole measurement chain, every intermediate shown."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from components import common, sidebar, story

from coatshield import style
from coatshield.chain.pipeline import SampledObject, process_object
from coatshield.compliance.registry import ModelIntegrityError, verified_path
from coatshield.config import Config
from coatshield.gate.silhouettes import CLASSES as GATE_CLASSES
from coatshield.seg.infer import Segmenter
from coatshield.twin.population import FINES, SINGLE, TWIN

common.page("Pellet inspector")
story.banner()
cfg = sidebar.render()

st.title("One pellet through the measurement chain")
st.write("Camera image, gate, OCT scan, boundary finder, graph search, solver and confidence. "
         "The scan is simulated from physics; everything after it is the real chain.")

try:
    model_path = str(verified_path("unet"))
except ModelIntegrityError as err:
    st.error(f"The boundary finder cannot be loaded: {err}")
    st.stop()


@st.cache_resource(show_spinner=False)
def segmenter(path: str, config_json: str) -> Segmenter:
    return Segmenter(path, Config.model_validate_json(config_json))


@st.cache_data(show_spinner="Scanning and measuring the pellet…", max_entries=32)
def measure(obj: SampledObject, pooled_n: float, path: str, config_json: str):
    c = Config.model_validate_json(config_json)
    rec = process_object(obj, c, segmenter(path, config_json), pooled_n)
    return rec.summary(), rec.intermediates


kinds = {"Single pellet": SINGLE, "Fused twin": TWIN, "Fines": FINES}
c1, c2, c3, c4, c5 = st.columns(5)
kind = c1.selectbox("Object", list(kinds))
pellet = c2.number_input("Pellet number", 1, 9999, 1, step=1)
snr = c3.slider("Signal-to-noise (dB)", 15.0, 45.0, st.session_state.pop("inspect_snr", 35.0),
                step=1.0)
pigment = c4.slider("Pigment in the coating", 0.0, 1.0, 0.05, step=0.05)
fouling = c5.slider("Window fouling", 0.0, 1.0, st.session_state.pop("inspect_fouling", 0.0),
                    step=0.05)
assume = st.toggle(f"Solver assumes n = {cfg.solve.assumed_n:g} instead of the pooled index "
                   f"({cfg.coating.n:g})", value=False)

thickness = cfg.batch.target_mean_um
diameter = (cfg.pellet.core_median_um + 2 * thickness if kinds[kind] != FINES
            else float(np.mean(cfg.wurster.fines_size_um)))
obj = SampledObject(true_class=kinds[kind], diameter_um=min(diameter, 800.0)
                    if kinds[kind] != TWIN else min(diameter, 520.0),
                    thickness_um=thickness, n_coat=cfg.coating.n, n_core=cfg.core.n,
                    speed_m_s=float(np.mean(cfg.oct_dataset.speed_m_s)), snr_db=snr,
                    fouling=fouling, pigment=pigment, seed=int(pellet))
pooled_n = cfg.solve.assumed_n if assume else cfg.coating.n
row, inter = measure(obj, pooled_n, model_path, cfg.model_dump_json())

status_text = {"measured": "Measured", "undecided": "Undecided: left out and counted",
               "gated": "Held back by the gate", "no_signal": "No usable signal: undecided"}
m1, m2, m3, m4 = st.columns(4)
m1.metric("Outcome", status_text[row["status"]].split(":")[0])
if row["status"] in ("measured", "undecided"):
    m2.metric("Thickness", f"{row['thickness_um']:.1f} µm")
    m2.caption(f"true {thickness:.1f} µm · error {row['error_um']:+.2f} µm")
    m3.metric("Confidence", f"{row['confidence']:.2f}")
    m3.caption(f"undecided below {cfg.chain.confidence_min:g}")
    m4.metric("Index from reflectance", f"{row['n_reflectance']:.3f}")
    m4.caption(f"true {cfg.coating.n:g} · one pellet, method A")
else:
    m2.metric("Thickness", "not measured")
    m3.metric("Gate verdict", row["gate_class"].capitalize())
    m3.caption(f"gate confidence {row['gate_confidence']:.2f}")
st.caption(status_text[row["status"]] + ".")

left, right = st.columns([1, 2])
with left:
    st.markdown("##### 1 · Camera image and gate")
    st.image(inter["camera"], use_container_width=True)
    feats = inter["gate_features"]
    st.markdown(f"Gate verdict: **{row['gate_class']}** (confidence {row['gate_confidence']:.2f})")
    if feats:
        st.caption(f"solidity {feats['solidity']:.3f} · deepest defect {feats['defect_depth']:.3f} "
                   f"of the radius · diameter {feats['equivalent_diameter_um']:.0f} µm. "
                   "The gate in use is the classical shape rule, so its evidence is these "
                   "measures; there is no heat map.")
with right:
    if "scan" not in inter:
        st.info("The gate held this object back, so no OCT scan is taken. Classes: "
                + ", ".join(GATE_CLASSES) + ".")
    else:
        st.markdown("##### 2 · Raw spectrum of the centre A-scan")
        spectrum = inter["spectrum"]
        fig = go.Figure(go.Scatter(y=spectrum, mode="lines",
                                   line=dict(color=style.BLUE, width=1), hoverinfo="skip"))
        common.layout(fig, height=200, showlegend=False)
        fig.update_xaxes(title="Spectrometer pixel (even in wavelength)")
        fig.update_yaxes(title="Counts", showticklabels=False)
        st.plotly_chart(fig, use_container_width=True)

if "scan" in inter:
    scan, valid = inter["scan"], inter["valid"]
    cols = np.arange(scan.shape[0])
    outer = np.where(valid, inter["outer_px"], np.nan)
    inner = np.where(valid, inter["inner_px"], np.nan)
    rows_shown = int(min(scan.shape[1], np.nanmax(inner, initial=120) + 120))
    a, b = st.columns(2)
    with a:
        st.markdown("##### 3 · Processed scan with the two surfaces")
        fig = go.Figure(go.Heatmap(z=scan.T[:rows_shown], colorscale="gray", showscale=False,
                                   hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=cols, y=outer, mode="lines", name="air-coating surface",
                                 line=dict(color=style.BLUE, width=2)))
        fig.add_trace(go.Scatter(x=cols, y=inner, mode="lines", name="coating-core surface",
                                 line=dict(color=style.YELLOW, width=2)))
        common.layout(fig, height=420)
        fig.update_yaxes(autorange="reversed", title="Depth (pixels of 0.36 µm optical path)")
        fig.update_xaxes(title="A-scan")
        st.plotly_chart(fig, use_container_width=True)
    with b:
        st.markdown("##### 4 · Boundary finder: probability of “coating”")
        prob = inter["prob"]
        fig = go.Figure(go.Heatmap(
            z=prob[1][:rows_shown], zmin=0, zmax=1, showscale=True,
            colorscale=[[i / (len(style.SEQUENTIAL) - 1), c]
                        for i, c in enumerate(style.SEQUENTIAL)],
            colorbar=dict(title="P(coating)", thickness=10),
            hovertemplate="A-scan %{x}<br>depth %{y}<br>P(coating) %{z:.2f}<extra></extra>"))
        common.layout(fig, height=420, showlegend=False)
        fig.update_yaxes(autorange="reversed", title="Depth (pixels)")
        fig.update_xaxes(title="A-scan")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("##### 5 · Solver")
    if row["status"] in ("measured", "undecided"):
        xc, zc, radius = inter["circle"]
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Fitted radius", f"{radius:.0f} µm")
        s1.caption(f"true {obj.diameter_um / 2:.0f} µm")
        s2.metric("Fit residual", f"{inter['fit_residual_um']:.2f} µm")
        s3.metric("A-scans used", f"{int(np.sum(inter['used']))}")
        s3.caption(f"below {cfg.solve.max_angle_deg:g}° incidence")
        s4.metric("Spread across A-scans", f"{100 * inter['intra_cv']:.1f} %")
        st.caption(f"Confidence = boundary finder {row['seg_confidence']:.2f} × fit "
                   f"{row['fit_confidence']:.2f} = {row['confidence']:.2f} (the gate passed it "
                   f"with {row['gate_confidence']:.2f}). Thickness is measured along the "
                   f"surface normal with Snell refraction, using n = {pooled_n:g}.")
    else:
        st.info("Too few A-scans carry both surfaces, so no thickness is reported.")
st.divider()
if st.button("Store the evidence for this decision",
             help="Saves the camera image, shape measures, probability map and surfaces under "
                  "a record ID and writes that ID to the audit trail."):
    from coatshield.compliance import registry
    from coatshield.compliance.audit import AuditTrail
    from coatshield.compliance.explain import save_explanation

    runtime = common.REPO_ROOT / "app" / ".runtime"
    model_hash = registry.combined_hash()
    rid = save_explanation(row, inter, model_hash, runtime / "explanations")
    AuditTrail(runtime / "audit.jsonl").append(
        "system", "system", "explanation_stored", f"record {rid}", new=row["status"],
        reason="evidence stored on request", model_hash=model_hash)
    st.success(f"Stored as record {rid}; the audit trail now points at it.")
common.footer(extra=f"pellet {int(pellet)}")
