"""Demo step 8: the agglomerate gate. Fused pellets are flagged, single pellets pass."""

import pandas as pd
import streamlit as st
from components import common, sidebar, story

from coatshield.gate.classical import classify
from coatshield.gate.silhouettes import CLASSES, render
from coatshield.seeds import rng

common.page("Gate")
story.banner()
cfg = sidebar.render()
gv = cfg.gate_vision

st.title("The agglomerate gate")
st.write("Before OCT measures a pellet, the camera image decides whether it is a single pellet "
         "worth measuring. Fused twins, touching pairs, cut-off, blurred and fouled images and "
         "fines are held back and counted.")


@st.cache_data(show_spinner=False)
def gallery(label: int, n: int, seed: int, config_json: str):
    from coatshield.config import Config

    c = Config.model_validate_json(config_json)
    out = []
    for i in range(n):
        image, _ = render(label, c, rng(f"app.gate.{label}.{i}", seed))
        verdict = classify(image, c)
        out.append((image, verdict.label, verdict.confidence,
                    verdict.features.as_dict() if verdict.features else None))
    return out


chosen = st.segmented_control("Class to show", CLASSES, default="twin",
                              format_func=str.capitalize) or "twin"
items = gallery(CLASSES.index(chosen), 8, cfg.seed, cfg.model_dump_json())
cols = st.columns(4)
for i, (image, label, confidence, features) in enumerate(items):
    passes = label == 0 and confidence >= gv.single_confidence_min
    with cols[i % 4]:
        st.image(image, use_container_width=True)
        verdict = "passes to OCT" if passes else "held back"
        st.markdown(f"**{CLASSES[label].capitalize()}** · {verdict}  \n"
                    f"confidence {confidence:.2f}")
        if features:
            st.caption(f"solidity {features['solidity']:.3f} · defect "
                       f"{features['defect_depth']:.3f}")
st.caption(f"Synthetic dark-field images, {gv.frame_px} × {gv.frame_px} pixels at "
           f"{gv.pixel_um:g} µm per pixel. Twin rule: solidity below {gv.solidity_min:g} or "
           f"convexity-defect depth above {gv.defect_max:g} of the radius.")

st.subheader("Classical method against the small CNN")
summary = common.latest_report("gate_comparison_*.csv")
matrix = common.latest_report("gate_confusion_classical_*.csv")
if summary is None or matrix is None:
    st.info("Run `python scripts/tune_gate.py` to fill this section.")
else:
    row = pd.read_csv(summary).iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Twin recall", f"{100 * row.twin_recall:.1f} %")
    c2.metric("Single pellets called twin", f"{100 * row.false_twin_rate:.1f} %")
    c3.metric("Single pellets that pass", f"{100 * row.single_pass_rate:.1f} %")
    c4.metric("Time per image (CPU)", f"{row.cpu_ms_per_image:.0f} ms")
    st.caption(f"Classical method on held-out synthetic images. Targets: twin recall at least "
               f"{100 * gv.twin_recall_min:.0f} %, false twins at most "
               f"{100 * gv.false_twin_max:.0f} %.")
    st.markdown("**Confusion matrix** (rows: true class, columns: gate verdict)")
    st.dataframe(pd.read_csv(matrix, index_col=0), use_container_width=True)
    st.info("The small CNN is defined but not trained yet, so the classical method is the one "
            "in use. The plan keeps the classical method unless the CNN beats it on this table.")
st.warning("Limits: these are synthetic silhouettes. How real fused pellets look, and how they "
           "misread in OCT, is unknown, so the gate's benefit is shown, not quantified. The "
           "check on real pellet photos (Roboflow) has not been run.")
common.footer()
