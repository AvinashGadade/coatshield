"""The sidebar shared by every page. Its widgets edit a copy of the config."""

from __future__ import annotations

import streamlit as st

from coatshield.config import (
    FAULT_SCENARIOS,
    PRODUCT_PRESETS,
    SCALE_PRESETS,
    Config,
    load_config,
)
from components import common  # noqa: F401  (puts the repo on sys.path)

LABELS = {
    "enteric": "Enteric", "sustained_release": "Sustained release", "taste_mask": "Taste mask",
    "multilab_2kg": "MultiLab 2 kg", "gpcg30_30kg": "GPCG 30 kg", "fbc125": "FBC 125",
    "none": "None", "substrate_shift": "Substrate shift", "nozzle_block": "Nozzle partial block",
    "over_wetting": "Over-wetting", "spray_drying": "Spray-drying",
    "maldistribution": "Maldistribution", "window_fouling": "Window fouling",
    "atline": "At-line sample", "coa": "Certificate of analysis", "oracle": "Oracle (debug)",
}
REFERENCES = ("atline", "coa", "oracle")


def _preset_config() -> Config:
    return load_config(presets=[st.session_state["product"], st.session_state["scale"]])


def defaults() -> dict:
    """Widget values that reproduce the default configuration."""
    base = load_config(presets=[PRODUCT_PRESETS[0], SCALE_PRESETS[1]])
    return {
        "product": PRODUCT_PRESETS[0],
        "scale": SCALE_PRESETS[1],
        "scenario": "none",
        "fault_start_h": base.fault.start_h,
        "m": base.window.size_bias_m,
        "k": base.wurster.size_growth_exponent_k,
        "gamma": base.window.hidden_selection_gamma,
        "spec": base.spec.d10_min_um,
        "objects_per_min": int(base.window.objects_per_min),
        "reference": base.estimator.reference,
        "seed": base.seed,
    }


def init_state() -> None:
    for key, value in defaults().items():
        st.session_state.setdefault(key, value)


def reset_state() -> None:
    st.session_state.update(defaults())


def _on_product_change() -> None:
    st.session_state["spec"] = _preset_config().spec.d10_min_um


def current_config() -> Config:
    """The config described by the sidebar state (without drawing the sidebar)."""
    init_state()
    s = st.session_state
    return _preset_config().with_overrides(
        {
            "fault.scenario": s["scenario"],
            "fault.start_h": float(s["fault_start_h"]),
            "window.size_bias_m": float(s["m"]),
            "wurster.size_growth_exponent_k": float(s["k"]),
            "window.hidden_selection_gamma": float(s["gamma"]),
            "spec.d10_min_um": float(s["spec"]),
            "window.objects_per_min": float(s["objects_per_min"]),
            "estimator.reference": s["reference"],
            "seed": int(s["seed"]),
        }
    )


def render() -> Config:
    """Draw the sidebar and return the config it describes."""
    init_state()
    with st.sidebar:
        st.header("Batch settings")
        st.selectbox("Product preset", PRODUCT_PRESETS, key="product", format_func=LABELS.get,
                     on_change=_on_product_change)
        st.selectbox("Scale preset", SCALE_PRESETS, key="scale", format_func=LABELS.get)
        st.selectbox("Fault scenario", FAULT_SCENARIOS, key="scenario", format_func=LABELS.get)
        st.slider("Fault starts at (h)", 0.0, 8.0, step=0.5, key="fault_start_h",
                  disabled=st.session_state["scenario"] in ("none", "substrate_shift"))
        st.subheader("Assumptions")
        st.slider("Window size bias m", 0.0, 4.0, step=0.5, key="m",
                  help="How strongly the window favours big pellets. Never measured: "
                       "the key assumption. 0 = no bias.")
        st.slider("Size-growth exponent k", 0.0, 1.5, step=0.25, key="k",
                  help="How much faster big pellets coat. Published large-to-small growth "
                       "ratios of 1.08-1.81 correspond to about 0.2-1.3.")
        st.slider("Hidden selection γ", 0.0, 1.0, step=0.25, key="gamma",
                  help="Selection the camera cannot see. The correction does not undo it.")
        st.subheader("Measurement and spec")
        st.number_input("Spec: d10 at least (µm)", 1.0, 40.0, step=0.5, key="spec")
        st.slider("Objects per minute", 100, 1000, step=100, key="objects_per_min")
        st.selectbox("Reference size distribution", REFERENCES, key="reference",
                     format_func=LABELS.get)
        st.number_input("Seed", 0, 2**31 - 1, step=1, key="seed")
        st.button("Reset to defaults", on_click=reset_state, use_container_width=True)
    return current_config()
