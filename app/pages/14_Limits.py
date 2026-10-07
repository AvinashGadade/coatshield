"""The honest limits list and every source."""

# ruff: noqa: E501  (long lines of prose)

import streamlit as st
from components import common, sidebar, story

common.page("Limits and sources")
story.banner()
sidebar.render()

st.title("Limits and sources")
st.header("What this prototype does not show")
st.markdown("""
- **No real coated-pellet scans were used.** The prototype makes no claim about accuracy on real pellets. Real OCT scans with microscopy cross-sections are the first thing to ask a plant for.
- **The headline depends on two assumptions.** The window favours big pellets (never measured), and big pellets coat faster (published: large pellets grow 1.08 to 1.81 times as fast as small ones within one batch). With either one absent, the raw rule does not ship extra out-of-spec pellets.
- **The correction only undoes selection by size**, which the camera sees. If the window also selects on something the camera cannot see, the corrected estimate degrades; the Sensitivity page shows by how much.
- **The estimator relies on a known measurement noise.** If the true noise is lower than the stated value, the corrected d10 reads high.
- **Retinal OCT is used only for pretraining.** Accuracy on eye layers is not accuracy on coatings.
- **Pigmented coatings and real pellet velocity are not modelled from measured data.**
- **How fused pellets misread is unknown**, so the gate's benefit is shown, not quantified. The gate was tested on synthetic silhouettes only.
- **Maldistribution is weakly detectable** at 1 µm of measurement noise in this simulation.
- **The refractive-index method based on reflectance** is only as accurate as its reflector calibration (a 10 % calibration error moves n by about 0.05) and reads low under window fouling.
- **The dissolution projection is illustrative**: a line through three published points.
- **The compliance features are demonstrations of the design**, not a validated GMP system. The system recommends; an operator decides. No code path connects to a safety interlock.
""")

st.header("Sources")
st.subheader("Data")
st.markdown("""
- OCT5k: retinal OCT with layer annotations, UCL Research Data Repository, doi:10.5522/04/22128671 (labels CC0; research use per the paper). Scan images from Rasti, Rabbani, Mehri, Hajizadeh, IEEE TMI 37(4):1024-1034, 2018.
- In-vivo human skin OCT dataset, Zenodo record 18095266 (CC BY 4.0).
- Duke Chiu 2015 DME dataset (research and education only; fallback, not redistributed).
- Roboflow pellets (CC BY 4.0; optional sanity check, not yet run).
""")
st.subheader("Physics and process parameters")
st.markdown("""
- Speckle simulation model for spectral-domain OCT, arXiv 1406.3448.
- Refractive index and thickness of multilayer samples by FD-OCT, arXiv 2404.11736.
- Refractive index of pharmaceutical powders in the short-wave infrared, arXiv 2401.10667.
- Wolfgang et al. 2025, industrial-scale multiparticulate OCT (PubMed 40174809); Wolfgang et al. 2020, at-line validation of OCT coating thickness.
- Lin et al. 2017, pharmaceutical film coating catalogue for SD-OCT.
- Sacher et al. 2021, in-line OCT coating attributes.
- Li 2015, Wurster particle motion and coating, Chalmers thesis.
- Jiang et al. 2020, Wurster coating uniformity by CFD-DEM-Monte Carlo.
- Polymer film thickness and drug release from coated pellets, Pharmaceutics 16:1307.
- Exploring a modern control strategy for Wurster coating, Pharmaceutical Technology.
- InnoGlobal Eyecon² specifications.
""")
st.subheader("Model choice")
st.markdown("""
- Kugelman et al. 2022, U-Net variants for OCT retinal layers (Sci Rep).
- Isensee et al. 2024, nnU-Net Revisited, arXiv 2404.09556.
- Matsoukas et al. 2021, CNNs against transformers for medical images.
- LightReSeg, light-weight retinal layer segmentation, arXiv 2404.16346.
""")
st.caption("Every parameter in the simulation is listed with its source in "
           "`configs/default.yaml`.")
common.footer()
