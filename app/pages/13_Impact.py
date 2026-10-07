"""Impact calculator: every input is the user's own number. Nothing is prefilled."""

import streamlit as st
from components import common, sidebar, story

common.page("Impact")
story.banner()
sidebar.render()

st.title("Impact calculator")
st.write("Enter your own figures. Nothing here is prefilled, and nothing is assumed about your "
         "plant. The calculator only multiplies what you enter.")

left, right = st.columns(2)
with left:
    batches = st.number_input("Batches per year", min_value=0, value=None, step=1,
                              placeholder="your number")
    value = st.number_input("Value of one batch (your currency)", min_value=0.0, value=None,
                            placeholder="your number")
    rate_now = st.number_input("Batches failing on coating today (%)", min_value=0.0,
                               max_value=100.0, value=None, placeholder="your number")
with right:
    rate_after = st.number_input("Batches you expect to fail with in-line control (%)",
                                 min_value=0.0, max_value=100.0, value=None,
                                 placeholder="your estimate")
    days_now = st.number_input("Days from end of coating to release today", min_value=0.0,
                               value=None, placeholder="your number")
    days_after = st.number_input("Days to release you expect with in-line data", min_value=0.0,
                                 value=None, placeholder="your estimate")

st.divider()
if None in (batches, rate_now, rate_after):
    st.info("Fill in batches per year and the two failure rates to see avoided batches.")
else:
    avoided = batches * (rate_now - rate_after) / 100.0
    c1, c2 = st.columns(2)
    c1.metric("Out-of-spec batches avoided per year", f"{avoided:,.1f}")
    if value is not None:
        c2.metric("Value of those batches per year", f"{avoided * value:,.0f}")
    else:
        c2.caption("Add the value of one batch to see this as money.")
if None in (batches, days_now, days_after):
    st.info("Fill in batches per year and the two release times to see release days saved.")
else:
    st.metric("Release days saved per year", f"{batches * (days_now - days_after):,.0f}")
st.caption("These are your inputs multiplied together. The prototype does not predict your "
           "failure rate or release time; real numbers need a trial on your equipment.")
common.footer()
