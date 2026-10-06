"""Story mode: a scripted walk through the demo. Next sets the sliders and switches page."""

from __future__ import annotations

from dataclasses import dataclass, field

import streamlit as st

from components import sidebar


@dataclass(frozen=True)
class Step:
    demo_step: str  # number in the demo script
    title: str
    page: str  # script path relative to app/
    say: str  # what the presenter says
    state: dict = field(default_factory=dict)  # sidebar values this step needs
    view: dict = field(default_factory=dict)  # page-specific view settings


STEPS = (
    Step("1", "The batch", "pages/1_Batch.py",
         "This is the bed and the measurement window. The window sees only a sliver of the "
         "batch, and bigger pellets reach it more often."),
    Step("2", "Three curves", "pages/2_Distributions.py",
         "The true thickness distribution, the raw window sample and the corrected estimate. "
         "The raw sample reads thicker than the batch really is.",
         view={"time": "C2"}),
    Step("3", "Two controllers, one batch", "pages/3_Controllers.py",
         "Every rule watches the same batch. The raw rules stop early and ship about twice the "
         "out-of-spec pellets they believe; the corrected rule stops later, on target."),
    Step("4", "The honesty slider: no window bias", "pages/3_Controllers.py",
         "Set the window bias to none. The raw and corrected rules now agree: the claim rests "
         "on this one assumption, and here it is on a slider.",
         state={"m": 0.0}),
    Step("4", "The honesty slider: across all settings", "pages/4_Sensitivity.py",
         "Every combination of window bias and size-dependent growth, from the validation run. "
         "The toggle shows where the correction stops working.",
         state={"m": 4.0}),
)


def _go(index: int) -> None:
    index = max(0, min(index, len(STEPS) - 1))
    step = STEPS[index]
    st.session_state.update(sidebar.defaults())
    st.session_state.update(step.state)
    st.session_state["story_index"] = index
    st.session_state["story_view"] = dict(step.view)
    st.session_state["_goto"] = step.page


def start() -> None:
    _go(0)


def stop() -> None:
    st.session_state.pop("story_index", None)
    st.session_state.pop("story_view", None)
    st.session_state.update(sidebar.defaults())
    st.session_state["_goto"] = "Home.py"


def follow_redirect() -> None:
    """Switch page if a story button asked for it (callbacks cannot switch pages themselves)."""
    target = st.session_state.pop("_goto", None)
    if target:
        st.switch_page(target)


def view(key: str, default=None):
    """A view setting requested by the current story step (consumed once)."""
    return st.session_state.get("story_view", {}).pop(key, default)


def banner() -> None:
    """The presenter's strip at the top of a page while story mode is on."""
    follow_redirect()
    index = st.session_state.get("story_index")
    if index is None:
        return
    step = STEPS[index]
    with st.container(border=True):
        text, back, nxt, end = st.columns([8, 1, 1, 1], vertical_alignment="center")
        text.markdown(f"**Demo step {step.demo_step} · {step.title}** "
                      f"({index + 1} of {len(STEPS)})  \n{step.say}")
        back.button("Back", on_click=_go, args=(index - 1,), disabled=index == 0,
                    use_container_width=True)
        nxt.button("Next step", type="primary", on_click=_go, args=(index + 1,),
                   disabled=index == len(STEPS) - 1, use_container_width=True)
        end.button("Reset", on_click=stop, use_container_width=True)
