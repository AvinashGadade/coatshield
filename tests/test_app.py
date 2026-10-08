"""Dashboard smoke tests: every page renders from the precomputed assets, story mode walks."""

import ast
import sys
import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from coatshield.config import REPO_ROOT

APP = REPO_ROOT / "app"
PAGES = sorted((APP / "pages").glob("*.py"))
pytestmark = pytest.mark.skipif(
    not any((APP / "assets" / "bundles").glob("*/meta.json")),
    reason="run scripts/precompute_scenarios.py first",
)


@pytest.fixture(autouse=True)
def app_on_path(monkeypatch):
    monkeypatch.syspath_prepend(str(APP))
    monkeypatch.chdir(APP)


def run(script: Path, **state) -> AppTest:
    at = AppTest.from_file(str(script), default_timeout=120)
    for key, value in state.items():
        at.session_state[key] = value
    return at.run()


@pytest.mark.parametrize("script", [APP / "Home.py", *PAGES], ids=lambda p: p.stem)
def test_page_renders_without_error(script):
    start = time.perf_counter()
    at = run(script)
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    assert at.title
    assert time.perf_counter() - start < 90  # first render includes imports; cached is far less


def test_default_pages_come_from_precomputed_assets():
    at = run(APP / "pages" / "3_Controllers.py")
    assert any("precomputed" in c.value for c in at.caption)
    labels = [m.label for m in at.metric]
    assert labels[:4] == ["C0 · Gravimetric", "C1 · Raw mean", "C2 · Raw d10", "C3 · CoatShield"]
    below = {m.label[:2]: float(m.value.split()[0]) for m in at.metric[:4]}
    assert sum("truly below spec" in c.value for c in at.caption) == 4
    assert below["C3"] <= 11 < below["C2"]
    assert any("× the" in w.value for w in at.warning)


def test_story_mode_walks_every_step():
    sys.path.insert(0, str(APP))
    from components import story

    at = run(APP / "Home.py")
    at.button[0].click().run()  # Start the demo
    assert at.session_state["story_index"] == 0
    for index, step in enumerate(story.STEPS):
        page = run(APP / step.page, story_index=index, story_view=dict(step.view), **step.state)
        assert not page.exception, (step.title, page.exception)
        assert any(step.title in md.value for md in page.markdown)
        for key, value in step.state.items():
            assert page.session_state[key] == value


def test_app_never_imports_torch():
    """The deployed app loads ONNX models only; torch must not be imported by app code."""
    offenders = []
    for path in [*APP.rglob("*.py"), REPO_ROOT / "coatshield" / "bundle.py",
                 REPO_ROOT / "coatshield" / "style.py"]:
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            if any(n.split(".")[0] == "torch" for n in names):
                offenders.append(str(path))
    assert not offenders
