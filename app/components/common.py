"""Shared page plumbing: paths, cached data access, chart layout, footer."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coatshield import style  # noqa: E402
from coatshield.bundle import Bundle, build_bundle, bundle_key, load_bundle  # noqa: E402
from coatshield.config import Config  # noqa: E402

ASSETS = REPO_ROOT / "app" / "assets"
BUNDLES = ASSETS / "bundles"
REPORTS = REPO_ROOT / "reports"
NOTE = "Demonstration on simulated data, not a validated GMP system."


def page(title: str) -> None:
    st.set_page_config(page_title=f"CoatShield · {title}", layout="wide")


@st.cache_resource(show_spinner=False, max_entries=24)
def _precomputed(key: str) -> Bundle | None:
    return load_bundle(key, BUNDLES)


@st.cache_resource(show_spinner="Simulating the batch for these settings…", max_entries=8)
def _live(key: str, config_json: str) -> Bundle:
    return build_bundle(Config.model_validate_json(config_json), bootstrap="needed")


def get_bundle(cfg: Config) -> tuple[Bundle, bool]:
    """The scenario bundle for cfg and whether it came from the precomputed assets.

    Precomputed runs use app.n_pellets; any other setting is simulated live with the
    smaller app.n_pellets_live so a slider move answers in seconds.
    """
    full = cfg.with_overrides({"batch.n_pellets": cfg.app.n_pellets})
    bundle = _precomputed(bundle_key(full))
    if bundle is not None:
        return bundle, True
    live = cfg.with_overrides({"batch.n_pellets": cfg.app.n_pellets_live})
    return _live(bundle_key(live), live.model_dump_json()), False


def require_estimates(bundle: Bundle, cfg: Config) -> None:
    """Stop the page with an explanation if the window never held enough pellets to estimate."""
    if bundle.est.hybrid_d10.notna().any():
        return
    per_window = cfg.window.objects_per_min * cfg.estimator.window_min
    st.warning(
        f"With these settings the window holds about {per_window:.0f} objects in "
        f"{cfg.estimator.window_min:g} minutes, and after the gate and undecided scans fewer "
        f"than the {cfg.controller.min_objects} accepted pellets needed for an estimate. "
        "Raise “Objects per minute” in the sidebar.")
    footer(bundle, False)
    st.stop()


def layout(fig, title: str | None = None, height: int = 420, **kwargs):
    """Recessive grid and axes, ink-coloured text, legend on top."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=style.TEXT), x=0, xanchor="left"),
        height=height,
        margin=dict(l=10, r=10, t=48 if title else 16, b=10),
        paper_bgcolor=style.SURFACE,
        plot_bgcolor=style.SURFACE,
        font=dict(color=style.TEXT_SECONDARY, size=13),
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0.0),
        hoverlabel=dict(bgcolor="white", font_color=style.TEXT),
        **kwargs,
    )
    fig.update_xaxes(gridcolor=style.GRID, linecolor=style.GRID, zeroline=False)
    fig.update_yaxes(gridcolor=style.GRID, linecolor=style.GRID, zeroline=False)
    return fig


def latest_report(pattern: str) -> Path | None:
    """Newest file in reports/ matching a glob pattern (reports are named by config hash)."""
    hits = sorted(REPORTS.glob(pattern), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def model_versions() -> str:
    """Active model versions for the footer and the batch record."""
    from coatshield.compliance.registry import active_versions

    versions = active_versions()
    if not versions:
        return "models: none registered yet (classical gate, no trained networks)"
    return "models: " + ", ".join(f"{name} {v['version']} ({v['short_hash']})"
                                  for name, v in versions.items())


def footer(bundle: Bundle | None = None, precomputed: bool | None = None,
           extra: str | None = None) -> None:
    parts = [NOTE]
    if bundle is not None:
        source = "precomputed" if precomputed else "simulated live"
        parts.append(f"config {bundle.meta['config_hash']} · {bundle.meta['n_pellets']:,} "
                     f"simulated pellets · {source}")
    if extra:
        parts.append(extra)
    parts.append(model_versions())
    st.caption(" · ".join(parts))
