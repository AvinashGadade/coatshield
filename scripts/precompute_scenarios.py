"""Precompute the dashboard's scenario bundles so the scripted demo is instant.

Writes one bundle (Parquet + NPZ + JSON, under 1 MB) per scenario to app/assets/bundles/:
every product preset x fault scenario, every scale preset, and the window-bias values the
story mode uses. Each bundle is named by its config hash; the app looks it up by hash.
"""

from __future__ import annotations

import argparse
import os
import time
from multiprocessing import get_context

for _var in ("NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from _common import REPO_ROOT  # noqa: E402

from coatshield.bundle import build_bundle, bundle_key, load_bundle, save_bundle  # noqa: E402
from coatshield.config import (  # noqa: E402
    FAULT_SCENARIOS,
    PRODUCT_PRESETS,
    SCALE_PRESETS,
    Config,
    load_config,
)

BUNDLES = REPO_ROOT / "app" / "assets" / "bundles"
DEFAULT_SCALE = "gpcg30_30kg"
STORY_M = (0.0, 1.0, 2.0, 4.0)  # window-bias values reachable from story mode and the slider


def scenario_configs() -> dict[str, Config]:
    """Label -> config, exactly as the app's sidebar would build it."""
    out: dict[str, Config] = {}

    def add(label: str, presets: list[str], overrides: dict | None = None) -> None:
        cfg = load_config(presets=presets)
        cfg = cfg.with_overrides({**(overrides or {}), "batch.n_pellets": cfg.app.n_pellets})
        out[label] = cfg

    for product in PRODUCT_PRESETS:
        for scenario in FAULT_SCENARIOS:
            add(f"{product} / {scenario}", [product, DEFAULT_SCALE], {"fault.scenario": scenario})
    for scale in SCALE_PRESETS:
        add(f"{PRODUCT_PRESETS[0]} / {scale}", [PRODUCT_PRESETS[0], scale])
    for m in STORY_M:
        add(f"{PRODUCT_PRESETS[0]} / m={m:g}", [PRODUCT_PRESETS[0], DEFAULT_SCALE],
            {"window.size_bias_m": m})
    return out


def _build(item: tuple[str, str]) -> tuple[str, str, float]:
    label, config_json = item
    start = time.perf_counter()
    bundle = build_bundle(Config.model_validate_json(config_json), bootstrap="all")
    save_bundle(bundle, BUNDLES)
    return label, bundle.key, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--force", action="store_true", help="rebuild bundles that exist")
    parser.add_argument("--only", default=None, help="build only labels containing this text")
    args = parser.parse_args()

    todo = []
    for label, cfg in scenario_configs().items():
        if args.only and args.only not in label:
            continue
        if args.force or load_bundle(bundle_key(cfg), BUNDLES) is None:
            todo.append((label, cfg.model_dump_json()))
    print(f"{len(todo)} bundles to build with {args.workers} workers -> {BUNDLES}")
    with get_context("spawn").Pool(args.workers) as pool:
        for label, key, seconds in pool.imap_unordered(_build, todo):
            print(f"  {key}  {label}  ({seconds:.0f} s)", flush=True)
    size = sum(f.stat().st_size for f in BUNDLES.rglob("*") if f.is_file())
    print(f"{len(list(BUNDLES.iterdir()))} bundles on disk, {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
