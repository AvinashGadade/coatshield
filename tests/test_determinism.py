"""Cross-process determinism: each model run in two fresh processes gives identical bytes."""

import sys

import pytest

from coatshield.compliance.registry import load_manifest
from coatshield.config import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from check_determinism import run_twice  # noqa: E402


def test_gate_is_identical_across_processes():
    first, second = run_twice("gate", 24)
    assert first == second


@pytest.mark.skipif("unet" not in load_manifest()["models"], reason="no trained boundary finder")
def test_boundary_finder_is_identical_across_processes():
    first, second = run_twice("unet", 24)
    assert first == second


def test_twin_and_estimator_are_identical_across_processes():
    first, second = run_twice("twin", 0)
    assert first == second
