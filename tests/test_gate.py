import numpy as np
import pytest

from coatshield.config import load_config
from coatshield.gate.classical import classify, measure, tune_twin_rule
from coatshield.gate.metrics import confusion_matrix, expected_calibration_error, gate_summary
from coatshield.gate.silhouettes import (
    CLASSES,
    DEFOCUSED,
    EMPTY,
    FINES,
    FOULED,
    PARTIAL,
    SINGLE,
    TOUCHING,
    TWIN,
    make_set,
    render,
)
from coatshield.seeds import rng


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def scored(cfg):
    images, labels, _ = make_set(cfg, "unit", 480)
    verdicts = [classify(im, cfg) for im in images]
    return (images, labels, np.array([v.label for v in verdicts]),
            np.array([v.confidence for v in verdicts]), verdicts)


def test_silhouettes_have_the_camera_format_and_are_reproducible(cfg):
    gv = cfg.gate_vision
    image, info = render(SINGLE, cfg, rng("test.gate.a"))
    assert image.shape == (gv.frame_px, gv.frame_px) == (256, 256) and image.dtype == np.uint8
    assert np.array_equal(image, render(SINGLE, cfg, rng("test.gate.a"))[0])
    assert info["class"] == "single"
    # Dark field: dark ground, bright rim, faint interior.
    assert np.median(image) < 20 and image.max() > 150
    # A 700 um pellet is about 127 pixels across.
    assert 700 / gv.pixel_um == pytest.approx(127, abs=1)
    assert len(CLASSES) == 8


def test_set_is_balanced(cfg):
    _, labels, infos = make_set(cfg, "balance", 64)
    assert np.bincount(labels).tolist() == [8] * 8
    assert all(infos[i]["class"] == CLASSES[labels[i]] for i in range(64))


def test_twin_has_lower_solidity_and_a_deeper_defect_than_a_single(cfg):
    single = measure(render(SINGLE, cfg, rng("test.gate.s"))[0], cfg)
    twin = measure(render(TWIN, cfg, rng("test.gate.t"))[0], cfg)
    assert single.solidity > 0.97 and single.circularity > 0.8
    assert twin.solidity < single.solidity - 0.03
    assert twin.defect_depth > 3 * single.defect_depth
    assert twin.aspect_ratio > 1.2 > single.aspect_ratio
    assert measure(render(EMPTY, cfg, rng("test.gate.e"))[0], cfg) is None


def test_twin_recall_and_false_twin_targets_on_held_out_images(cfg, scored):
    _, labels, predicted, confidence, _ = scored
    gv = cfg.gate_vision
    summary = gate_summary(labels, predicted, confidence, gv.single_confidence_min, gv.ece_bins)
    assert summary["twin_recall"] >= gv.twin_recall_min
    assert summary["false_twin_rate"] <= gv.false_twin_max + 0.02  # 60 singles: allow one more
    assert summary["twin_leak"] <= gv.twin_leak_max
    assert summary["non_single_leak"] <= 0.01
    assert summary["single_pass_rate"] > 0.85
    assert summary["accuracy"] > 0.9


@pytest.mark.parametrize("label", [TWIN, TOUCHING, PARTIAL, DEFOCUSED, FOULED, EMPTY, FINES])
def test_nothing_but_a_single_passes_the_gate(cfg, scored, label):
    _, labels, _, _, verdicts = scored
    passed = [v.passes and v.confidence >= cfg.gate_vision.single_confidence_min
              for v, lab in zip(verdicts, labels, strict=True) if lab == label]
    assert np.mean(passed) <= 0.02


def test_touching_pellets_are_told_apart_from_fused_ones(scored):
    _, labels, predicted, _, _ = scored
    assert (predicted[labels == TOUCHING] == TOUCHING).mean() > 0.6
    assert (predicted[labels == TWIN] == TWIN).mean() > 0.9


def test_tuning_respects_the_false_twin_limit(cfg, scored):
    images, labels, _, _, _ = scored
    tuned = tune_twin_rule([measure(im, cfg) for im in images], labels, cfg)
    assert tuned["false_twin_rate"] <= cfg.gate_vision.false_twin_max
    assert tuned["twin_recall"] >= cfg.gate_vision.twin_recall_min
    assert tuned["solidity_min"] in cfg.gate_vision.solidity_grid


def test_metrics():
    labels = np.array([0, 0, 1, 1, 2])
    predicted = np.array([0, 1, 1, 1, 0])
    matrix = confusion_matrix(labels, predicted)
    assert matrix.shape == (8, 8) and matrix[0, 0] == 1 and matrix[0, 1] == 1 and matrix[1, 1] == 2
    perfect = expected_calibration_error(np.array([1.0, 1.0]), np.array([True, True]), 10)
    overconfident = expected_calibration_error(np.array([0.9, 0.9]), np.array([True, False]), 10)
    assert perfect == 0.0 and overconfident == pytest.approx(0.4)


def test_cnn_gate_is_small_and_calibratable(cfg):
    import torch

    from coatshield.gate.cnn import GateCNN, fit_temperature

    torch.manual_seed(0)
    model = GateCNN(cfg.gate_vision.cnn_channels).eval()
    assert sum(p.numel() for p in model.parameters()) < 500_000
    logits = model(torch.rand(4, 1, 256, 256))
    assert logits.shape == (4, 8)
    # Overconfident logits are cooled by temperature scaling.
    labels = torch.tensor([0, 1, 2, 3] * 25)
    sharp = torch.nn.functional.one_hot(labels, 8).float() * 20
    sharp[::3] = sharp[::3].roll(1, dims=1)  # a third of them confidently wrong
    assert fit_temperature(model, sharp, labels) > 1.5
