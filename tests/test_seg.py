import numpy as np
import pytest

from coatshield.config import load_config
from coatshield.seeds import rng
from coatshield.seg.dp import find_surfaces, shortest_path, surface_costs
from coatshield.seg.metrics import boundary_error, dice_per_class, thinnest_separable


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def probabilities(outer, inner, n_rows=200, sharp=0.97, noise=0.0, gen=None):
    """Class probabilities [3, rows, cols] for given surfaces, optionally corrupted."""
    n_cols = len(outer)
    rows = np.arange(n_rows)[:, None]
    label = (rows >= np.asarray(outer)[None, :]).astype(int) + (rows >= np.asarray(inner)[None, :])
    prob = np.full((3, n_rows, n_cols), (1 - sharp) / 2)
    for k in range(3):
        prob[k][label == k] = sharp
    if noise:
        prob = prob + noise * gen.random(prob.shape)
        prob /= prob.sum(axis=0, keepdims=True)
    return prob


def run(prob, cfg):
    sc = cfg.seg
    return find_surfaces(prob, sc.dp_max_jump_px, sc.dp_min_gap_px, sc.valid_min_prob)


# --- graph search ---------------------------------------------------------------


def test_graph_search_recovers_clean_surfaces(cfg):
    cols = np.arange(128)
    outer = (40 + 0.004 * (cols - 64) ** 2).astype(int)
    inner = outer + 30
    found = run(probabilities(outer, inner), cfg)
    assert np.array_equal(found["outer"], outer)
    assert np.array_equal(found["inner"], inner)
    assert found["valid"].all()
    assert found["margin_outer"].min() > 0.9


def test_graph_search_is_robust_to_noisy_probabilities(cfg):
    gen = rng("test.dp")
    cols = np.arange(128)
    outer = (60 + 15 * np.sin(cols / 20)).astype(int)
    inner = outer + 12
    found = run(probabilities(outer, inner, sharp=0.7, noise=0.3, gen=gen), cfg)
    assert np.abs(found["outer"] - outer).mean() < 1.0
    assert np.abs(found["inner"] - inner).mean() < 1.0


@pytest.mark.parametrize("case", ["thin", "crossing", "uniform", "inverted", "edge"])
def test_surfaces_never_cross(cfg, case):
    gen = rng(f"test.dp.{case}")
    n_cols, n_rows = 96, 120
    cols = np.arange(n_cols)
    if case == "thin":  # a film thinner than one pixel
        prob = probabilities(np.full(n_cols, 50), np.full(n_cols, 50), n_rows)
    elif case == "crossing":  # probabilities that ask for the inner surface above the outer
        prob = probabilities(40 + cols // 3, 60 - cols // 3, n_rows)
    elif case == "uniform":  # no information at all
        prob = np.full((3, n_rows, n_cols), 1 / 3)
    elif case == "inverted":  # pure noise
        prob = gen.random((3, n_rows, n_cols))
        prob /= prob.sum(axis=0, keepdims=True)
    else:  # surfaces at the last rows of the image
        prob = probabilities(np.full(n_cols, n_rows - 2), np.full(n_cols, n_rows - 1), n_rows)
    found = run(prob, cfg)
    assert (found["inner"] >= found["outer"] + cfg.seg.dp_min_gap_px).all() or case == "edge"
    assert (found["inner"] >= found["outer"]).all()
    for key in ("outer", "inner"):
        assert np.abs(np.diff(found[key])).max() <= cfg.seg.dp_max_jump_px
        assert found[key].min() >= 0 and found[key].max() < n_rows


def test_columns_without_signal_are_flagged_not_guessed(cfg):
    outer, inner = np.full(64, 30), np.full(64, 50)
    prob = probabilities(outer, inner)
    prob[:, :, 40:] = np.array([0.98, 0.01, 0.01])[:, None, None]  # nothing but "above"
    found = run(prob, cfg)
    assert found["valid"][:40].all() and not found["valid"][40:].any()


def test_smoothness_limit_overrides_an_isolated_jump(cfg):
    outer = np.full(64, 40)
    outer[30] = 80  # one column disagrees wildly
    found = run(probabilities(outer, outer + 20), cfg)
    assert abs(int(found["outer"][30]) - 40) <= cfg.seg.dp_max_jump_px


def test_shortest_path_respects_the_lower_bound():
    cost = np.ones((50, 20))
    cost[10, :] = 0.0  # cheapest row
    free = shortest_path(cost, 3, np.zeros(20, dtype=np.int32))
    assert (free == 10).all()
    bounded = shortest_path(cost, 3, np.full(20, 25, dtype=np.int32))
    assert (bounded >= 25).all()


def test_surface_costs_are_lowest_at_the_true_rows():
    prob = probabilities(np.full(8, 20), np.full(8, 35), n_rows=60)
    outer_cost, inner_cost = surface_costs(prob)
    assert (outer_cost.argmin(axis=0) == 20).all()
    assert (inner_cost.argmin(axis=0) == 35).all()


# --- metrics --------------------------------------------------------------------


def test_boundary_error_uses_only_valid_columns():
    truth = np.array([10.0, 10.0, np.nan, 10.0])
    found = np.array([11.0, 9.0, 50.0, 30.0])
    valid = np.array([True, True, True, False])
    err = boundary_error(found, truth, valid, px_um=0.36)
    assert err["mae_px"] == pytest.approx(1.0) and err["n"] == 2
    assert err["mae_um"] == pytest.approx(0.36)
    assert np.isnan(boundary_error(found, truth, np.zeros(4, bool))["mae_px"])


def test_dice_per_class():
    truth = np.array([[0, 0, 1, 1, 2, 2]])
    pred = np.array([[0, 1, 1, 1, 2, 2]])
    dice = dice_per_class(pred, truth, 4)
    assert dice[0] == pytest.approx(2 / 3) and dice[1] == pytest.approx(0.8) and dice[2] == 1.0
    assert np.isnan(dice[3])


def test_thinnest_separable_film():
    thickness = np.exp(np.linspace(np.log(2), np.log(40), 600))
    separated = thickness > 5.0
    assert thinnest_separable(thickness, separated) == pytest.approx(5.0, rel=0.3)
    assert thinnest_separable(thickness, np.ones(600, bool)) == pytest.approx(2.0)
    assert np.isnan(thinnest_separable(thickness, np.zeros(600, bool)))


# --- network, data and export (torch side) ---------------------------------------


def test_unet_shape_and_parameter_count(cfg):
    import torch

    from coatshield.seg.unet import CompactUNet, count_parameters

    model = CompactUNet(3, cfg.seg.widths)
    assert 1.5e6 < count_parameters(model) < 2.5e6  # "about 2 M"
    out = model(torch.zeros(2, 1, 64, 32))
    assert out.shape == (2, 3, 64, 32)
    model.replace_head(6)
    assert model(torch.zeros(1, 1, 64, 32)).shape == (1, 6, 64, 32)
    assert not any(isinstance(m, torch.nn.Dropout) for m in model.modules())


def test_hybrid_model_adds_a_transformer_bottleneck(cfg):
    import torch

    from coatshield.seg.hybrid import HybridUNet
    from coatshield.seg.unet import CompactUNet, count_parameters

    hybrid = HybridUNet(3, cfg.seg.widths)
    assert count_parameters(hybrid) > count_parameters(CompactUNet(3, cfg.seg.widths))
    assert hybrid(torch.zeros(1, 1, 64, 32)).shape == (1, 3, 64, 32)


def test_loss_falls_when_overfitting_one_batch(cfg):
    import torch

    from coatshield.seeds import set_torch_determinism
    from coatshield.seg.train import segmentation_loss
    from coatshield.seg.unet import CompactUNet

    set_torch_determinism(1, warn_only=True)
    model = CompactUNet(3, (4, 8, 16))
    rows = torch.arange(32)[None, :, None].expand(2, 32, 16)
    target = (rows >= 10).long() + (rows >= 20).long()
    image = target.float()[:, None] / 2 + 0.1 * torch.rand(2, 1, 32, 16)
    weight = torch.ones(2, 32, 16)
    weight[:, :, :3] = 0  # unweighted columns must not matter
    optim = torch.optim.AdamW(model.parameters(), lr=1e-2)
    first = None
    for _ in range(120):
        optim.zero_grad()
        loss = segmentation_loss(model(image), target, weight, 1.0)
        loss.backward()
        optim.step()
        first = first if first is not None else float(loss)
    assert float(loss) < 0.6 * first
    torch.use_deterministic_algorithms(False)


def test_augmentation_keeps_layer_order_and_marks_wrapped_rows(cfg):
    from coatshield.seg.datasets import augment

    gen = rng("test.augment")
    rows = np.arange(512)[:, None]
    mask = np.repeat(((rows >= 100).astype(int) + (rows >= 160)), 128, axis=1)
    image = mask / 2.0
    for _ in range(20):
        img, m, w = augment(image, mask, np.ones(mask.shape, np.float32), cfg.seg.synthetic_crop,
                            cfg, gen)
        assert img.shape == m.shape == w.shape == tuple(cfg.seg.synthetic_crop)
        assert 0.0 <= img.min() and img.max() <= 1.0
        ordered = np.diff(m.astype(int), axis=0) >= 0
        trusted = (w[1:] > 0) & (w[:-1] > 0)
        assert ordered[trusted].all()  # top-to-bottom order wherever the label is trusted


def test_oct5k_split_is_by_patient_volume(cfg):
    from coatshield.seg.datasets import OCT5K_ROOT, oct5k_pairs, split_by_volume

    if not (OCT5K_ROOT / "Images" / "Images_Manual").exists():
        pytest.skip("OCT5k images not downloaded")
    pairs = oct5k_pairs()
    assert len(pairs) == 1672
    splits = split_by_volume(pairs, cfg)
    volumes = {name: {p[2] for p in part} for name, part in splits.items()}
    assert not volumes["train"] & volumes["val"]
    assert not volumes["train"] & volumes["test"]
    assert not volumes["val"] & volumes["test"]
    assert sum(len(v) for v in splits.values()) == 1672
    assert all(len(v) >= 3 for v in volumes.values())


def test_pellet_dataset_returns_images_masks_and_weights(cfg):
    from coatshield.seg.datasets import SYNTHETIC_ROOT, PelletDataset

    if not (SYNTHETIC_ROOT / "val" / "images.npy").exists():
        pytest.skip("synthetic pellet set not generated")
    ds = PelletDataset("val", cfg, train=True)
    image, mask, weight = ds[0]
    assert image.shape == (1, *cfg.seg.synthetic_crop)
    assert mask.shape == weight.shape == tuple(cfg.seg.synthetic_crop)
    assert set(np.unique(mask.numpy())) <= {0, 1, 2}
    again = ds[0]
    assert (image == again[0]).all()  # augmentation is reproducible per epoch and index
    ds.epoch = 1
    assert not (image == ds[0][0]).all()


def test_onnx_matches_pytorch_and_is_bit_identical_across_runs(cfg, tmp_path):
    import sys

    import torch

    from coatshield.config import REPO_ROOT
    from coatshield.seg.infer import Segmenter
    from coatshield.seg.unet import CompactUNet

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from export_onnx import export

    torch.manual_seed(0)
    model = CompactUNet(3, cfg.seg.widths)
    path = tmp_path / "unet.onnx"
    assert export(model, path, cfg.oct.depth_pixels, cfg.oct.out_ascans) < 1e-3
    seg = Segmenter(path, cfg)
    scan = rng("test.onnx").integers(0, 256, (cfg.oct.out_ascans, cfg.oct.depth_pixels),
                                     dtype=np.uint8)
    first = seg.probabilities(scan)
    assert first.shape == (3, cfg.oct.depth_pixels, cfg.oct.out_ascans)
    assert np.allclose(first.sum(axis=0), 1.0, atol=1e-5)
    for _ in range(25):
        assert seg.probabilities(scan).tobytes() == first.tobytes()
    found = seg.surfaces(scan)
    assert (found["inner"] >= found["outer"] + cfg.seg.dp_min_gap_px).all()


def test_steep_pellet_with_empty_sides_is_followed(cfg):
    """A sphere climbs several depth pixels per column and is flanked by empty columns;
    the search must follow it and leave the empty columns unflagged, not be dragged off."""
    cols = np.arange(128)
    x = (cols - 63.5) * 4.0  # um
    radius, px = 300.0, 0.364
    inside = np.abs(x) < 0.55 * radius
    sag = (radius - np.sqrt(radius**2 - np.where(inside, x, 0.0) ** 2)) / px
    outer = np.where(inside, 40 + sag, 0).astype(int)
    inner = outer + 45
    assert np.abs(np.diff(outer[inside])).max() > 3  # steeper than the guide's 3 px
    prob = probabilities(outer, inner, n_rows=512)
    prob[:, :, ~inside] = np.array([0.98, 0.01, 0.01])[:, None, None]
    found = run(prob, cfg)
    assert np.array_equal(found["valid"], inside)
    assert np.abs(found["outer"][inside] - outer[inside]).max() <= 1
    assert np.abs(found["inner"][inside] - inner[inside]).max() <= 1
