import subprocess
import sys

import numpy as np

from coatshield.config import REPO_ROOT, load_config
from coatshield.seeds import rng, set_torch_determinism


def test_same_name_same_numbers():
    assert np.array_equal(rng("twin").random(100), rng("twin").random(100))


def test_names_are_independent_streams():
    a, b = rng("twin").random(1000), rng("oct").random(1000)
    assert not np.array_equal(a, b)
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.1


def test_master_seed_changes_stream_and_default_is_config_seed():
    assert not np.array_equal(rng("twin", seed=1).random(10), rng("twin", seed=2).random(10))
    seed = load_config().seed
    assert np.array_equal(rng("twin").random(10), rng("twin", seed=seed).random(10))


def test_reproducible_across_processes():
    code = "from coatshield.seeds import rng; print(rng('twin').random(5).tolist())"
    runs = [
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO_ROOT
        )
        for _ in range(2)
    ]
    assert runs[0].stdout == runs[1].stdout
    assert runs[0].stdout.strip() == str(rng("twin").random(5).tolist())


def test_set_torch_determinism():
    import torch

    set_torch_determinism(seed=7)
    a = torch.rand(5)
    set_torch_determinism(seed=7)
    assert torch.equal(a, torch.rand(5))
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.benchmark is False
    torch.use_deterministic_algorithms(False)
