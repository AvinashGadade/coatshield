"""Single source of randomness: rng(name) and torch determinism."""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np

_DEFAULT_MASTER_SEED: int | None = None


def _master_seed(seed: int | None) -> int:
    if seed is not None:
        return int(seed)
    global _DEFAULT_MASTER_SEED
    if _DEFAULT_MASTER_SEED is None:
        from coatshield.config import load_config

        _DEFAULT_MASTER_SEED = load_config().seed
    return _DEFAULT_MASTER_SEED


def name_key(name: str) -> int:
    """Stable 64-bit integer from a stream name (independent of PYTHONHASHSEED)."""
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")


def seed_sequence(name: str, seed: int | None = None) -> np.random.SeedSequence:
    return np.random.SeedSequence([_master_seed(seed), name_key(name)])


def rng(name: str, seed: int | None = None) -> np.random.Generator:
    """NumPy Generator derived from the master seed plus a fixed name.

    Each name is its own stream, so adding a new random step never changes the
    numbers of an old one. `seed` defaults to the master seed in configs/default.yaml.
    """
    return np.random.default_rng(seed_sequence(name, seed))


def set_torch_determinism(seed: int | None = None, warn_only: bool = False) -> None:
    """Seed everything and force deterministic torch algorithms.

    Training on GPU passes warn_only=True; the guarantee is on CPU inference.
    """
    master = _master_seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(master % (2**32))
    random.seed(master)
    np.random.seed(master % (2**32))  # noqa: NPY002 - seeds third-party code using the legacy API

    import torch

    torch.manual_seed(master)
    torch.cuda.manual_seed_all(master)
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
