"""Model registry: models/manifest.json lists every model with its SHA-256.

The app checks each hash before loading a model and refuses to run on a mismatch.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from coatshield.config import REPO_ROOT

MODELS_DIR = REPO_ROOT / "models"


class ModelIntegrityError(Exception):
    """A model file is missing, unlisted, or does not match its registered hash."""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest(models_dir: Path = MODELS_DIR) -> dict:
    path = Path(models_dir) / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else {"models": {}}


def verified_path(name: str, models_dir: Path = MODELS_DIR) -> Path:
    """Path of a registered model, after checking the file against its registered hash."""
    entry = load_manifest(models_dir)["models"].get(name)
    if entry is None:
        raise ModelIntegrityError(f"model '{name}' is not in the manifest")
    path = Path(models_dir) / entry["file"]
    if not path.exists():
        raise ModelIntegrityError(f"model file {entry['file']} is missing")
    actual = file_sha256(path)
    if actual != entry["sha256"]:
        raise ModelIntegrityError(
            f"model '{name}' does not match its registered hash "
            f"(registered {entry['sha256'][:12]}, found {actual[:12]}); refusing to load")
    return path


def active_versions(models_dir: Path = MODELS_DIR) -> dict[str, dict]:
    """name -> version, short hash and status of every registered model (for page footers)."""
    return {
        name: {"version": e.get("version", "?"), "sha256": e["sha256"],
               "short_hash": e["sha256"][:12], "status": e.get("status", "?")}
        for name, e in load_manifest(models_dir)["models"].items()
    }


def combined_hash(models_dir: Path = MODELS_DIR) -> str:
    """One hash over all registered models, written into every audit entry."""
    versions = active_versions(models_dir)
    if not versions:
        return ""
    blob = json.dumps({k: v["sha256"] for k, v in versions.items()}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()
