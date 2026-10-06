"""Shared helpers for the data scripts: downloads and the checksum registry."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RAW_DIR = REPO_ROOT / "data" / "raw"
CHECKSUMS = RAW_DIR / "checksums.json"
REPORTS_DIR = REPO_ROOT / "reports"


def file_hash(path: Path, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_hash(root: Path, pattern: str = "**/*") -> tuple[str, int]:
    """One SHA-256 over every file under root (sorted relative paths + contents)."""
    h = hashlib.sha256()
    files = sorted(p for p in root.glob(pattern) if p.is_file())
    for p in files:
        h.update(p.relative_to(root).as_posix().encode())
        h.update(bytes.fromhex(file_hash(p)))
    return h.hexdigest(), len(files)


def record_checksum(key: str, entry: dict) -> None:
    """Add or replace one entry in data/raw/checksums.json."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(CHECKSUMS.read_text()) if CHECKSUMS.exists() else {}
    data[key] = entry
    CHECKSUMS.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def download(url: str, dest: Path, expected_md5: str | None = None, timeout: int = 60) -> Path:
    """Download url to dest (resuming a partial file), verify md5 if given."""
    import requests
    from tqdm import tqdm

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (expected_md5 is None or file_hash(dest, "md5") == expected_md5):
        print(f"[skip] {dest.name} already on disk")
        return dest

    part = dest.with_suffix(dest.suffix + ".part")
    start = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    with requests.get(url, stream=True, timeout=timeout, headers=headers) as resp:
        if start and resp.status_code != 206:
            start = 0
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0)) + start
        with (
            open(part, "ab" if start else "wb") as fh,
            tqdm(total=total, initial=start, unit="B", unit_scale=True, desc=dest.name) as bar,
        ):
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))
    part.rename(dest)
    if expected_md5 is not None:
        got = file_hash(dest, "md5")
        if got != expected_md5:
            raise RuntimeError(f"{dest.name}: md5 {got} does not match expected {expected_md5}")
    return dest
