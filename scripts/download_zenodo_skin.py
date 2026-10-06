"""Download the in-vivo skin OCT dataset (Zenodo record 18095266, CC BY 4.0).

Used only to compare speckle statistics with the synthetic generator (Phase 4).
Writes SHA-256 checksums to data/raw/checksums.json.
"""

from __future__ import annotations

import zipfile

import numpy as np
from _common import RAW_DIR, download, file_hash, record_checksum, tree_hash

RECORD = "18095266"
URL = f"https://zenodo.org/api/records/{RECORD}/files/DATASET.zip/content"
MD5 = "6c461f35a74b768b2960e0391687c10f"  # from the Zenodo record's file listing
DEST = RAW_DIR / "zenodo_skin"


def load_scans(root=DEST) -> list[np.ndarray]:
    """Every scan in the dataset as a 2D array."""
    return [np.load(p) for p in sorted(root.rglob("*.npy"))]


def main() -> None:
    archive = download(URL, DEST / "DATASET.zip", expected_md5=MD5)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(DEST)
    scans = load_scans()
    shapes = sorted({s.shape for s in scans})
    dtypes = sorted({str(s.dtype) for s in scans})
    digest, n_files = tree_hash(DEST, "**/*.npy")
    record_checksum(
        "zenodo_skin",
        {
            "source": f"https://zenodo.org/records/{RECORD}",
            "licence": "CC BY 4.0",
            "archive": archive.name,
            "archive_sha256": file_hash(archive),
            "n_scans": len(scans),
            "shapes": [list(s) for s in shapes],
            "dtypes": dtypes,
            "scans_tree_sha256": digest,
            "n_files": n_files,
        },
    )
    print(f"{len(scans)} scans, shapes {shapes}, dtypes {dtypes}")


if __name__ == "__main__":
    main()
