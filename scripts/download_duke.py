"""Fallback: download the Duke Chiu 2015 DME retinal OCT dataset.

Licence: research and education only. Never commit or redistribute it
(data/raw is git-ignored). Use only if the OCT5k image download fails.
Writes SHA-256 checksums to data/raw/checksums.json.
"""

from __future__ import annotations

import zipfile

from _common import RAW_DIR, download, file_hash, record_checksum

PAGE = "https://people.duke.edu/~sf59/Chiu_BOE_2014_dataset.htm"
URL = "https://people.duke.edu/~sf59/Datasets/2015_BOE_Chiu2.zip"
DEST = RAW_DIR / "duke_chiu2015"


def main() -> None:
    archive = download(URL, DEST / "2015_BOE_Chiu2.zip")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(DEST)
    mats = sorted(DEST.rglob("*.mat"))
    record_checksum(
        "duke_chiu2015",
        {
            "source": PAGE,
            "licence": "Research and education only; no redistribution",
            "archive": archive.name,
            "archive_sha256": file_hash(archive),
            "n_mat_files": len(mats),
            "mat_sha256": {m.name: file_hash(m) for m in mats},
        },
    )
    print(f"{len(mats)} .mat files in {DEST}")


if __name__ == "__main__":
    main()
