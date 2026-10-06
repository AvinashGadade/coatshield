"""Download OCT5k: labels from the UCL repository, scan images from the original source.

Labels (CC0 on the UCL record; research use per the paper) come as one zip.
The scan images are not in it: the dataset's own notebook
(Scripts/Prepare_Images_Download.ipynb) fetches the Rasti et al. macular OCT archive
and maps it onto the label tree with Scripts/paths/manual_paths.csv. This script
follows the same steps for the 1,672 manually graded scans (resize to 512 x 512,
save as PNG), which is what the U-Net pretraining uses.

The image archive is password-protected by its owner. The password is published on
the owner's dataset page (PASSWORD_PAGE below), with the request to cite Rasti et al.,
IEEE TMI 37(4), 2018. Pass it with --password or the OCT5K_IMAGES_PASSWORD variable.

Writes SHA-256 checksums to data/raw/checksums.json. If the image step fails,
run scripts/download_duke.py instead.
"""

from __future__ import annotations

import argparse
import csv
import os
import zipfile
from pathlib import Path

import cv2
import numpy as np
from _common import RAW_DIR, download, file_hash, record_checksum, tree_hash

LABELS_URL = "https://ndownloader.figshare.com/files/44436359"
LABELS_MD5 = "e3480fb94728c8481ad20ebd54ed8e7e"  # from the UCL record's file listing
LABELS_DOI = "10.5522/04/22128671"
# The notebook resolves http://cutt.ly/wGOhQeK to this Google Drive file id.
IMAGES_SHORTLINK = "http://cutt.ly/wGOhQeK"
IMAGES_DRIVE_ID = "1y7yKlDR4sP8bJ-_updZFBtHD6Yq_FO03"
IMAGES_ARCHIVE = "Macular-Dataset-R.Rasti_old.rar"
SOURCE_PREFIX = "Macular-Dataset-R.Rasti_old/"
PASSWORD_PAGE = (
    "https://sites.google.com/site/hosseinrabbanikhorasgani/available-datasets/"
    "dataset-for-oct-classification-50-normal-48-amd-50-dme"
)

DEST = RAW_DIR / "oct5k"
ROOT = DEST / "OCT5k"
IMAGE_SIZE = (512, 512)  # the dataset's notebook resizes manual-grading scans to 512 x 512


def _drive_url(file_id: str) -> str:
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def _resolve_drive_id() -> str:
    """Follow the dataset's short link; fall back to the id recorded above."""
    import requests

    try:
        resp = requests.head(IMAGES_SHORTLINK, allow_redirects=True, timeout=30)
        parts = resp.url.split("/")
        if "drive.google.com" in resp.url and len(parts) > 5:
            return parts[5]
    except requests.RequestException:
        pass
    return IMAGES_DRIVE_ID


def _norm(path: str) -> str:
    """Archive member or CSV source path -> key relative to the Rasti dataset root."""
    path = path.replace("\\", "/").replace("//", "/").lstrip("./")
    return path.split(SOURCE_PREFIX, 1)[-1].lower()


def manual_path_map() -> dict[str, Path]:
    """Source key in the Rasti archive -> target PNG under OCT5k/Images/Images_Manual."""
    mapping = {}
    with open(ROOT / "Scripts" / "paths" / "manual_paths.csv", newline="") as fh:
        for to_path, from_path in csv.reader(fh):
            mapping[_norm(from_path)] = ROOT / to_path.replace("../", "", 1)
    return mapping


def extract_manual_images(archive: Path, password: str) -> int:
    """Write the scans named in manual_paths.csv as 512 x 512 PNGs.

    The RAR holds four password-protected zip parts; each is unpacked to a temporary
    file, read, and removed, so only one part is on disk at a time.
    """
    import libarchive

    wanted = manual_path_map()
    done = 0
    with libarchive.file_reader(str(archive)) as reader:
        for entry in reader:
            if not entry.pathname.lower().endswith(".zip"):
                continue
            part = archive.with_name(Path(entry.pathname).name)
            with open(part, "wb") as fh:
                for block in entry.get_blocks():
                    fh.write(block)
            try:
                done += _extract_part(part, password, wanted)
            finally:
                part.unlink()
    missing = len(wanted) - done
    if missing:
        raise RuntimeError(f"{missing} of {len(wanted)} scans were not found in {archive.name}")
    return done


def _extract_part(part: Path, password: str, wanted: dict[str, Path]) -> int:
    from tqdm import tqdm

    done = 0
    with zipfile.ZipFile(part) as zf:
        members = [i for i in zf.infolist() if _norm(i.filename) in wanted]
        for info in tqdm(members, desc=part.name, unit="scan"):
            image = cv2.imdecode(
                np.frombuffer(zf.read(info, pwd=password.encode()), np.uint8),
                cv2.IMREAD_UNCHANGED,
            )
            if image is None:
                raise RuntimeError(f"Could not decode {info.filename}")
            target = wanted[_norm(info.filename)]
            target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target), cv2.resize(image, dsize=IMAGE_SIZE))
            done += 1
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--labels-only", action="store_true", help="skip the scan images")
    parser.add_argument("--keep-archive", action="store_true", help="keep the 376 MB image RAR")
    parser.add_argument("--password", default=os.environ.get("OCT5K_IMAGES_PASSWORD"),
                        help="image archive password, published at " + PASSWORD_PAGE)
    args = parser.parse_args()

    labels = download(LABELS_URL, DEST / "OCT5k.zip", expected_md5=LABELS_MD5)
    if not (ROOT / "README.txt").exists():
        with zipfile.ZipFile(labels) as zf:
            zf.extractall(DEST)
    masks_digest, n_masks = tree_hash(ROOT / "Masks" / "Masks_Manual", "**/*.png")
    entry = {
        "source": f"https://doi.org/{LABELS_DOI}",
        "licence": "CC0 on the UCL record; research use per the OCT5k paper",
        "labels_archive": labels.name,
        "labels_archive_sha256": file_hash(labels),
        "manual_masks_tree_sha256": masks_digest,
        "n_manual_masks": n_masks,
    }
    record_checksum("oct5k", entry)
    print(f"Labels ready: {n_masks} manual masks")
    if args.labels_only:
        return
    if not args.password:
        raise SystemExit(f"The scan images need a password; it is published at {PASSWORD_PAGE}")

    archive = download(_drive_url(_resolve_drive_id()), DEST / IMAGES_ARCHIVE)
    archive_sha = file_hash(archive)
    n_images = extract_manual_images(archive, args.password)
    images_digest, n_files = tree_hash(ROOT / "Images" / "Images_Manual", "**/*.png")
    entry.update(
        {
            "images_source": "Rasti et al. 2018 macular OCT (Heidelberg Spectralis), via "
            + IMAGES_SHORTLINK,
            "images_citation": "Rasti, Rabbani, Mehri, Hajizadeh, IEEE TMI 37(4):1024-1034, 2018",
            "images_archive": IMAGES_ARCHIVE,
            "images_archive_sha256": archive_sha,
            "manual_images_tree_sha256": images_digest,
            "n_manual_images": n_files,
        }
    )
    record_checksum("oct5k", entry)
    if not args.keep_archive:
        archive.unlink()
    print(f"Images ready: {n_images} scans at {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}")


if __name__ == "__main__":
    main()
