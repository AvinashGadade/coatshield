"""Pack what the Kaggle training notebook needs into dist/, ready to upload as private datasets.

No code hosting is involved: the code travels as a zip of the committed files.
  dist/coatshield_code.zip       -> Kaggle dataset "coatshield-code"
  dist/coatshield_synthetic.zip  -> Kaggle dataset "coatshield-synthetic" (train/ and val/)
  dist/oct5k.zip                 -> Kaggle dataset "oct5k" (manual images and masks only)
The locked test set is never packed.
"""

from __future__ import annotations

import argparse
import subprocess
import zipfile

from _common import RAW_DIR, REPO_ROOT

DIST = REPO_ROOT / "dist"


def _zip_tree(root, archive, keep) -> int:
    n = 0
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as zf:
        for path in sorted(root.rglob("*")):
            if path.is_file() and keep(path.relative_to(root)):
                zf.write(path, path.relative_to(root.parent))
                n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--skip-data", action="store_true", help="pack the code only")
    args = parser.parse_args()
    DIST.mkdir(exist_ok=True)

    code = DIST / "coatshield_code.zip"
    subprocess.run(["git", "archive", "--format=zip", "--prefix=coatshield/", "-o", str(code),
                    "HEAD"], cwd=REPO_ROOT, check=True)
    print(f"{code.name}: {code.stat().st_size / 1e6:.1f} MB (committed files at HEAD)")
    if args.skip_data:
        return

    synthetic = REPO_ROOT / "data" / "synthetic"
    if (synthetic / "train" / "manifest.json").exists():
        out = DIST / "coatshield_synthetic.zip"
        n = _zip_tree(synthetic, out, lambda rel: rel.parts[0] in ("train", "val"))
        print(f"{out.name}: {n} files, {out.stat().st_size / 1e6:.0f} MB")
    else:
        print("synthetic set not built yet: run scripts/make_synthetic_oct.py first")

    oct5k = RAW_DIR / "oct5k" / "OCT5k"
    if (oct5k / "Images" / "Images_Manual").exists():
        out = DIST / "oct5k.zip"
        wanted = ("Images/Images_Manual", "Masks/Masks_Manual/Grading_1")
        n = _zip_tree(oct5k, out, lambda rel: rel.as_posix().startswith(wanted))
        print(f"{out.name}: {n} files, {out.stat().st_size / 1e6:.0f} MB")
    else:
        print("OCT5k images not downloaded: run scripts/download_oct5k.py first")


if __name__ == "__main__":
    main()
