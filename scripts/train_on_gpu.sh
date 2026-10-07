#!/usr/bin/env bash
# Train the CoatShield boundary finder on a GPU server (for example a college V100 machine).
#
#   bash scripts/train_on_gpu.sh /path/to/folder/with/the/zips
#
# The folder must hold coatshield_synthetic.zip and oct5k.zip (made on the laptop by
# scripts/package_for_kaggle.py). Run this from the unpacked code folder (coatshield/).
# It creates a Python environment, unpacks the data, pretrains on OCT5k, fine-tunes on the
# synthetic pellets, trains from scratch for comparison, evaluates, exports ONNX and writes
# coatshield_models.zip next to the zips. Safe to run again: finished steps are skipped.
set -euo pipefail

ZIPS="${1:?usage: bash scripts/train_on_gpu.sh /path/to/folder/with/the/zips}"
ZIPS="$(cd "$ZIPS" && pwd)"
PYTHON="${PYTHON:-python3}"
cd "$(dirname "$0")/.."

for f in coatshield_synthetic.zip oct5k.zip; do
  [ -f "$ZIPS/$f" ] || { echo "missing $ZIPS/$f"; exit 1; }
done
"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' || {
  echo "Python 3.11 or newer is needed (found: $("$PYTHON" --version 2>&1))."
  echo "Load a newer module or use conda, then run:  PYTHON=/path/to/python3.11 bash $0 $ZIPS"
  exit 1
}

echo "== 1/6 Python environment"
if [ ! -x .venv/bin/python ]; then
  "$PYTHON" -m venv .venv
fi
.venv/bin/pip install -q --upgrade pip
# requirements.txt asks for CPU builds of torch (for the laptop); here the GPU build is wanted.
grep -v -E '^(--extra-index-url)' requirements.txt > .requirements-gpu.txt
.venv/bin/pip install -q -r .requirements-gpu.txt
.venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("No GPU visible to PyTorch. Check `nvidia-smi`; if the driver is old, "
                     "install a matching build, e.g.\n  .venv/bin/pip install torch torchvision "
                     "--index-url https://download.pytorch.org/whl/cu118")
print("GPU:", torch.cuda.get_device_name(0))
torch.zeros(1).cuda()  # fails here if this torch build does not support the card
PY

echo "== 2/6 Data"
[ -f data/synthetic/train/images.npy ] || unzip -q -o "$ZIPS/coatshield_synthetic.zip" -d data
[ -d data/raw/oct5k/OCT5k/Images ] || { mkdir -p data/raw/oct5k; unzip -q -o "$ZIPS/oct5k.zip" -d data/raw/oct5k; }
ls data/synthetic/train/images.npy data/synthetic/val/images.npy >/dev/null

WORKERS="${WORKERS:-4}"
run_stage() {  # stage name, checkpoint file
  if [ -f "models/checkpoints/$2.pt" ]; then
    echo "   $1: checkpoint exists, skipping"
  else
    .venv/bin/python -m coatshield.seg.train --stage "$1" --device cuda --workers "$WORKERS" \
      2>&1 | tee "reports/train_$1.log"
  fi
}
mkdir -p reports models/checkpoints
echo "== 3/6 Pretrain on OCT5k";              run_stage pretrain unet_oct5k
echo "== 4/6 Fine-tune on synthetic pellets"; run_stage finetune unet_pellets
echo "== 5/6 Same network from scratch";      run_stage scratch  unet_pellets_scratch

echo "== 6/6 Evaluate, export ONNX, pack the results"
.venv/bin/python scripts/evaluate_seg.py
.venv/bin/python scripts/export_onnx.py --checkpoint unet_pellets --name unet
rm -f "$ZIPS/coatshield_models.zip"
zip -q -r "$ZIPS/coatshield_models.zip" models reports/seg_* reports/train_*.log
echo "Done. Copy $ZIPS/coatshield_models.zip back to the laptop."
