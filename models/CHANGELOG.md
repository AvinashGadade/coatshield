# Model change log

A small-scale version of a pre-approved change protocol. Every model in `manifest.json`
has an entry here. A retrained model gets a new version, a re-run of its acceptance tests
and a new entry; the locked evaluation is run once per version by the test owner.

| Date | Model | Version | Change | Acceptance tests | Locked evaluation |
| --- | --- | --- | --- | --- | --- |
| 2026-10-07 | unet | 0.1.0 | First trained boundary finder (compact U-Net, 2,160,163 parameters), trained from scratch on 10,000 synthetic pellet scans on a V100. Chosen over the OCT5k-pretrained, fine-tuned run because it scored better on every validation measure (see reports/seg_evaluation_*.md). Status: locked at v1.0 (2026-10-08). | tests/test_seg.py passed; scripts/evaluate_seg.py: outer surface 0.54 px, inner surface 1.06 px mean (0.72 median) on clear coats at SNR >= 25 dB, thinnest separable film 2.0 um; ONNX matches PyTorch to 1.2e-5 | 2026-10-08, 2,000 scans, once: outer 0.55 px, inner 4.26 px all scans (2.48 px at SNR >= 25 dB), pigmented coats included |
| 2026-10-08 | error_model | 0.1.0 | First chain error model: 4,000 objects through the full chain with unet 0.1.0; undecided threshold tuned to 0.35. Status: locked at v1.0 (2026-10-08). | scripts/calibrate_chain.py: reading error +0.03 um median, 0.12 um spread; no twins leak; 3.8% of singles held back. Consistency check against the full chain: see reports/chain_consistency_*.md | - |
| | gate | - | No trained gate. The gate in use is the classical, rule-based method (no weights). | scripts/tune_gate.py | - |

## Acceptance tests per model

- **unet** (boundary finder): `pytest tests/test_seg.py`; `python scripts/evaluate_seg.py`
  (boundary error at SNR >= 25 dB at most 1 pixel on synthetic validation scans; thinnest
  separable film recorded); ONNX matches PyTorch and repeated CPU runs are bit-identical.
- **gate** (if a CNN replaces the classical method): `python scripts/tune_gate.py` (twin
  recall >= 95% with false twins <= 2%, and better than the classical row).
- **error_model**: `python scripts/calibrate_chain.py`, then the consistency check of the
  full chain against the error model on a reduced batch.
