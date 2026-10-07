# Model change log

A small-scale version of a pre-approved change protocol. Every model in `manifest.json`
has an entry here. A retrained model gets a new version, a re-run of its acceptance tests
and a new entry; the locked evaluation is run once per version by the test owner.

| Date | Model | Version | Change | Acceptance tests | Locked evaluation |
| --- | --- | --- | --- | --- | --- |
| (none yet) | | | No trained model has been registered. The gate in use is the classical, rule-based method (no weights). | | |

## Acceptance tests per model

- **unet** (boundary finder): `pytest tests/test_seg.py`; `python scripts/evaluate_seg.py`
  (boundary error at SNR >= 25 dB at most 1 pixel on synthetic validation scans; thinnest
  separable film recorded); ONNX matches PyTorch and repeated CPU runs are bit-identical.
- **gate** (if a CNN replaces the classical method): `python scripts/tune_gate.py` (twin
  recall >= 95% with false twins <= 2%, and better than the classical row).
- **error_model**: `python scripts/calibrate_chain.py`, then the consistency check of the
  full chain against the error model on a reduced batch.
