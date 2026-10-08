# Chain calibration (trained boundary finder)

Config `88f396776323` · 4000 objects · error model `error_model.json` sha256 `6fc2bd050a3831af9418ab613d1558b8a66441ca794e1851bc8ffb650af99045`

| Outcome | Share of objects |
| --- | --- |
| measured | 48.2% |
| undecided | 18.9% |
| gated | 17.8% |
| no_signal | 15.1% |

Thickness error of measured single pellets: median +0.02 um, robust spread 0.14 um (all conditions); +0.03 um and 0.09 um with a clean window and SNR of at least 25 dB.

Gate: 0.0% of twins pass as single; 3.8% of single pellets are held back.

Undecided curve (logistic): {"intercept": 2.8312124469684914, "fouling": 5.819223752984437, "pigment": 3.6366058636903946, "snr_per_10db": -2.4543303800514}

Confidence threshold for at most 1% of accepted pellets off by more than 2 um: 0.501 (leaves 28.1% undecided; config has 0.5).
