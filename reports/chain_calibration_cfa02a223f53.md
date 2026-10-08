# Chain calibration (trained boundary finder)

Config `cfa02a223f53` · 4000 objects · error model `error_model.json` sha256 `f07ce192fa45e264716439f889d27a11f4d51a481f3808b8c9e380ef8762ec5b`

| Outcome | Share of objects |
| --- | --- |
| measured | 36.4% |
| undecided | 30.6% |
| gated | 17.8% |
| no_signal | 15.1% |

Thickness error of measured single pellets: median +0.03 um, robust spread 0.12 um (all conditions); +0.03 um and 0.08 um with a clean window and SNR of at least 25 dB.

Gate: 0.0% of twins pass as single; 3.8% of single pellets are held back.

Undecided curve (logistic): {"intercept": 2.418713547034926, "fouling": 3.102829486848701, "pigment": 1.6241195389580312, "snr_per_10db": -1.3446237151678027}

Confidence threshold for at most 1% of accepted pellets off by more than 2 um: 0.347 (leaves 45.7% undecided; config has 0.25).
