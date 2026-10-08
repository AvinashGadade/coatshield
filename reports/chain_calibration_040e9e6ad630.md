# Chain calibration (trained boundary finder)

Config `040e9e6ad630` · 4000 objects · error model `error_model.json` sha256 `a55b2114eb121397d8362ad06dbc98e0ccb75b40225cd2c19fd6440a69a70cf6`

| Outcome | Share of objects |
| --- | --- |
| undecided | 38.9% |
| measured | 28.2% |
| gated | 17.8% |
| no_signal | 15.1% |

Thickness error of measured single pellets: median +0.03 um, robust spread 0.11 um (all conditions); +0.03 um and 0.08 um with a clean window and SNR of at least 25 dB.

Gate: 0.0% of twins pass as single; 3.8% of single pellets are held back.

Undecided curve (logistic): {"intercept": 2.806015858505254, "fouling": 2.762474471646638, "pigment": 1.5622096644193593, "snr_per_10db": -1.2229518703995865}

Confidence threshold for at most 1% of accepted pellets off by more than 2 um: 0.348 (leaves 57.9% undecided; config has 0.25).
