# Agglomerate gate: method comparison

Config `4e65c7148e86` · thresholds tuned on 2400 images, scored on 2400 held-out synthetic images, 8 balanced classes

| Method | Twin recall | False twins | Twins leaking as single | Accuracy (8 classes) | ECE | CPU ms per image |
| --- | --- | --- | --- | --- | --- | --- |
| Classical (solidity < 0.9 or defect > 0.04) | 100.0% | 0.0% | 0.0% | 94.5% | 0.087 | 9.9 |
| Small CNN | not run: no trained checkpoint (needs a GPU session; `python -m coatshield.gate.train`) | | | | | |

Target: twin recall >= 95% with false twins <= 2%: **met** by the classical method.

Chosen method: classical. 96.3% of single pellets pass to OCT; 0.00% of all other objects pass.

Limits: the images are synthetic dark-field silhouettes; how real fused pellets look and misread is unknown, so the gate's benefit is shown, not quantified.

## Confusion matrix (rows: true class)

| | single | twin | touching | partial | defocused | fouled | empty | fines |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single | 289 | 0 | 0 | 11 | 0 | 0 | 0 | 0 |
| twin | 0 | 289 | 11 | 0 | 0 | 0 | 0 | 0 |
| touching | 0 | 80 | 220 | 0 | 0 | 0 | 0 | 0 |
| partial | 0 | 0 | 0 | 300 | 0 | 0 | 0 | 0 |
| defocused | 0 | 0 | 0 | 0 | 300 | 0 | 0 | 0 |
| fouled | 0 | 0 | 0 | 0 | 0 | 300 | 0 | 0 |
| empty | 0 | 0 | 0 | 0 | 0 | 0 | 300 | 0 |
| fines | 0 | 2 | 4 | 23 | 0 | 0 | 0 | 271 |
