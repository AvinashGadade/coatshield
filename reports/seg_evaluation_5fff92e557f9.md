# Boundary finder evaluation

Config `5fff92e557f9` · synthetic validation set, 1000 scans · mean absolute boundary error in depth pixels over columns that both the labels and the graph search call valid (median over scans in brackets)

| Model | Parameters | Scans | Outer, all | Inner, all | Outer, clear coat and SNR >= 25 dB | Inner, clear coat and SNR >= 25 dB | Inner, pigmented and SNR >= 25 dB | Valid columns kept | Dice coating | Thinnest separable film (um) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OCT5k-pretrained, fine-tuned | 2,160,163 | 1000 | 0.56 (0.54) | 4.77 (1.13) | 0.55 (0.54) | 1.34 (0.93) | 8.18 (3.72) | 95% | 0.945 | 2.6 (clear coats) |
| From scratch | 2,160,163 | 1000 | 0.54 (0.54) | 3.74 (0.84) | 0.54 (0.54) | 1.06 (0.72) | 6.66 (1.78) | 99% | 0.964 | 2.0 (clear coats) |

Clear coat: pigment at most 0.15. In pigmented coats the coating-core interface is hidden by scatter, as in the published catalogue where only 6 of 22 commercial coatings were readable; those scans must end undecided rather than be measured.

## Real retinal OCT (OCT5k), held-out patients

Pretrained network on 149 scans from 5 patient volumes that were never used for training (split by patient). Mean absolute boundary error in pixels:

| ILM | OPL-Henle | IS/OS | inner RPE | outer RPE |
| --- | --- | --- | --- | --- |
| 1.70 | 3.60 | 1.25 | 1.43 | 1.64 |

Ablation (ResNet-18 U-Net, transformer hybrid): not run unless listed above.
