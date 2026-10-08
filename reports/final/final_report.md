# CoatShield final report

Built 2026-10-08T11:40:54+00:00 from the files in `reports/`. Every number names its source file; file names carry the hash of the configuration that produced them. Demonstration on simulated and public data, not a validated GMP system.

## 1. Headline with the placeholder measurement (1 um noise)

Source: `validation_cells_db3aeeba7626.csv`

| Stopping rule | Stops at (h) | Truly below spec | True d10 (um) |
| --- | --- | --- | --- |
| C0 Gravimetric | 9.75 | 5.6% | 12.68 |
| C1 Raw mean | 8.66 | 16.9% | 11.32 |
| C2 Raw d10 | 8.47 | 20.0% | 11.08 |
| C3 CoatShield | 9.26 | 9.5% | 12.07 |

- The raw-d10 rule ships **2.1x** the out-of-spec pellets of the CoatShield rule.
- Corrected d10 error: 0.05 um median absolute, 0.10 um in the worst grid cell (raw sample: 0.64 um).
- Strongest assumptions in the grid: the raw rule ships 28% below spec.
- Hidden selection (gamma = 0.5): corrected d10 reads +0.08 um; CoatShield ships 11.5% below spec on average, 16.2% in the worst cell.

## 2. Headline with the measurement chain's fitted error model

Source: `validation_cells_f5ba3ea2e75f.csv`

| Stopping rule | Stops at (h) | Truly below spec | True d10 (um) |
| --- | --- | --- | --- |
| C0 Gravimetric | 9.75 | 5.6% | 12.68 |
| C1 Raw mean | 8.66 | 16.9% | 11.32 |
| C2 Raw d10 | 8.31 | 23.1% | 10.87 |
| C3 CoatShield | 9.24 | 9.6% | 12.05 |

- The raw-d10 rule ships **2.4x** the out-of-spec pellets of the CoatShield rule.
- Corrected d10 error: 0.03 um median absolute, 0.04 um in the worst grid cell (raw sample: 0.47 um).
- Strongest assumptions in the grid: the raw rule ships 30% below spec.
- Hidden selection (gamma = 0.5): corrected d10 reads +0.12 um; CoatShield ships 12.8% below spec on average, 19.8% in the worst cell.

The ratio is 2.1x with the placeholder and 2.4x with the chain's error model: the headline **holds** under the measurement error of the simulated chain.

## 3. Batch twin (default scenario)

Source: `app/assets/bundles/bceb510b2581` (200,000 simulated pellets for 96 million real ones)

- Large-to-small growth ratio 1.59 (published 1.08-1.81).
- Size-adjusted variability decays with slope -0.50 on log-log axes (the published law is -0.5).
- Weight gain at 16 um mean coat: 11.3%.
- Share of the batch the window has measured when CoatShield stops: 0.28%.

## 4. Measurement modules

**Boundary finder** (`seg_evaluation_5fff92e557f9.md`, model `unet` 0.1.0, 2,160,163 parameters): outer surface 0.54 px and inner surface 1.06 px mean error (median 0.72) on clear coats at SNR >= 25 dB; one pixel is 0.36 um of optical path. Pigmented coats are not readable.

**Refractive index** (`solver_index_e94e303c4e9b.csv`, SNR 35 dB, surfaces = truth + 1 px): error in n -0.008 (A reflectance), -0.009 (B camera-OCT fusion), +0.008 (C at-line anchor); a 10% reflector calibration error moves method A by +0.060.

**Gate** (`gate_comparison_64419312f96a.csv`, classical method, held-out synthetic images): twin recall 100.0%, false twins 0.0%, 96.3% of single pellets pass.

**Full chain** (`chain_calibration_88f396776323.md`, error model file sha256 `236b8dbf43fb`): reading spread 0.07 um at the operating point; undecided share 0.4% with a clean window and 56% fully fouled; 0.0% of twins pass the gate; 3.8% of single pellets are held back.

**Chain against error model** (`chain_consistency_2f55b4746125.md`): Target: stop times within one step and d10 within 0.3 um. Result: stop times differ by at most 1 step(s); d10 differs by 0.06 um: **met**.

## 5. Models in force

| Model | Version | SHA-256 | Status |
| --- | --- | --- | --- |
| error_model | 0.1.0 | `236b8dbf43fb36f5f3d87f91329719afdfeafccc2cc260d582e675bae87a8110` | locked |
| unet | 0.1.0 | `2abe0b6b30a76bca08f29f1373c6fd342d8d93656b09d3cc727cae563b51b947` | locked |

## 6. Locked test set

`locked_eval_unet_2abe0b6b30a7.json`: 2000 scans the model never saw, run once by Avinash at 2026-10-08T11:34:36+00:00 on model `2abe0b6b30a7`.

| Surface | All scans (px) | SNR >= 25 dB (px) |
| --- | --- | --- |
| Outer (air-coating) | 0.55 | 0.54 |
| Inner (coating-core) | 4.26 | 2.48 |

These figures include the pigmented scans (a fifth of the set), in which the inner surface is hidden; the locked evaluation did not split clear from pigmented coats, and it is not rerun to add that split. On the validation set the same mixed figures were 0.54 and 3.74 px (all scans); clear coats alone gave 1.06 px for the inner surface.

Thinnest separable film: not determined on the locked set, because the pigmented scans fail to separate at every thickness and were not excluded. On clear validation coats it was 2.0 um.
