# Validation summary (full run)

Config hash `f5ba3ea2e75f` · 2000 grid batches and 140 fault batches at 100,000 pellets · spec d10 >= 12 um

## Headline (window bias m = 3, size-growth exponent k = 1, no hidden selection)

| Stopping rule | Stop (h) | True % below spec | True d10 (um) | Believed d10 (um) | Excess coating (%) |
| --- | --- | --- | --- | --- | --- |
| C0 Gravimetric | 9.75 | 5.6 | 12.68 | nan | 5.9 |
| C1 Raw mean | 8.66 | 16.9 | 11.32 | 12.50 | -6.0 |
| C2 Raw d10 | 8.31 | 23.1 | 10.87 | 12.03 | -9.8 |
| C3 CoatShield | 9.24 | 9.6 | 12.05 | 12.10 | 0.3 |

The raw-d10 rule ships 2.4x the out-of-spec pellets of the CoatShield rule on the same batches.

## Estimator error on d10 (gamma = 0, all grid cells, scored from 2 h)

| Estimator | Median absolute error (um), mean over cells | Worst cell | Mean bias (um) |
| --- | --- | --- | --- |
| raw | 0.47 | 1.69 | +0.47 |
| ipw | 0.06 | 0.20 | +0.07 |
| model | 0.03 | 0.03 | +0.02 |
| hybrid | 0.03 | 0.04 | +0.02 |

## No false claim of bias (m = 0 or k = 0)

True d10 at the raw-d10 stop minus at the CoatShield stop: -0.15 to -0.00 um across those cells. The two rules stop at almost the same coat; the small negative gap is the margin CoatShield adds by waiting for 95% confidence. The raw rule ships at most 12.1% below spec in these cells.

## Hidden selection (gamma > 0): where the correction degrades

| gamma | Corrected d10 bias (um), mean over cells | Worst cell | CoatShield true % below spec, mean | Worst cell |
| --- | --- | --- | --- | --- |
| 0 | +0.02 | +0.03 (m=4, k=1.5) | 9.4 | 9.7 |
| 0.5 | +0.12 | +0.15 (m=4, k=0) | 12.8 | 19.8 |

The correction assumes the window selects pellets by size only. With gamma > 0 the window also favours pellets that have made more passes than their size predicts; the camera cannot see that, so the corrected d10 reads high by the amount in the table and the CoatShield rule ships more than 10% below spec.

## Fault scenarios (mean over seeds)

| Scenario | C0 below spec % | C1 below spec % | C2 below spec % | C3 below spec % | C0 mean (um) | C1 mean (um) | C2 mean (um) | C3 mean (um) | C3 stopped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | 5.6 | 16.9 | 23.3 | 9.6 | 16.26 | 14.52 | 13.93 | 15.45 | 100% of batches |
| substrate_shift | 1.8 | 16.9 | 23.0 | 9.6 | 17.82 | 14.51 | 13.95 | 15.45 | 100% of batches |
| nozzle_block | 12.2 | 17.0 | 23.3 | 12.2 | 15.06 | 14.50 | 13.92 | 15.06 | 0% of batches |
| over_wetting | 12.5 | 17.0 | 23.0 | 9.7 | 15.02 | 14.51 | 13.95 | 15.43 | 100% of batches |
| spray_drying | 44.4 | 17.2 | 23.1 | 12.2 | 12.52 | 14.48 | 13.94 | 15.06 | 0% of batches |
| maldistribution | 5.7 | 17.1 | 23.2 | 9.6 | 16.26 | 14.51 | 13.95 | 15.47 | 100% of batches |
| window_fouling | 5.6 | 16.9 | 22.7 | 0.4 | 16.26 | 14.52 | 13.98 | 19.82 | 0% of batches |

A rule that never stops within the simulated spray time is scored at the end of the batch. CoatShield holds on purpose while the undecided share is above its limit (window fouling).

Measurement: fitted chain error model `models/error_model.json` (estimator noise 0.07 um), not the placeholder.
