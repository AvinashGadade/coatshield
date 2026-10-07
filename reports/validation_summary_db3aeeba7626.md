# Validation summary (full run)

Config hash `db3aeeba7626` (diagnosis settings are not part of this hash; the runs were made under the earlier tag 1cf9e246611c with identical settings) · 2000 grid batches and 140 fault batches at 100,000 pellets · spec d10 >= 12 um

## Headline (window bias m = 3, size-growth exponent k = 1, no hidden selection)

| Stopping rule | Stop (h) | True % below spec | True d10 (um) | Believed d10 (um) | Excess coating (%) |
| --- | --- | --- | --- | --- | --- |
| C0 Gravimetric | 9.75 | 5.6 | 12.68 | nan | 5.9 |
| C1 Raw mean | 8.66 | 16.9 | 11.32 | 12.25 | -5.9 |
| C2 Raw d10 | 8.47 | 20.0 | 11.08 | 12.04 | -8.0 |
| C3 CoatShield | 9.26 | 9.5 | 12.07 | 12.11 | 0.5 |

The raw-d10 rule ships 2.1x the out-of-spec pellets of the CoatShield rule on the same batches.

## Estimator error on d10 (gamma = 0, all grid cells, scored from 2 h)

| Estimator | Median absolute error (um), mean over cells | Worst cell | Mean bias (um) |
| --- | --- | --- | --- |
| raw | 0.64 | 1.49 | -0.06 |
| ipw | 0.47 | 0.97 | -0.49 |
| model | 0.05 | 0.11 | -0.03 |
| hybrid | 0.05 | 0.10 | -0.03 |

## No false claim of bias (m = 0 or k = 0)

True d10 at the raw-d10 stop minus at the CoatShield stop: +0.01 to +0.90 um across those cells. Where it is positive, the raw rule stops late, not early: measurement noise widens the raw sample, so its d10 reads low. The raw rule ships at most 9.6% below spec in these cells.

## Hidden selection (gamma > 0): where the correction degrades

| gamma | Corrected d10 bias (um), mean over cells | Worst cell | CoatShield true % below spec, mean | Worst cell |
| --- | --- | --- | --- | --- |
| 0 | -0.03 | -0.10 (m=4, k=0) | 8.7 | 9.8 |
| 0.5 | +0.08 | +0.09 (m=0, k=0.5) | 11.5 | 16.2 |

The correction assumes the window selects pellets by size only. With gamma > 0 the window also favours pellets that have made more passes than their size predicts; the camera cannot see that, so the corrected d10 reads high by the amount in the table and the CoatShield rule ships more than 10% below spec.

## Fault scenarios (mean over seeds)

| Scenario | C0 below spec % | C1 below spec % | C2 below spec % | C3 below spec % | C0 mean (um) | C1 mean (um) | C2 mean (um) | C3 mean (um) | C3 stopped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | 5.6 | 16.8 | 20.0 | 9.4 | 16.26 | 14.52 | 14.21 | 15.49 | 100% of batches |
| substrate_shift | 1.8 | 17.0 | 19.7 | 9.3 | 17.82 | 14.51 | 14.24 | 15.50 | 100% of batches |
| nozzle_block | 12.2 | 17.0 | 20.4 | 12.2 | 15.06 | 14.50 | 14.17 | 15.06 | 0% of batches |
| over_wetting | 12.5 | 17.4 | 19.9 | 8.7 | 15.02 | 14.46 | 14.22 | 15.60 | 100% of batches |
| spray_drying | 44.4 | 17.2 | 20.5 | 12.2 | 12.52 | 14.48 | 14.16 | 15.06 | 0% of batches |
| maldistribution | 5.7 | 17.0 | 20.0 | 9.3 | 16.26 | 14.52 | 14.22 | 15.52 | 100% of batches |
| window_fouling | 5.6 | 17.0 | 20.5 | 0.4 | 16.26 | 14.50 | 14.16 | 19.82 | 0% of batches |

A rule that never stops within the simulated spray time is scored at the end of the batch. CoatShield holds on purpose while the undecided share is above its limit (window fouling).
