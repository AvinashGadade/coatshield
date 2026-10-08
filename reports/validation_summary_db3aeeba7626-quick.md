# Validation summary (quick run)

Config hash `db3aeeba7626-quick` · 32 grid batches and 28 fault batches at 100,000 pellets · spec d10 >= 12 um

## Headline (window bias m = 3, size-growth exponent k = 1, no hidden selection)

| Stopping rule | Stop (h) | True % below spec | True d10 (um) | Believed d10 (um) | Excess coating (%) |
| --- | --- | --- | --- | --- | --- |
| C0 Gravimetric | 9.75 | 5.6 | 12.69 | nan | 5.9 |
| C1 Raw mean | 8.67 | 16.6 | 11.34 | 12.31 | -5.8 |
| C2 Raw d10 | 8.51 | 19.3 | 11.13 | 12.03 | -7.6 |
| C3 CoatShield | 9.22 | 9.7 | 12.03 | 12.11 | 0.2 |

The raw-d10 rule ships 2.0x the out-of-spec pellets of the CoatShield rule on the same batches.

## Estimator error on d10 (gamma = 0, all grid cells, scored from 2 h)

| Estimator | Median absolute error (um), mean over cells | Worst cell | Mean bias (um) |
| --- | --- | --- | --- |
| raw | 0.72 | 0.97 | -0.41 |
| ipw | 0.62 | 0.97 | -0.64 |
| model | 0.06 | 0.09 | -0.04 |
| hybrid | 0.06 | 0.09 | -0.04 |

## No false claim of bias (m = 0 or k = 0)

True d10 at the raw-d10 stop minus at the CoatShield stop: +0.19 to +0.86 um across those cells. Where it is positive, the raw rule stops late, not early: measurement noise widens the raw sample, so its d10 reads low. The raw rule ships at most 8.3% below spec in these cells.

## Hidden selection (gamma > 0): where the correction degrades

| gamma | Corrected d10 bias (um), mean over cells | Worst cell | CoatShield true % below spec, mean | Worst cell |
| --- | --- | --- | --- | --- |
| 0 | -0.04 | -0.08 (m=3, k=0) | 8.0 | 9.7 |
| 0.5 | +0.08 | +0.09 (m=3, k=1) | 11.5 | 15.5 |

The correction assumes the window selects pellets by size only. With gamma > 0 the window also favours pellets that have made more passes than their size predicts; the camera cannot see that, so the corrected d10 reads high by the amount in the table and the CoatShield rule ships more than 10% below spec.

## Fault scenarios (mean over seeds)

| Scenario | C0 below spec % | C1 below spec % | C2 below spec % | C3 below spec % | C0 mean (um) | C1 mean (um) | C2 mean (um) | C3 mean (um) | C3 stopped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | 5.6 | 16.6 | 19.3 | 9.7 | 16.27 | 14.55 | 14.28 | 15.43 | 100% of batches |
| substrate_shift | 1.8 | 16.9 | 19.1 | 9.1 | 17.83 | 14.52 | 14.30 | 15.54 | 100% of batches |
| nozzle_block | 12.1 | 17.3 | 19.8 | 12.1 | 15.07 | 14.47 | 14.23 | 15.07 | 0% of batches |
| over_wetting | 12.5 | 17.1 | 20.9 | 8.8 | 15.03 | 14.49 | 14.14 | 15.59 | 100% of batches |
| spray_drying | 44.2 | 17.6 | 20.5 | 12.1 | 12.52 | 14.44 | 14.16 | 15.07 | 0% of batches |
| maldistribution | 5.7 | 16.9 | 19.9 | 9.3 | 16.27 | 14.53 | 14.24 | 15.53 | 100% of batches |
| window_fouling | 5.6 | 17.5 | 19.0 | 0.4 | 16.27 | 14.46 | 14.31 | 19.83 | 0% of batches |

A rule that never stops within the simulated spray time is scored at the end of the batch. CoatShield holds on purpose while the undecided share is above its limit (window fouling).
