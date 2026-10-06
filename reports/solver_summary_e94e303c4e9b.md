# Solver validation

Config `e94e303c4e9b` · 132 of 132 scans solved · surfaces = truth + 1 px localisation noise · true n = 1.48

## Refractive index at SNR 35 dB

| Method | Error in n |
| --- | --- |
| A reflectance, pooled over 12 pellets | -0.0083 |
| A reflectance, one pellet (mean absolute) | 0.0085 |
| A with reflector calibration +10% / -10% | +0.0514 / -0.0653 |
| A ratio variant (known core index), pooled | -0.0085 |
| B camera-OCT fusion (2500 camera readings) | -0.0092 |
| C at-line anchor (30 pellets, microscopy sd 0.5 um) | +0.0080 |

## Thickness, true index, 4.5 to 30 um

Mean absolute error per pellet: 0.04 um (worst thickness: 0.06 um).  
A-scans beyond 10 degrees: 0.30 um mean error without the refraction correction, 0.02 um with it.  
Assuming n = 1.5 instead of 1.48: thickness reads -1.3% (measured -1.3%).

Method A is precise per pellet but only as accurate as the reflector calibration; methods B and C carry no reflectance calibration.
