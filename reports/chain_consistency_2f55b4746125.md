# Full chain against the error model

Config `2f55b4746125` · 20,000 pellets, 1 h, 50 objects per minute (3328 objects), coating 10x faster than normal · error model `models/error_model.json`

| | Full chain | Error model |
| --- | --- | --- |
| Objects accepted | 2755 | 2884 |
| Median |reading - truth| of accepted singles (um) | 0.044 | 0.043 |
| C1 stops at step | 50 | 49 |
| C2 stops at step | 48 | 48 |
| C3 stops at step | 56 | 56 |

Corrected d10, full chain minus error model: median absolute difference 0.055 um over 57 time steps (mean +0.013 um).

Target: stop times within one step and d10 within 0.3 um. Result: stop times differ by at most 1 step(s); d10 differs by 0.06 um: **met**.
