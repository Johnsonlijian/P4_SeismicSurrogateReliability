# R34 upgrade results: filter value, shift-aware conformal, vacuous-pass guard

Date: 2026-06-10. All results derive from cached prediction/decision exports; no model retraining.

## U2 Filter decision value (threshold x cost surface, N=500 trace)

Policies: RMSE-best model; filter-constrained minimum-loss model (beta_FS,cons >= 2.5); oracle minimum-loss model. Losses are event-equal expected screening losses.

| protocol | cells | infeasible | mean regret RMSE-pick | mean regret filter-pick | worst regret RMSE-pick | worst regret filter-pick | worst-cell loss RMSE | worst-cell loss filter | max P_FS RMSE-pick | max P_FS filter-pick |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| event-disjoint target | 49 | 0 | 0.0219 | 0.0089 | 0.3031 | 0.0958 | 0.488 | 0.273 | 0.0047 | 0.0020 |
| main event-held-out | 49 | 0 | 0.0219 | 0.0001 | 0.1850 | 0.0028 | 0.402 | 0.225 | 0.0037 | 0.0031 |

- High-cost cells (C >= 25), event-disjoint target: RMSE-pick mean regret 0.0490 vs filter-pick 0.0035; worst-case loss 0.488 vs 0.273.
- High-cost cells (C >= 25), main event-held-out: RMSE-pick mean regret 0.0486 vs filter-pick 0.0000; worst-case loss 0.402 vs 0.225.

## U1 Shift-aware conformal baseline (event-disjoint, N=2000, level 0.90, medians over 10 reps)

| model | method | event-equal coverage | P_FS at 1% IDR | beta_FS,cons | P_FU | median q (log10) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Ridge direct | standard split | 0.509 | 0.0016 | 2.65 | 0.122 | 0.300 |
| Ridge direct | event-weighted | 0.438 | 0.0017 | 2.65 | 0.117 | 0.276 |
| Ridge direct | Mondrian PGA-tercile | 0.492 | 0.0016 | 2.65 | 0.122 | 0.297 |
| LGBM direct | standard split | 0.556 | 0.0049 | 2.37 | 0.012 | 0.121 |
| LGBM direct | event-weighted | 0.559 | 0.0050 | 2.36 | 0.012 | 0.114 |
| LGBM direct | Mondrian PGA-tercile | 0.543 | 0.0050 | 2.37 | 0.012 | 0.113 |
| XGB direct | standard split | 0.523 | 0.0050 | 2.34 | 0.014 | 0.106 |
| XGB direct | event-weighted | 0.509 | 0.0053 | 2.33 | 0.013 | 0.104 |
| XGB direct | Mondrian PGA-tercile | 0.494 | 0.0053 | 2.33 | 0.014 | 0.099 |
| MLP scratch | standard split | 0.636 | 0.0064 | 2.09 | 0.026 | 0.231 |
| MLP scratch | event-weighted | 0.641 | 0.0066 | 2.09 | 0.029 | 0.229 |
| MLP scratch | Mondrian PGA-tercile | 0.637 | 0.0064 | 2.10 | 0.028 | 0.228 |

## U3 Vacuous-pass guard (add P_FU <= 0.30 to the filter)

| model | N | beta_FS,cons | P_FU | plain filter | guarded filter |
| --- | ---: | ---: | ---: | --- | --- |
| LGBM direct | 50 | 6.00 | 0.982 | Pass | Vacuous pass |
| LGBM direct | 100 | 3.16 | 0.078 | Pass | Pass |
| LGBM direct | 250 | 2.53 | 0.026 | Pass | Pass |
| LGBM direct | 500 | 2.56 | 0.021 | Pass | Pass |
| LGBM direct | 1000 | 2.50 | 0.016 | Pass | Pass |
| LGBM direct | 2000 | 2.37 | 0.012 | Fail | Fail |
| MLP scratch | 50 | 2.10 | 0.303 | Fail | Fail |
| MLP scratch | 100 | 2.10 | 0.105 | Fail | Fail |
| MLP scratch | 250 | 2.10 | 0.072 | Fail | Fail |
| MLP scratch | 500 | 2.10 | 0.050 | Fail | Fail |
| MLP scratch | 1000 | 2.10 | 0.035 | Fail | Fail |
| MLP scratch | 2000 | 2.10 | 0.026 | Fail | Fail |
| Ridge direct | 50 | 3.22 | 0.204 | Pass | Pass |
| Ridge direct | 100 | 2.98 | 0.136 | Pass | Pass |
| Ridge direct | 250 | 2.69 | 0.141 | Pass | Pass |
| Ridge direct | 500 | 2.65 | 0.132 | Pass | Pass |
| Ridge direct | 1000 | 2.65 | 0.116 | Pass | Pass |
| Ridge direct | 2000 | 2.65 | 0.122 | Pass | Pass |
| XGB direct | 50 | 2.35 | 0.050 | Fail | Fail |
| XGB direct | 100 | 2.32 | 0.036 | Fail | Fail |
| XGB direct | 250 | 2.38 | 0.023 | Fail | Fail |
| XGB direct | 500 | 2.37 | 0.019 | Fail | Fail |
| XGB direct | 1000 | 2.33 | 0.015 | Fail | Fail |
| XGB direct | 2000 | 2.35 | 0.014 | Fail | Fail |

- 1 pass cell(s) become vacuous under the guard; the most extreme is LGBM direct at N=50 with P_FU = 0.982 (flags nearly every case as unsafe, so the false-safe pass carries no screening value).

