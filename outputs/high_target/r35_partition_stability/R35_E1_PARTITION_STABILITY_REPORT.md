# R35-E1 partition stability

24 partitions (original 20260613 + 23 fresh), 5 reps, budgets [500, 2000], beta* = 2.5, P_FU,max = 0.3.

## Pass fractions over partitions (per-partition rep-medians)

| model | N | plain pass | guarded pass | beta med [p05, p95] | P_FU med | cond. miss med |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| lgbm_direct | 500 | 25% | 25% | 2.19 [1.39, 2.94] | 0.027 | 0.284 |
| lgbm_direct | 2000 | 14% | 14% | 2.09 [1.57, 2.65] | 0.015 | 0.338 |
| ridge_direct | 500 | 33% | 33% | 1.94 [1.67, 5.60] | 0.089 | 0.270 |
| ridge_direct | 2000 | 38% | 38% | 2.06 [1.75, 6.00] | 0.081 | 0.291 |
| scratch_mlp | 500 | 8% | 8% | 1.59 [1.33, 3.01] | 0.045 | 0.540 |
| scratch_mlp | 2000 | 10% | 10% | 2.00 [1.47, 3.08] | 0.030 | 0.500 |
| xgb_direct | 500 | 17% | 17% | 2.12 [1.48, 2.83] | 0.020 | 0.335 |
| xgb_direct | 2000 | 19% | 19% | 2.07 [1.60, 2.57] | 0.015 | 0.333 |

## N=500 -> N=2000 beta contrast per partition

| model | fraction of partitions with beta drop | median delta | [p05, p95] |
| --- | ---: | ---: | --- |
| lgbm_direct | 58% | -0.06 | [nan, nan] |
| ridge_direct | 46% | -0.00 | [nan, nan] |
| scratch_mlp | 17% | 0.07 | [nan, nan] |
| xgb_direct | 50% | -0.02 | [nan, nan] |
