# R35-E1c threshold-resolved partition-ensemble eligibility

24 partitions, 5 reps, guarded filter (beta* = 2.5, P_FU,max = 0.3), level 0.90.

| model | N | tau | guarded pass | beta med | cond. miss med | unsafe rows med | events w/ unsafe med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lgbm_direct | 500 | 0.5% | 58% | 2.60 | 0.035 | 626 | 3 |
| lgbm_direct | 2000 | 0.5% | 29% | 2.25 | 0.061 | 582 | 3 |
| ridge_direct | 500 | 0.5% | 25% | 1.59 | 0.188 | 626 | 3 |
| ridge_direct | 2000 | 0.5% | 29% | 1.62 | 0.198 | 582 | 3 |
| scratch_mlp | 500 | 0.5% | 4% | 1.25 | 0.424 | 626 | 3 |
| scratch_mlp | 2000 | 0.5% | 10% | 1.55 | 0.372 | 582 | 3 |
| xgb_direct | 500 | 0.5% | 25% | 2.37 | 0.062 | 626 | 3 |
| xgb_direct | 2000 | 0.5% | 24% | 2.35 | 0.069 | 582 | 3 |
| lgbm_direct | 500 | 1% | 25% | 2.21 | 0.284 | 221 | 3 |
| lgbm_direct | 2000 | 1% | 14% | 2.07 | 0.338 | 216 | 3 |
| ridge_direct | 500 | 1% | 33% | 1.94 | 0.270 | 221 | 3 |
| ridge_direct | 2000 | 1% | 38% | 2.06 | 0.291 | 216 | 3 |
| scratch_mlp | 500 | 1% | 8% | 1.61 | 0.540 | 221 | 3 |
| scratch_mlp | 2000 | 1% | 10% | 2.00 | 0.500 | 216 | 3 |
| xgb_direct | 500 | 1% | 17% | 2.12 | 0.335 | 221 | 3 |
| xgb_direct | 2000 | 1% | 19% | 2.07 | 0.333 | 216 | 3 |
| lgbm_direct | 500 | 2% | 38% | 1.92 | 1.000 | 48 | 2 |
| lgbm_direct | 2000 | 2% | 43% | 2.00 | 1.000 | 46 | 1 |
| ridge_direct | 500 | 2% | 46% | 2.40 | 0.734 | 48 | 2 |
| ridge_direct | 2000 | 2% | 52% | 3.16 | 0.717 | 46 | 1 |
| scratch_mlp | 500 | 2% | 38% | 2.01 | 0.935 | 48 | 2 |
| scratch_mlp | 2000 | 2% | 43% | 2.20 | 0.946 | 46 | 1 |
| xgb_direct | 500 | 2% | 38% | 1.92 | 1.000 | 48 | 2 |
| xgb_direct | 2000 | 2% | 43% | 2.00 | 1.000 | 46 | 1 |
