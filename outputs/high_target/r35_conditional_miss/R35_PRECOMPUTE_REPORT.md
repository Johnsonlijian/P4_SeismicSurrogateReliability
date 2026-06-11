# R35 conditional miss, CRC baseline, event metadata

Row-level CRC target alpha* = Phi(-2.5) = 0.0062 (false-safe risk at 1% IDR), calibrated with the (n+1) finite-sample correction on the event-disjoint calibration set (N=2000), evaluated on held-out disjoint test events.

## CRC baseline (medians over 10 reps)

| model | lambda (log10) | test FS row-level | test FS event-equal | test FU row-level | reps violating alpha* |
| --- | ---: | ---: | ---: | ---: | ---: |
| lgbm_direct | 0.019 | 0.0201 | 0.0091 | 0.008 | 100% |
| ridge_direct | 0.106 | 0.0069 | 0.0036 | 0.192 | 50% |
| scratch_mlp | 0.089 | 0.0192 | 0.0083 | 0.013 | 100% |
| xgb_direct | 0.013 | 0.0174 | 0.0083 | 0.008 | 100% |

## Conditional miss by budget (1% IDR, event-disjoint, median over reps)

| model | N | conditional miss (median) | p05-p95 |
| --- | ---: | ---: | --- |
| lgbm_direct | 50 | 0.000 | [0.000, 1.000] |
| lgbm_direct | 100 | 0.014 | [0.000, 0.126] |
| lgbm_direct | 250 | 0.150 | [0.043, 0.339] |
| lgbm_direct | 500 | 0.143 | [0.043, 0.285] |
| lgbm_direct | 1000 | 0.231 | [0.098, 0.455] |
| lgbm_direct | 2000 | 0.341 | [0.248, 0.404] |
| ridge_direct | 50 | 0.010 | [0.000, 0.106] |
| ridge_direct | 100 | 0.018 | [0.000, 0.161] |
| ridge_direct | 250 | 0.067 | [0.011, 0.137] |
| ridge_direct | 500 | 0.085 | [0.019, 0.085] |
| ridge_direct | 1000 | 0.085 | [0.064, 0.105] |
| ridge_direct | 2000 | 0.085 | [0.064, 0.085] |
| scratch_mlp | 50 | 0.345 | [0.333, 0.386] |
| scratch_mlp | 100 | 0.349 | [0.333, 0.557] |
| scratch_mlp | 250 | 0.358 | [0.339, 0.493] |
| scratch_mlp | 500 | 0.352 | [0.335, 0.398] |
| scratch_mlp | 1000 | 0.379 | [0.347, 0.416] |
| scratch_mlp | 2000 | 0.370 | [0.333, 0.479] |
| xgb_direct | 50 | 0.220 | [0.033, 0.727] |
| xgb_direct | 100 | 0.228 | [0.017, 0.683] |
| xgb_direct | 250 | 0.233 | [0.097, 0.556] |
| xgb_direct | 500 | 0.309 | [0.164, 0.358] |
| xgb_direct | 1000 | 0.318 | [0.177, 0.418] |
| xgb_direct | 2000 | 0.363 | [0.251, 0.523] |
