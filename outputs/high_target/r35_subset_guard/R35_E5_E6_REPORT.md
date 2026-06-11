# R35-E5/E6 story-subset and guard sensitivity

## E5: 1% IDR filter quantities by story subset (event-disjoint, medians over 10 reps)

| model | N | subset | beta_FS,cons | P_FU | cond. miss | unsafe rows | filter |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| lgbm_direct | 500 | 3-story | 2.34 | 0.017 | 0.168 | 76 | Fail |
| lgbm_direct | 500 | 6-story | 4.48 | 0.024 | 0.125 | 16 | Pass |
| lgbm_direct | 500 | all | 2.56 | 0.021 | 0.143 | 92 | Pass |
| lgbm_direct | 2000 | 3-story | 2.26 | 0.007 | 0.249 | 76 | Fail |
| lgbm_direct | 2000 | 6-story | 2.50 | 0.016 | 0.833 | 16 | Fail |
| lgbm_direct | 2000 | all | 2.37 | 0.012 | 0.341 | 92 | Fail |
| ridge_direct | 500 | 3-story | 2.40 | 0.144 | 0.095 | 76 | Fail |
| ridge_direct | 500 | 6-story | 6.00 | 0.121 | 0.000 | 16 | Pass |
| ridge_direct | 500 | all | 2.65 | 0.132 | 0.085 | 92 | Pass |
| ridge_direct | 2000 | 3-story | 2.40 | 0.130 | 0.095 | 76 | Fail |
| ridge_direct | 2000 | 6-story | 6.00 | 0.114 | 0.000 | 16 | Pass |
| ridge_direct | 2000 | all | 2.65 | 0.122 | 0.085 | 92 | Pass |
| scratch_mlp | 500 | 3-story | 1.93 | 0.055 | 0.351 | 76 | Fail |
| scratch_mlp | 500 | 6-story | 2.35 | 0.044 | 0.500 | 16 | Fail |
| scratch_mlp | 500 | all | 2.10 | 0.050 | 0.352 | 92 | Fail |
| scratch_mlp | 2000 | 3-story | 1.93 | 0.030 | 0.375 | 76 | Fail |
| scratch_mlp | 2000 | 6-story | 2.35 | 0.023 | 0.562 | 16 | Fail |
| scratch_mlp | 2000 | all | 2.10 | 0.026 | 0.370 | 92 | Fail |
| xgb_direct | 500 | 3-story | 2.20 | 0.016 | 0.237 | 76 | Fail |
| xgb_direct | 500 | 6-story | 2.47 | 0.022 | 0.833 | 16 | Fail |
| xgb_direct | 500 | all | 2.38 | 0.019 | 0.309 | 92 | Fail |
| xgb_direct | 2000 | 3-story | 2.24 | 0.009 | 0.289 | 76 | Fail |
| xgb_direct | 2000 | 6-story | 2.50 | 0.016 | 0.833 | 16 | Fail |
| xgb_direct | 2000 | all | 2.33 | 0.014 | 0.363 | 92 | Fail |

## E6: guarded verdicts vs P_FU,max

Differences relative to P_FU,max = 0.30:

- P_FU,max = 0.15: Ridge direct N=50: Pass -> Vacuous/guard-fail
- P_FU,max = 0.5: identical verdict set.
