# R35-B second-domain demonstration: C-MAPSS FD002 turbofan RUL

Mapping: engine unit = event; tau = 30 cycles; unsafe = RUL <= tau; surrogate declares safe iff conformal lower bound (level 0.90) > tau. 60 pool engines, 200-engine population, N = 2000 labels, 3 reps, 300 8-engine draws.

| model | RMSE (cycles) | pop beta | pop P_FU | pop cond miss | pop guarded | 8-engine pass | assessable@8 | agreement@8 |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| lgbm_direct | 22.1 | 2.18 | 0.076 | 0.079 | Fail | 2% | 100% | 98% |
| ridge_direct | 22.9 | 2.00 | 0.049 | 0.124 | Fail | 1% | 100% | 99% |
| scratch_mlp | 22.1 | 2.10 | 0.062 | 0.096 | Fail | 2% | 100% | 98% |
| xgb_direct | 21.9 | 2.21 | 0.080 | 0.073 | Fail | 4% | 100% | 96% |
