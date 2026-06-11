# R35-E7 evidence-scaling study

Population = 37 non-fit events; draws with replacement; guarded verdict = beta>=2.5 & P_FU<=0.30 & cond<=0.25 & >= 3 unsafe-bearing events.

## Population-limit (all 37 events) verdicts

| model | N | beta (point) | P_FU | cond. miss | guarded pass | unsafe-bearing events |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| lgbm_direct | 500 | 2.68 | 0.029 | 0.126 | Pass | 12 |
| lgbm_direct | 2000 | 2.54 | 0.012 | 0.193 | Pass | 12 |
| ridge_direct | 500 | 2.79 | 0.112 | 0.059 | Pass | 12 |
| ridge_direct | 2000 | 2.76 | 0.094 | 0.067 | Pass | 12 |
| scratch_mlp | 500 | 1.73 | 0.041 | 0.527 | Fail | 12 |
| scratch_mlp | 2000 | 1.85 | 0.027 | 0.405 | Fail | 12 |
| xgb_direct | 500 | 2.51 | 0.024 | 0.204 | Pass | 12 |
| xgb_direct | 2000 | 2.46 | 0.012 | 0.214 | Fail | 12 |

## Verdict stabilization vs number of test events E (N=2000, mean over reps)

| model | E | U95 coverage | pass fraction | assessable | agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| lgbm_direct | 6 | 72% | 2% | 31% | 98% |
| lgbm_direct | 8 | 78% | 3% | 52% | 97% |
| lgbm_direct | 12 | 81% | 10% | 80% | 90% |
| lgbm_direct | 16 | 85% | 12% | 94% | 88% |
| lgbm_direct | 20 | 84% | 14% | 98% | 86% |
| lgbm_direct | 24 | 87% | 14% | 100% | 86% |
| lgbm_direct | 28 | 88% | 14% | 100% | 86% |
| lgbm_direct | 32 | 90% | 12% | 100% | 88% |
| ridge_direct | 6 | 66% | 8% | 31% | 92% |
| ridge_direct | 8 | 77% | 15% | 52% | 85% |
| ridge_direct | 12 | 80% | 34% | 80% | 66% |
| ridge_direct | 16 | 84% | 52% | 94% | 54% |
| ridge_direct | 20 | 87% | 64% | 98% | 64% |
| ridge_direct | 24 | 89% | 68% | 100% | 68% |
| ridge_direct | 28 | 88% | 76% | 100% | 76% |
| ridge_direct | 32 | 90% | 78% | 100% | 78% |
| scratch_mlp | 6 | 49% | 0% | 31% | 100% |
| scratch_mlp | 8 | 45% | 1% | 52% | 99% |
| scratch_mlp | 12 | 46% | 1% | 80% | 99% |
| scratch_mlp | 16 | 48% | 0% | 94% | 100% |
| scratch_mlp | 20 | 48% | 1% | 98% | 99% |
| scratch_mlp | 24 | 56% | 1% | 100% | 99% |
| scratch_mlp | 28 | 58% | 0% | 100% | 100% |
| scratch_mlp | 32 | 64% | 0% | 100% | 100% |
| xgb_direct | 6 | 67% | 2% | 31% | 98% |
| xgb_direct | 8 | 74% | 3% | 52% | 97% |
| xgb_direct | 12 | 74% | 8% | 80% | 92% |
| xgb_direct | 16 | 80% | 10% | 94% | 90% |
| xgb_direct | 20 | 83% | 10% | 98% | 90% |
| xgb_direct | 24 | 87% | 9% | 100% | 91% |
| xgb_direct | 28 | 85% | 9% | 100% | 91% |
| xgb_direct | 32 | 90% | 7% | 100% | 93% |
