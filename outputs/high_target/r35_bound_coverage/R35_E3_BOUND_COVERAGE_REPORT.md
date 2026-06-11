# R35-E3 bound-coverage study

Population = 37 non-fit events; 500 random 8-event subsets per rep, 3 reps; nominal target 95%.

| model | N | population P_FS | U95 coverage | U95+floor | U99 | t-bound | t+floor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lgbm_direct | 500 | 0.0035 | 78.9% | 79.2% | 85.3% | 79.9% | 80.3% |
| lgbm_direct | 2000 | 0.0059 | 81.9% | 81.9% | 87.9% | 82.3% | 82.3% |
| ridge_direct | 500 | 0.0033 | 71.4% | 73.2% | 72.9% | 72.1% | 73.9% |
| ridge_direct | 2000 | 0.0027 | 80.1% | 80.8% | 80.1% | 80.1% | 80.8% |
| scratch_mlp | 500 | 0.0230 | 63.8% | 63.8% | 67.1% | 63.7% | 63.7% |
| scratch_mlp | 2000 | 0.0322 | 55.7% | 55.7% | 61.5% | 56.6% | 56.6% |
| xgb_direct | 500 | 0.0047 | 80.3% | 80.4% | 84.8% | 81.3% | 81.5% |
| xgb_direct | 2000 | 0.0072 | 76.5% | 76.5% | 83.0% | 77.2% | 77.2% |
