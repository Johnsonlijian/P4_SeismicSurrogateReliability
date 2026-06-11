# R35-E2 Sa(T1)/Sa_avg feature ablation

Paired reps (N_reps=10, identical seeds). beta* = 2.5; spectral features: Sa(T1), Sa_avg(0.2T1-3T1), Newmark linear SDOF zeta=5% from the stored 1000-record array.

## Event-disjoint verdict comparison

| model | N | RMSE base->spec (gain%) | beta base->spec | cond.miss base->spec | filter base->spec | flip |
| --- | ---: | --- | --- | --- | --- | --- |
| lgbm_direct | 50 | 0.467->0.467 (+0.0%) | 6.00->6.00 | 0.00->0.00 | Pass->Pass | no |
| lgbm_direct | 100 | 0.244->0.226 (+7.3%) | 6.00->4.33 | 0.00->0.04 | Pass->Pass | no |
| lgbm_direct | 250 | 0.190->0.134 (+29.4%) | 2.81->2.34 | 0.05->0.20 | Pass->Fail | YES |
| lgbm_direct | 500 | 0.167->0.104 (+38.1%) | 2.48->2.32 | 0.17->0.21 | Fail->Fail | no |
| lgbm_direct | 1000 | 0.157->0.090 (+42.7%) | 2.47->2.21 | 0.31->0.33 | Fail->Fail | no |
| lgbm_direct | 2000 | 0.155->0.086 (+44.3%) | 2.33->2.19 | 0.33->0.31 | Fail->Fail | no |
| ridge_direct | 50 | 0.553->0.453 (+18.2%) | 3.22->6.00 | 0.01->0.00 | Pass->Pass | no |
| ridge_direct | 100 | 0.412->0.336 (+18.5%) | 2.92->3.02 | 0.05->0.03 | Pass->Pass | no |
| ridge_direct | 250 | 0.479->0.476 (+0.6%) | 2.94->2.87 | 0.05->0.05 | Pass->Pass | no |
| ridge_direct | 500 | 0.501->0.435 (+13.1%) | 2.65->2.65 | 0.08->0.08 | Pass->Pass | no |
| ridge_direct | 1000 | 0.462->0.426 (+7.7%) | 2.65->2.65 | 0.08->0.08 | Pass->Pass | no |
| ridge_direct | 2000 | 0.487->0.435 (+10.7%) | 2.65->2.65 | 0.08->0.08 | Pass->Pass | no |
| scratch_mlp | 50 | 1.847->2.030 (-9.9%) | 2.10->2.10 | 0.35->0.39 | Fail->Fail | no |
| scratch_mlp | 100 | 1.584->1.467 (+7.4%) | 2.10->2.10 | 0.37->0.39 | Fail->Fail | no |
| scratch_mlp | 250 | 1.400->1.406 (-0.4%) | 2.10->2.10 | 0.36->0.43 | Fail->Fail | no |
| scratch_mlp | 500 | 0.837->1.056 (-26.1%) | 2.10->2.10 | 0.34->0.40 | Fail->Fail | no |
| scratch_mlp | 1000 | 0.764->0.808 (-5.8%) | 2.10->2.10 | 0.39->0.40 | Fail->Fail | no |
| scratch_mlp | 2000 | 0.533->0.575 (-8.0%) | 2.09->2.09 | 0.41->0.38 | Fail->Fail | no |
| xgb_direct | 50 | 0.288->0.213 (+26.1%) | 2.37->2.33 | 0.22->0.26 | Fail->Fail | no |
| xgb_direct | 100 | 0.212->0.160 (+24.5%) | 2.18->2.23 | 0.32->0.29 | Fail->Fail | no |
| xgb_direct | 250 | 0.184->0.111 (+39.5%) | 2.49->2.19 | 0.21->0.32 | Fail->Fail | no |
| xgb_direct | 500 | 0.168->0.098 (+41.7%) | 2.34->2.18 | 0.29->0.41 | Fail->Fail | no |
| xgb_direct | 1000 | 0.159->0.092 (+42.4%) | 2.38->2.18 | 0.34->0.39 | Fail->Fail | no |
| xgb_direct | 2000 | 0.159->0.088 (+44.6%) | 2.40->2.22 | 0.29->0.37 | Fail->Fail | no |

## Main protocol (N=500, 7 families)

| model | featureset | RMSE | coverage | beta | cond. miss |
| --- | --- | ---: | ---: | ---: | ---: |
| hgb_direct | base | 0.157 | 0.700 | 2.58 | 0.099 |
| hgb_direct | spectral | 0.097 | 0.777 | 6.00 | 0.000 |
| lgbm_direct | base | 0.154 | 0.761 | 2.59 | 0.099 |
| lgbm_direct | spectral | 0.098 | 0.775 | 6.00 | 0.000 |
| pretrained_finetune | base | 0.188 | 0.820 | 2.46 | 0.104 |
| pretrained_finetune | spectral | 0.111 | 0.885 | 6.00 | 0.000 |
| rf_direct | base | 0.166 | 0.755 | 2.48 | 0.144 |
| rf_direct | spectral | 0.092 | 0.847 | 3.04 | 0.032 |
| ridge_direct | base | 0.237 | 0.756 | 2.59 | 0.100 |
| ridge_direct | spectral | 0.233 | 0.745 | 2.64 | 0.046 |
| scratch_mlp | base | 0.191 | 0.895 | 2.76 | 0.047 |
| scratch_mlp | spectral | 0.185 | 0.906 | 2.45 | 0.080 |
| xgb_direct | base | 0.151 | 0.715 | 2.53 | 0.133 |
| xgb_direct | spectral | 0.079 | 0.808 | 2.79 | 0.049 |
