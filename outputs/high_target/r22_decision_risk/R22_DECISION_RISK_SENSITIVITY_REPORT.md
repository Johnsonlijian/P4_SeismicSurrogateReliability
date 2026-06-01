# R22 decision-risk sensitivity

Decision rule: a case is predicted safe only when the upper conformal bound is below the drift threshold.
False-safe errors are weighted by cost ratio C; false-unsafe errors have unit cost.
Rates and expected losses are event-equal.

## Winner changes across threshold and cost ratio

### event-disjoint target
- XGB direct: winner in 19 of 49 threshold-cost cells.
- HGB direct: winner in 18 of 49 threshold-cost cells.
- LGBM direct: winner in 9 of 49 threshold-cost cells.
- Ridge direct: winner in 3 of 49 threshold-cost cells.

| threshold | cost ratio | winner | expected loss | false-safe | false-unsafe |
| ---: | ---: | --- | ---: | ---: | ---: |
| 1.00% | 1 | XGB direct | 0.029 | 0.0106 | 0.0182 |
| 1.00% | 10 | HGB direct | 0.092 | 0.0070 | 0.0224 |
| 1.00% | 50 | Ridge direct | 0.368 | 0.0020 | 0.2675 |
| 1.00% | 100 | Ridge direct | 0.468 | 0.0020 | 0.2675 |

### main event-held-out
- LGBM direct: winner in 15 of 49 threshold-cost cells.
- XGB direct: winner in 13 of 49 threshold-cost cells.
- HGB direct: winner in 9 of 49 threshold-cost cells.
- Ridge direct: winner in 6 of 49 threshold-cost cells.
- MLP scratch: winner in 3 of 49 threshold-cost cells.
- MLP finetune: winner in 3 of 49 threshold-cost cells.

| threshold | cost ratio | winner | expected loss | false-safe | false-unsafe |
| ---: | ---: | --- | ---: | ---: | ---: |
| 1.00% | 1 | XGB direct | 0.051 | 0.0050 | 0.0462 |
| 1.00% | 10 | LGBM direct | 0.088 | 0.0032 | 0.0554 |
| 1.00% | 50 | LGBM direct | 0.217 | 0.0032 | 0.0554 |
| 1.00% | 100 | Ridge direct | 0.360 | 0.0029 | 0.0741 |

## Manuscript-safe claim

Decision-risk sensitivity is threshold- and cost-dependent. Therefore the manuscript should not report a single universally safest model. It should show a sensitivity surface and state that engineering risk preferences can reorder the model ranking.

## Files

- `decision_risk_sensitivity_detail.csv`
- `decision_risk_sensitivity_winners.csv`
- `fig_r22_decision_risk_sensitivity.*`
