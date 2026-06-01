# R28 event-disjoint large-budget export

This run regenerates prediction residuals for event-disjoint target calibration/test splits rather than extrapolating from the N=500 trace.

- Fit/calibration event rows available: 2960
- Test event rows: 2992
- Feasible nominal target-label budgets: 50, 100, 250, 500, 1000, 2000
- Skipped requested budgets: none
- Models: ridge_direct, lgbm_direct, xgb_direct, scratch_mlp
- Repetitions: 10

## Output files

- `event_disjoint_large_budget_predictions.csv`: full calibration and test residual predictions.
- `event_disjoint_large_budget_summary.csv`: per-run residual and conformal metrics.
- `event_disjoint_large_budget_aggregate.csv`: model-budget aggregates.
