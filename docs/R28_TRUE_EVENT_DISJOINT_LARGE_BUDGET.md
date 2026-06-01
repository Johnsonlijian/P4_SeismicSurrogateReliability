# R28 true event-disjoint large-budget gate stability closeout

Date: 2026-06-01
Target journal: Structural Safety

## Objective

R28 addresses the explicit hard request: regenerate event-disjoint N=1000 and N=2000 prediction exports rather than extrapolating from the previous N=500 trace.

## Completed hard increments

- Added `code/66_r28_event_disjoint_large_budget.py` to rerun the event-disjoint target calibration/test protocol.
- Regenerated true event-disjoint prediction residual exports for nominal target-label budgets N=50, 100, 250, 500, 1000 and 2000.
- Used 10 repetitions and four representative models: Ridge direct, LGBM direct, XGB direct and MLP scratch.
- Exported full calibration and test residual predictions locally in `event_disjoint_large_budget_predictions.csv`.
- Added `code/67_r28_gate_large_budget_sensitivity.py` to recompute false-safe gate metrics from the regenerated prediction export.
- Rebuilt Table 7, Table 8 and Figure 7 from the true R28 prediction export.
- Updated the manuscript text from the R27 calibration-subset stress-test wording to true target-label-budget stability wording.
- Recompiled the manuscript PDF and rebuilt the R28 submission package.

## Evidence result

At the 1% IDR threshold, Ridge direct remains false-safe-gate eligible across N=50 to N=2000. LGBM direct remains near the boundary and passes through N=1000 but falls below the beta*=2.5 diagnostic gate at N=2000 under the 90% interval. XGB direct and MLP scratch remain below the gate across the true regenerated budgets. At N=2000, widening to the 0.975 conformal interval recovers gate eligibility for LGBM direct and XGB direct, but increases false-unsafe burden.

## Interpretation

The key R28 conclusion is that gate stability is not equivalent to an average-error learning curve. More labels can tighten intervals and lower false-unsafe burden, but this can expose false-safe cases. This directly strengthens the Structural Safety narrative because model eligibility is treated as a reliability-efficiency decision, not a leaderboard.

## Public-repo boundary

The full 796,080-row prediction export is kept locally as a reproducibility output but is not committed to GitHub to avoid bloating the public repository. The public repo receives scripts, metadata, aggregate/summary CSVs, reports and figures; the full prediction export is reproducible by rerunning the R28 script.
