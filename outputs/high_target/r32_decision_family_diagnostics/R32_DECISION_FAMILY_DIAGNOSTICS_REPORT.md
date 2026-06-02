# R32 decision and structural-family diagnostic addendum

## Scope and evidence boundary

This addendum uses existing, already generated evidence from the R22/R24 decision-risk and false-safe reliability tables and the R28 true event-disjoint N=2000 prediction export. It does not introduce simulated placeholder values, fabricated observations, or AI-generated numerical results.

The structural-family analysis is a stratified diagnostic within the event-disjoint N=2000 stress test. It is not a true leave-family-out retraining experiment and should not be described as proof of extrapolation to unseen building inventories.

## Outputs

- `outputs/figures/high_target/fig_r32_decision_impact_map.*`
- `outputs/figures/high_target/fig_r32_structural_family_stress.*`
- `outputs/high_target/r32_decision_family_diagnostics/r32_decision_impact_selected_actions.csv`
- `outputs/high_target/r32_decision_family_diagnostics/r32_structural_family_stress_summary.csv`
- `submission/structural_safety_2026-06-01/supplementary_R32/Supplementary_R32_decision_family_diagnostics.pdf`

## Decision-map interpretation

The decision map selects the loss-minimizing action among models satisfying the conservative false-safe reliability threshold beta_FS,cons >= 2.5. If no surrogate satisfies the diagnostic filter for a threshold-cost cell, the action is shown as an NTHA fallback. The overlaid contours and cell percentages report the fraction of cases routed to downstream nonlinear analysis rather than treated as screened-safe.

## Structural-family diagnostic interpretation

The family diagnostic reports family-stratified conservative false-safe reliability and false-unsafe workload at a 1.0% drift threshold. The table below lists the weakest family for each evaluated model.

| model_label   | family                     |   false_safe_rate_median |   false_safe_rate_q95_rep |   beta_fs_cons_family |   false_unsafe_rate_median |   coverage_median | pass_beta_2p5   |
|:--------------|:---------------------------|-------------------------:|--------------------------:|----------------------:|---------------------------:|------------------:|:----------------|
| Ridge direct  | 3F soft-first-story T=1.8s |               0.00647645 |                0.00647645 |               2.48506 |                 0.138881   |          0.513432 | False           |
| LGBM direct   | 3F soft-first-story T=1.8s |               0.00960145 |                0.0116848  |               2.26734 |                 0.00815217 |          0.554875 | False           |
| XGB direct    | 3F soft-first-story T=1.8s |               0.0116848  |                0.0128306  |               2.2313  |                 0.00815217 |          0.528649 | False           |
| MLP scratch   | 3F soft-first-story T=1.8s |               0.0101562  |                0.013921   |               2.19951 |                 0.0337296  |          0.58379  | False           |

## Recommended manuscript use

Use these outputs as supplementary material or as a concise addendum during revision. Avoid inserting them into the already clean R31 main text unless a target-specific reason emerges, because the current main manuscript is internally stable and these diagnostics mainly strengthen reviewer reassurance rather than change the central claim.
