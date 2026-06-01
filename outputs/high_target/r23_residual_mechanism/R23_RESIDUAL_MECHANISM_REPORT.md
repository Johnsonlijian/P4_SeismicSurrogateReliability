# R23 residual-mechanism analysis

## Purpose
This analysis addresses the reviewer/panel concern that metric-dependent ranking should be explained mechanistically rather than reported only as a data audit. It uses the existing full residual traces and does not add unverified external facts.

## Mechanism claim supported by the current residual traces
Mean-error metrics, conformal coverage, interval/tail behavior and false-safe decision loss interrogate different regions of the residual distribution. Under event-level shift, residual scale mismatch and tail amplification can therefore reorder model preference even when all models are evaluated on the same finite-label budget.

## Protocol-level findings
### main event-held-out
- Lowest RMSE: XGB direct (RMSE=0.150, coverage=0.747, q95|resid|=0.305).
- Highest coverage: MLP scratch (coverage=0.888, RMSE=0.192).
- Smallest q95 absolute residual: XGB direct (q95|resid|=0.305, RMSE=0.150).
- Largest test/calibration residual-scale ratio: HGB direct (ratio=1.471).

### event-disjoint target
- Lowest RMSE: LGBM direct (RMSE=0.170, coverage=0.685, q95|resid|=0.329).
- Highest coverage: LGBM direct (coverage=0.685, RMSE=0.170).
- Smallest q95 absolute residual: XGB direct (q95|resid|=0.327, RMSE=0.171).
- Largest test/calibration residual-scale ratio: MLP scratch (ratio=3.740).

## Manuscript-safe interpretation
The mechanism evidence supports a bounded claim: metric conflict is consistent with residual distribution geometry under event-level shift. It does not prove a universal law for all seismic systems or a new conformal-prediction theorem.

## Recommended manuscript insertion
Add a Results subsection titled `Residual geometry explains why reliability metrics disagree`. Use Fig. R23 as the mechanism figure after the current rank figure and before the decision-risk sensitivity figure, or move it to Extended Data if the target journal enforces a strict display-item budget.

## Generated outputs
- Metrics table: `R:\NAS_DRIVE\IMUT\1-Research_Output\1-Papers\1_In_Preparation\2026-Claude\projects\P4_SeismicFoundationModel\public_repo\P4_SeismicSurrogateReliability\outputs\high_target\r23_residual_mechanism\residual_mechanism_metrics.csv`
- Figure base: `R:\NAS_DRIVE\IMUT\1-Research_Output\1-Papers\1_In_Preparation\2026-Claude\projects\P4_SeismicFoundationModel\public_repo\P4_SeismicSurrogateReliability\outputs\figures\high_target\fig_r23_residual_mechanism` in PNG/SVG/PDF/TIFF