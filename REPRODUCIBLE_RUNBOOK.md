# Reproducible runbook

1. Create a Python environment with `numpy`, `pandas` and `matplotlib`.
2. Run `python scripts/rebuild_residual_mechanism_figure.py` to rebuild the residual-mechanism metrics and figure from the derived residual traces.
3. Run `python scripts/rebuild_decision_risk_sensitivity.py` to rebuild the decision-risk sensitivity tables and figure.
4. Compare regenerated figures under `outputs/figures/high_target/` with the committed figures.

The package does not include raw third-party ground-motion records or active submission files.

## R24 Structural Safety reliability tables

Run `python code/64_r24_structural_safety_tables_reliability.py` from the project root to regenerate the R24 protocol/model/metric/decision tables and the false-safe reliability-index figure. The script reads derived residual traces and public NSMP metadata, and writes only derived CSV/figure outputs. Active submission manuscripts are intentionally excluded from this public repository.

