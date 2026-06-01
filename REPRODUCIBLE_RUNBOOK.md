# Reproducible runbook

1. Create a Python environment with `numpy`, `pandas` and `matplotlib`.
2. Run `python scripts/rebuild_residual_mechanism_figure.py` to rebuild the residual-mechanism metrics and figure from the derived residual traces.
3. Run `python scripts/rebuild_decision_risk_sensitivity.py` to rebuild the decision-risk sensitivity tables and figure.
4. Compare regenerated figures under `outputs/figures/high_target/` with the committed figures.

The package does not include raw third-party ground-motion records or active submission files.
