# R30 Structural Safety sprint reproducibility note

Date: 2026-06-02

This public repository update adds the R30 derived conditional-coverage diagnostic for the Structural Safety submission sprint.

Included public artefacts:

- `code/69_r30_conditional_coverage_intensity.py`: computes event-equal PGA-binned coverage diagnostics from the true event-disjoint N=2000 prediction export.
- `code/70_r30_polish_conditional_coverage_figure.py`: redraws the derived Figure 8 heatmap from the summary CSV without recomputing predictions.
- `outputs/high_target/r30_conditional_coverage_intensity/`: derived diagnostic CSV/Markdown/LaTeX outputs.
- `outputs/figures/high_target/fig_r30_conditional_coverage_pga_bins.*`: derived figure assets.

Boundary:

- The diagnostic is descriptive, event-equal, and used to make the marginal-vs-conditional coverage limitation visible.
- It is not a new conditional conformal guarantee.
- Unpublished manuscript PDFs, submission letters, `rounds/`, logs, raw third-party data and downloaded archives are intentionally excluded from the public repository.
