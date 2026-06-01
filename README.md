# P4 Seismic Surrogate Reliability

Public reproducibility package for the manuscript **Metric-dependent reliability of seismic surrogate calibration under event-level separation**.

This repository contains code, derived residual traces, derived tables, generated figures, citation metadata and run instructions. It intentionally excludes active submission manuscripts, cover letters, reviewer drafts, internal rounds/logs, credentials and raw third-party data whose redistribution rights are unclear.

## Main outputs

- `outputs/figures/high_target/Figure_1_research_object_mechanism.png`
- `outputs/figures/high_target/Figure_2_metric_ranks.png`
- `outputs/figures/high_target/Figure_3_residual_mechanism.png`
- `outputs/figures/high_target/Figure_4_decision_risk_sensitivity.png`

## Rebuild

Use Python 3.10+ with `numpy`, `pandas` and `matplotlib`.

```bash
python scripts/rebuild_residual_mechanism_figure.py
python scripts/rebuild_decision_risk_sensitivity.py
```

## Data boundary

The CSV files in `outputs/high_target/` are derived residual traces and derived decision-risk tables used to reproduce the figures. Raw third-party ground-motion records and source simulation inputs are not redistributed.

## R24 reliability-index extension

The repository now includes the R24 Structural Safety reproducibility increment: derived protocol/model/metric tables, a false-safe reliability-index figure, and verified reference metadata notes. Active submission manuscripts and PDFs remain excluded.


## Repository scope

This repository contains reproducibility materials only: code, configuration, derived tables, generated figures, runbooks, source registries, and citation metadata needed to reproduce the reported computational results. It intentionally excludes active manuscript drafts, journal source files, cover letters, private review rounds, raw third-party archives, credentials, and materials whose redistribution rights are unclear.

The R28 large-budget event-disjoint prediction export contains 796,080 regenerated prediction rows locally; because of size and redistribution boundaries, the public repository provides the generation scripts, aggregate summaries, derived figures, and reproducibility runbook rather than using the repository as an active submission manuscript archive.
