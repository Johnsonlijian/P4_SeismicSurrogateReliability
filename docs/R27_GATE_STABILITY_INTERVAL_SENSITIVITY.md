# R27 gate stability and interval sensitivity closeout

Date: 2026-06-01
Target journal: Structural Safety

## Trigger

The external critique after R26 was correct that the false-safe gate narrative needed stability evidence. R27 therefore adds an evidence-bound stress test rather than another narrative-only rewrite.

## Completed hard increments

- Added a new reproducible script: `code/65_r27_gate_stability_sensitivity.py`.
- Computed calibration-label-budget gate stability using the event-disjoint residual trace at the 1% IDR threshold.
- Computed interval-widening sensitivity at conformal levels 0.90, 0.95 and 0.975.
- Added manuscript Table 7 for calibration-label-budget stability.
- Added manuscript Table 8 for interval-widening sensitivity.
- Added manuscript Figure 7 as a four-panel stability figure.
- Added an R27 results subsection connecting finite calibration information, false-safe gate eligibility and false-unsafe screening burden.
- Updated highlights to include gate stability.
- Added `\RequirePackage{cmap}`, `T1` font encoding and Latin Modern to improve PDF text extraction and reduce ligature artefacts.

## Evidence boundary

The event-disjoint trace contains 250 calibration residuals per replicate for the nominal N=500 run. R27 therefore reports calibration-label stress tests at 50, 100 and 250 calibration residuals. N=1000/2000 curves were not imputed, because no corresponding event-disjoint prediction export is present in the current project outputs.

This is an intentional safer implementation. It strengthens the paper without inventing unsupported label-budget results.

## Substantive result

The added evidence shows that gate eligibility is not equivalent to average accuracy. Ridge direct remains gate-eligible across the tested calibration budgets but carries a larger false-unsafe burden. LGBM direct sits near the diagnostic gate boundary. XGB direct and MLP scratch remain below the beta*=2.5 gate under the 90% interval. Interval widening can recover gate eligibility for some models, but increases the false-unsafe burden.

## Remaining optional hardening

A future R28 could regenerate true event-disjoint prediction traces for larger nominal target budgets, for example N=1000/2000, but this requires rerunning the model-training/export pipeline and is not claimed in R27.
