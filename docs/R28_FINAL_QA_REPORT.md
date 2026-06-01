# Structural Safety R28 final QA report

Date: 2026-06-01
Manuscript title: A false-safe reliability gate for finite-label seismic surrogate screening under event-level separation

## R28 additions

- True event-disjoint large-budget prediction export for N=1000 and N=2000.
- Updated Table 7: true event-disjoint target-label-budget gate stability from N=50 to N=2000.
- Updated Table 8: N=2000 interval-widening sensitivity.
- Updated Figure 7: true target-label-budget gate stability and interval sensitivity.
- Updated manuscript results subsection and abstract sentence.

## Build and QA

- LaTeX build completed with pdflatex, bibtex, pdflatex, pdflatex.
- Final PDF has 17 pages.
- Log gate found no fatal errors, no emergency stops and no undefined citations or references.
- Rendered pages containing Table 7, Table 8 and Figure 7 were visually inspected.
- Figure 7 is complete and not clipped.

## Evidence boundary

- R28 does not infer N=1000/2000 from N=500; it regenerates event-disjoint predictions.
- The full local prediction export has 796,080 rows and is retained outside the public GitHub repository.
- The public repository includes scripts and summary/aggregate outputs sufficient to reproduce and audit the result without committing the large derived trace.

## Decision

R28 is materially stronger than R27 for Structural Safety because the main reviewer-risk item, large-budget gate stability, is now supported by regenerated event-disjoint prediction exports.
