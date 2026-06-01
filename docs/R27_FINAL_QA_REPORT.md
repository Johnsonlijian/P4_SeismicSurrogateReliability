# Structural Safety R27 final QA report

Date: 2026-06-01
Manuscript title: A false-safe reliability gate for finite-label seismic surrogate screening under event-level separation

## R27 additions

- New Table 7: calibration-label-budget gate stability.
- New Table 8: interval-widening sensitivity.
- New Figure 7: label-budget and interval-widening sensitivity under event-disjoint testing.
- New results subsection: gate stability under calibration-budget and interval sensitivity.
- Updated highlights and source package.

## Build and visual QA

- LaTeX build completed with pdflatex, bibtex, pdflatex, pdflatex.
- Final PDF has 17 pages.
- Log gate found no fatal errors, no emergency stops, and no undefined citations or references.
- New Table 7 and Table 8 were rendered and visually checked.
- Figure 7 was rendered after removing label overlap and switching budget panels to clean linear x-axis labels.
- PDF text extraction was checked for common ligature artefacts: finite, finetune, different and sufficient extract correctly; no fi/fl ligature code points or replacement characters were found in pdftotext output.

## Known harmless warnings

- CAS template float-position warnings remain cosmetic.
- A front-matter overfull hbox remains cosmetic.
- BibTeX reports empty page fields for several verified software/method references.

## Decision

R27 is stronger than R26 for Structural Safety because it adds direct stability evidence for the false-safe reliability gate while preserving evidence boundaries.
