# P4 Seismic Surrogate Reliability

This is the public reproducibility package for the RESS submission:

**False-safe eligibility auditing of machine-learning surrogates for seismic demand screening under event-level distribution shift**

The repository is intentionally limited to reproducible, non-sensitive
materials: scripts, derived tables, generated figures, source registries,
run instructions, and citation metadata. It does not contain active manuscript
source, cover letters, reviewer-response drafts, internal working rounds, raw
third-party data archives, credentials, or files with unclear redistribution
rights.

## What is included

- `code/`: scripts for the final R34/R35 evidence increments and final figure
  generation.
- `outputs/figures/ress_final/`: final manuscript figure assets in PDF, PNG,
  and SVG form.
- `outputs/tables/ress_final/`: final derived table snippets used to audit the
  reported results.
- `outputs/high_target/`: derived CSV/JSON/Markdown reports needed to inspect
  the reliability-filter experiments.
- `DATASETS_AND_LINKS.csv`: source-data registry and redistribution boundary.
- `CITATION.cff` and `.zenodo.json`: citation and DOI-minting metadata.

## Reproducibility boundary

The public files are sufficient to inspect the reported numerical summaries and
final figures. Full reruns require rebuilding or downloading external source
data listed in `DATASETS_AND_LINKS.csv`. Raw NSMP ground-motion arrays and NASA
C-MAPSS text files are not redistributed here.

## Suggested citation

Please cite the associated manuscript and this repository. If Zenodo is enabled
for this GitHub repository, the `v1.0.1-ress` release should mint an archival
DOI using the metadata in `.zenodo.json`.
