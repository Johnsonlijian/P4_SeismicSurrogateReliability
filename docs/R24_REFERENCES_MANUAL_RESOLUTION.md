# R24 reference audit manual resolution

Date checked: 2026-06-01.

`workflow-config\\imut.cmd references-check` was run on `references_r24.bib` as a Crossref-first audit. The automatic checker reported 17 `VERIFIED_DOI`, 2 `UNVERIFIED_ERROR`, and 4 `WEAK_METADATA_CANDIDATE` items.

## Manual resolution

- Rackwitz (2001), `Reliability analysis--a review and some perspectives`, Structural Safety: Crossref query verified DOI `10.1016/s0167-4730(02)00009-7`; the BibTeX DOI was normalized to the lowercase Crossref form.
- Cornell et al. (2002), `Probabilistic Basis for 2000 SAC Federal Emergency Management Agency Steel Moment Frame Guidelines`, Journal of Structural Engineering: Crossref query verified DOI `10.1061/(ASCE)0733-9445(2002)128:4(526)`. The audit script truncated the closing parenthesis in its parser, so this is treated as a manual DOI pass rather than a content failure.
- Romano et al. (2019), `Conformalized Quantile Regression`: retained as an official NeurIPS proceedings entry with stable proceedings URL.
- Angelopoulos and Bates (2021), `A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification`: retained as arXiv `2107.07511`.
- Ke et al. (2017), `LightGBM`: retained as an official NeurIPS proceedings entry with stable proceedings URL.
- Pedregosa et al. (2011), `Scikit-learn`: retained as the official JMLR paper page.

## Submission decision

Reference integrity status for R24: `PASS_WITH_MANUAL_RESOLUTION`. The listed manual-resolution entries are not invented references; they are DOI-, arXiv-, official-proceedings-, or official-journal-page verified.
