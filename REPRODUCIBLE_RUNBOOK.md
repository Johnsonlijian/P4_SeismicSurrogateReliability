# Reproducible Runbook

## Environment

Use Python 3.10+ with:

```bash
pip install numpy pandas scipy scikit-learn matplotlib
```

Optional model backends used by some scripts include LightGBM and XGBoost. When
those packages are unavailable, inspect the provided derived CSV outputs rather
than treating the repository as a complete raw-data rerun environment.

## Inspect final artifacts

Final figure assets:

```bash
ls outputs/figures/ress_final
```

Final table snippets:

```bash
ls outputs/tables/ress_final
```

R35 evidence reports:

```bash
ls outputs/high_target/r35_*
```

## Rebuild notes

The R35 scripts use the same pipeline as the manuscript evidence audit. Full
reruns require locally reconstructed NSMP and nonlinear-MDOF derived inputs, and
the C-MAPSS FD002 source file downloaded from the public dataset link. The
repository provides derived outputs for verification and transparency while
avoiding redistribution of raw third-party data.

Key scripts:

```bash
python code/74_r35_conditional_miss_crc_and_metadata.py
python code/75_r35_partition_stability.py
python code/76_r35_sa_feature_ablation.py
python code/77_r35_bound_coverage.py
python code/78_r35_solver_verification.py
python code/79_r35_subset_and_guard_sensitivity.py
python code/80_r35_e7_evidence_scaling.py
python code/82_r35_second_domain_cmapss.py
```

## DOI/release workflow

1. Commit the clean repository state.
2. Tag the release, for example `v1.0.0-ress`.
3. Push the branch and tag to GitHub.
4. If Zenodo is connected to the GitHub repository, create/publish the GitHub
   release so Zenodo archives the snapshot and mints a DOI.
