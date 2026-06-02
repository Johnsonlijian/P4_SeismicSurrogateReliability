# R31 final cleanliness note

Date: 2026-06-02

This update records the final Structural Safety sprint cleanup applied after the R30 package:

- Figure 1 generation was corrected so the eligibility-filter box displays a real line break rather than a literal `\n`.
- The manuscript text now explicitly states that nonlinear labels are generated with an internal Python/NumPy Bouc-Wen story-shear benchmark integrated by a fixed-step fourth-order Runge-Kutta scheme.
- The conservative index notation was restored to `beta_FS,cons` / `\beta_{\mathrm{FS,cons}}` for consistent manuscript typography.
- The public repository README now points to the scripts and derived outputs for Table 7, Table 8, Table 9 and Figure 8.

No manuscript PDF, submission package, cover letter, private rounds, logs or raw third-party data are included in this public repository.
