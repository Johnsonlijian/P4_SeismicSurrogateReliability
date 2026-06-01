# R27 gate stability and interval sensitivity

Date: 2026-06-01

This round addresses the main external critique after R26: the manuscript had a clear false-safe gate but needed stability evidence.

## Evidence boundary

- The analysis uses the existing event-disjoint N=500 residual trace.
- Calibration-label budgets Calibration-label budgets 50/100/250 are stress tests obtained by recalibrating the interval from subsets of the available calibration residuals under fixed trained predictions.
- N=500 as a fresh calibration split and N=1000/2000 are not imputed because no corresponding event-disjoint prediction trace exists in the current project outputs.
- Interval widening is evaluated at conformal levels 0.90/0.95/0.975.

## Main conclusions

- Ridge direct at N=50: beta_FS_cons=2.65, gate=Pass.
- Ridge direct at N=100: beta_FS_cons=2.65, gate=Pass.
- Ridge direct at N=250: beta_FS_cons=2.65, gate=Pass.
- LGBM direct at N=50: beta_FS_cons=2.54, gate=Pass.

## Interval sensitivity summary

- LGBM direct level 0.900: beta_FS_cons=2.48, PFU=2.0%, gate=Fail.
- LGBM direct level 0.950: beta_FS_cons=2.83, PFU=2.6%, gate=Pass.
- LGBM direct level 0.975: beta_FS_cons=4.68, PFU=4.0%, gate=Pass.
- MLP scratch level 0.900: beta_FS_cons=2.10, PFU=5.4%, gate=Fail.
- MLP scratch level 0.950: beta_FS_cons=2.10, PFU=6.5%, gate=Fail.
- MLP scratch level 0.975: beta_FS_cons=2.10, PFU=7.6%, gate=Fail.
- Ridge direct level 0.900: beta_FS_cons=2.65, PFU=12.6%, gate=Pass.
- Ridge direct level 0.950: beta_FS_cons=3.16, PFU=15.6%, gate=Pass.
- Ridge direct level 0.975: beta_FS_cons=6.00, PFU=17.3%, gate=Pass.
- XGB direct level 0.900: beta_FS_cons=2.32, PFU=1.7%, gate=Fail.
- XGB direct level 0.950: beta_FS_cons=2.45, PFU=2.0%, gate=Fail.
- XGB direct level 0.975: beta_FS_cons=2.58, PFU=2.6%, gate=Pass.
