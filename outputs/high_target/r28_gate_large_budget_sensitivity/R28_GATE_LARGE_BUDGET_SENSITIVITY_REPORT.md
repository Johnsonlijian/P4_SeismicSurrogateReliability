# R28 true event-disjoint filter stability

Date: 2026-06-01

R28 replaces the R27 calibration-subset stress test with true regenerated event-disjoint prediction exports for N=1000 and N=2000.

## Main budget-gate summary

- LGBM direct N=50: beta_FS_cons=6.00, PFS=0.00%, PFU=98.2%, gate=Pass.
- LGBM direct N=100: beta_FS_cons=3.16, PFS=0.03%, PFU=7.8%, gate=Pass.
- LGBM direct N=250: beta_FS_cons=2.53, PFS=0.24%, PFU=2.6%, gate=Pass.
- LGBM direct N=500: beta_FS_cons=2.56, PFS=0.27%, PFU=2.1%, gate=Pass.
- LGBM direct N=1000: beta_FS_cons=2.50, PFS=0.34%, PFU=1.6%, gate=Pass.
- LGBM direct N=2000: beta_FS_cons=2.37, PFS=0.49%, PFU=1.2%, gate=Fail.
- MLP scratch N=50: beta_FS_cons=2.10, PFS=0.63%, PFU=30.3%, gate=Fail.
- MLP scratch N=100: beta_FS_cons=2.10, PFS=0.62%, PFU=10.5%, gate=Fail.
- MLP scratch N=250: beta_FS_cons=2.10, PFS=0.64%, PFU=7.2%, gate=Fail.
- MLP scratch N=500: beta_FS_cons=2.10, PFS=0.62%, PFU=5.0%, gate=Fail.
- MLP scratch N=1000: beta_FS_cons=2.10, PFS=0.64%, PFU=3.5%, gate=Fail.
- MLP scratch N=2000: beta_FS_cons=2.10, PFS=0.64%, PFU=2.6%, gate=Fail.
- Ridge direct N=50: beta_FS_cons=3.22, PFS=0.02%, PFU=20.4%, gate=Pass.
- Ridge direct N=100: beta_FS_cons=2.98, PFS=0.05%, PFU=13.6%, gate=Pass.
- Ridge direct N=250: beta_FS_cons=2.69, PFS=0.14%, PFU=14.1%, gate=Pass.
- Ridge direct N=500: beta_FS_cons=2.65, PFS=0.16%, PFU=13.2%, gate=Pass.
- Ridge direct N=1000: beta_FS_cons=2.65, PFS=0.16%, PFU=11.6%, gate=Pass.
- Ridge direct N=2000: beta_FS_cons=2.65, PFS=0.16%, PFU=12.2%, gate=Pass.
- XGB direct N=50: beta_FS_cons=2.35, PFS=0.38%, PFU=5.0%, gate=Fail.
- XGB direct N=100: beta_FS_cons=2.32, PFS=0.38%, PFU=3.6%, gate=Fail.
- XGB direct N=250: beta_FS_cons=2.38, PFS=0.39%, PFU=2.3%, gate=Fail.
- XGB direct N=500: beta_FS_cons=2.37, PFS=0.44%, PFU=1.9%, gate=Fail.
- XGB direct N=1000: beta_FS_cons=2.33, PFS=0.50%, PFU=1.5%, gate=Fail.
- XGB direct N=2000: beta_FS_cons=2.35, PFS=0.50%, PFU=1.4%, gate=Fail.

## N=2000 interval sensitivity

- LGBM direct level 0.900: beta_FS_cons=2.36, PFU=1.2%, gate=Fail.
- LGBM direct level 0.950: beta_FS_cons=2.43, PFU=1.5%, gate=Fail.
- LGBM direct level 0.975: beta_FS_cons=2.56, PFU=1.9%, gate=Pass.
- MLP scratch level 0.900: beta_FS_cons=2.10, PFU=2.6%, gate=Fail.
- MLP scratch level 0.950: beta_FS_cons=2.10, PFU=3.5%, gate=Fail.
- MLP scratch level 0.975: beta_FS_cons=2.10, PFU=4.6%, gate=Fail.
- Ridge direct level 0.900: beta_FS_cons=2.65, PFU=12.2%, gate=Pass.
- Ridge direct level 0.950: beta_FS_cons=3.16, PFU=14.4%, gate=Pass.
- Ridge direct level 0.975: beta_FS_cons=4.68, PFU=16.1%, gate=Pass.
- XGB direct level 0.900: beta_FS_cons=2.35, PFU=1.4%, gate=Fail.
- XGB direct level 0.950: beta_FS_cons=2.39, PFU=1.6%, gate=Fail.
- XGB direct level 0.975: beta_FS_cons=2.54, PFU=1.9%, gate=Pass.
