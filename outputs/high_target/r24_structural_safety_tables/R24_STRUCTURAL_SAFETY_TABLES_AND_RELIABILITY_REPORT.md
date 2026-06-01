# R24 Structural Safety tables and false-safe reliability index

## Manuscript contribution strengthened

The analysis upgrades the manuscript from a metric-comparison note to a reliability-oriented article. It maps interval screening errors to a false-safe probability and a reliability-style index, while preserving event-level separation and event-equal aggregation.

## Dataset/protocol facts used

| protocol              | purpose                                                                                                                       |   source_events |   target_fit_or_cal_events |   target_test_events |   target_fit_or_cal_components |   target_test_components |   event_overlap_cal_test |   N_target_labels |   test_rows_per_model_all_reps |   test_events_in_trace |   systems | recorded_population                    |
|:----------------------|:------------------------------------------------------------------------------------------------------------------------------|----------------:|---------------------------:|---------------------:|-------------------------------:|-------------------------:|-------------------------:|------------------:|-------------------------------:|-----------------------:|----------:|:---------------------------------------|
| main event-held-out   | Source-to-target transfer with target-event holdout; target calibration/test are component-level splits within target events. |              29 |                         15 |                   14 |                            186 |                      186 |                       14 |               500 |                          29760 |                     14 |        48 | 44 events, 1000 components, Mw 5.4-8.8 |
| event-disjoint target | Stricter stress test where target fit/calibration events and target-test events are mutually disjoint.                        |              29 |                          7 |                    8 |                            185 |                      187 |                        0 |               500 |                          29920 |                      8 |        48 | 44 events, 1000 components, Mw 5.4-8.8 |

## Main metric summary

| protocol              | model        |   event_count |   row_count | RMSE_log_event_mean_CI   | coverage_event_mean_CI   |   coverage_gap_vs_0p90 | interval_score_90_CI   |   q95_abs_log |   tail_ratio |   test_cal_resid_sd_ratio |
|:----------------------|:-------------|--------------:|------------:|:-------------------------|:-------------------------|-----------------------:|:-----------------------|--------------:|-------------:|--------------------------:|
| event-disjoint target | XGB direct   |             8 |       29920 | 0.162 [0.135, 0.189]     | 0.646 [0.553, 0.745]     |                 -0.254 | 0.914 [0.672, 1.183]   |         0.327 |         2.67 |                      1.71 |
| event-disjoint target | LGBM direct  |             8 |       29920 | 0.166 [0.139, 0.196]     | 0.684 [0.586, 0.771]     |                 -0.216 | 0.899 [0.637, 1.207]   |         0.329 |         2.81 |                      1.56 |
| event-disjoint target | HGB direct   |             8 |       29920 | 0.175 [0.150, 0.202]     | 0.636 [0.543, 0.729]     |                 -0.264 | 0.981 [0.724, 1.283]   |         0.338 |         2.9  |                      1.65 |
| event-disjoint target | RF direct    |             8 |       29920 | 0.195 [0.169, 0.221]     | 0.644 [0.545, 0.743]     |                 -0.256 | 1.056 [0.810, 1.311]   |         0.375 |         2.64 |                      1.62 |
| event-disjoint target | MLP finetune |             8 |       29920 | 0.244 [0.166, 0.338]     | 0.696 [0.566, 0.820]     |                 -0.204 | 1.585 [0.796, 2.577]   |         0.795 |         4.08 |                      2.72 |
| event-disjoint target | Ridge direct |             8 |       29920 | 0.376 [0.297, 0.468]     | 0.489 [0.333, 0.650]     |                 -0.411 | 2.532 [1.487, 3.830]   |         0.909 |         2.76 |                      1.88 |
| event-disjoint target | MLP scratch  |             8 |       29920 | 0.476 [0.240, 0.775]     | 0.660 [0.491, 0.821]     |                 -0.24  | 3.991 [1.008, 7.777]   |         2.106 |         6.08 |                      3.74 |
| main event-held-out   | XGB direct   |            14 |       29760 | 0.156 [0.137, 0.178]     | 0.706 [0.615, 0.787]     |                 -0.194 | 0.820 [0.642, 1.025]   |         0.305 |         3.1  |                      1.44 |
| main event-held-out   | LGBM direct  |            14 |       29760 | 0.161 [0.140, 0.183]     | 0.748 [0.677, 0.814]     |                 -0.152 | 0.801 [0.617, 1.023]   |         0.317 |         3.13 |                      1.33 |
| main event-held-out   | HGB direct   |            14 |       29760 | 0.167 [0.145, 0.190]     | 0.701 [0.618, 0.778]     |                 -0.199 | 0.890 [0.690, 1.101]   |         0.33  |         3.24 |                      1.47 |
| main event-held-out   | RF direct    |            14 |       29760 | 0.170 [0.147, 0.195]     | 0.752 [0.662, 0.833]     |                 -0.148 | 0.841 [0.660, 1.059]   |         0.333 |         3.13 |                      1.33 |
| main event-held-out   | MLP finetune |            14 |       29760 | 0.182 [0.154, 0.209]     | 0.820 [0.736, 0.892]     |                 -0.08  | 0.853 [0.698, 1.023]   |         0.373 |         3.17 |                      1.07 |
| main event-held-out   | MLP scratch  |            14 |       29760 | 0.183 [0.157, 0.211]     | 0.892 [0.836, 0.936]     |                 -0.008 | 0.812 [0.693, 0.987]   |         0.382 |         3.08 |                      0.91 |
| main event-held-out   | Ridge direct |            14 |       29760 | 0.235 [0.180, 0.294]     | 0.763 [0.640, 0.871]     |                 -0.137 | 1.191 [0.882, 1.628]   |         0.452 |         2.82 |                      1.16 |

## Decision-risk reliability summary at 1% IDR

| protocol              | IDR_threshold   |   false_safe_cost | winner       |   true_exceed |   false_safe |   false_safe_U95 |   beta_FS |   beta_FS_cons |   false_unsafe |   expected_loss |
|:----------------------|:----------------|------------------:|:-------------|--------------:|-------------:|-----------------:|----------:|---------------:|---------------:|----------------:|
| event-disjoint target | 1.0%            |                 1 | XGB direct   |         0.018 |       0.0047 |           0.0109 |      2.6  |           2.29 |         0.0166 |           0.021 |
| event-disjoint target | 1.0%            |                10 | LGBM direct  |         0.018 |       0.0031 |           0.0072 |      2.73 |           2.45 |         0.0202 |           0.051 |
| event-disjoint target | 1.0%            |                50 | LGBM direct  |         0.018 |       0.0031 |           0.0072 |      2.73 |           2.45 |         0.0202 |           0.177 |
| event-disjoint target | 1.0%            |               100 | Ridge direct |         0.018 |       0.0016 |           0.0042 |      2.95 |           2.64 |         0.1155 |           0.273 |
| main event-held-out   | 1.0%            |                 1 | XGB direct   |         0.034 |       0.0031 |           0.0062 |      2.73 |           2.5  |         0.0335 |           0.037 |
| main event-held-out   | 1.0%            |                10 | LGBM direct  |         0.034 |       0.0021 |           0.0047 |      2.86 |           2.6  |         0.0397 |           0.061 |
| main event-held-out   | 1.0%            |                50 | LGBM direct  |         0.034 |       0.0021 |           0.0047 |      2.86 |           2.6  |         0.0397 |           0.147 |
| main event-held-out   | 1.0%            |               100 | MLP scratch  |         0.034 |       0.0015 |           0.0028 |      2.98 |           2.77 |         0.0796 |           0.225 |

## Event-level residual variance decomposition

| protocol              | model               | model_label   |   event_count |   between_event_residual_var |   within_event_residual_var |   between_event_share |   p90_abs_event_mean_residual |   mean_event_rmse |
|:----------------------|:--------------------|:--------------|--------------:|-----------------------------:|----------------------------:|----------------------:|------------------------------:|------------------:|
| event-disjoint target | scratch_mlp         | MLP scratch   |             8 |                   0.2237     |                   0.180038  |              0.554072 |                      0.741102 |          0.475604 |
| event-disjoint target | ridge_direct        | Ridge direct  |             8 |                   0.0381105  |                   0.0445251 |              0.461187 |                      0.463254 |          0.376388 |
| event-disjoint target | rf_direct           | RF direct     |             8 |                   0.0187411  |                   0.0231943 |              0.446905 |                      0.200349 |          0.19531  |
| event-disjoint target | xgb_direct          | XGB direct    |             8 |                   0.0110733  |                   0.0176146 |              0.385993 |                      0.158611 |          0.161528 |
| event-disjoint target | hgb_direct          | HGB direct    |             8 |                   0.0126746  |                   0.0202404 |              0.38507  |                      0.172238 |          0.17485  |
| event-disjoint target | lgbm_direct         | LGBM direct   |             8 |                   0.0104676  |                   0.0190707 |              0.354373 |                      0.166332 |          0.165754 |
| event-disjoint target | pretrained_finetune | MLP finetune  |             8 |                   0.011407   |                   0.0634921 |              0.152299 |                      0.154465 |          0.243517 |
| main event-held-out   | ridge_direct        | Ridge direct  |            14 |                   0.0287666  |                   0.0321203 |              0.47246  |                      0.328738 |          0.235041 |
| main event-held-out   | hgb_direct          | HGB direct    |            14 |                   0.0111302  |                   0.0187836 |              0.372077 |                      0.177019 |          0.167238 |
| main event-held-out   | lgbm_direct         | LGBM direct   |            14 |                   0.0101795  |                   0.0174569 |              0.368337 |                      0.177496 |          0.160797 |
| main event-held-out   | rf_direct           | RF direct     |            14 |                   0.0106138  |                   0.0206179 |              0.339841 |                      0.172481 |          0.169868 |
| main event-held-out   | xgb_direct          | XGB direct    |            14 |                   0.00885195 |                   0.0175744 |              0.334967 |                      0.143212 |          0.156305 |
| main event-held-out   | pretrained_finetune | MLP finetune  |            14 |                   0.0114361  |                   0.0251105 |              0.312919 |                      0.158069 |          0.182354 |
| main event-held-out   | scratch_mlp         | MLP scratch   |            14 |                   0.00775538 |                   0.0251805 |              0.235469 |                      0.164336 |          0.182985 |

## False-safe gate decision summary

| protocol              | model_label   |   threshold_idr |   false_safe_rate |   false_safe_ci95_hi |   beta_false_safe |   beta_false_safe_cons |   gate_target_beta | gate_status   | action                                               |
|:----------------------|:--------------|----------------:|------------------:|---------------------:|------------------:|-----------------------:|-------------------:|:--------------|:-----------------------------------------------------|
| event-disjoint target | Ridge direct  |            0.01 |        0.00157722 |           0.00415874 |           2.95227 |                2.6389  |                2.5 | Pass          | eligible; select by loss/PFU                         |
| event-disjoint target | LGBM direct   |            0.01 |        0.00312726 |           0.00724185 |           2.73413 |                2.44504 |                2.5 | Fail          | collect labels, widen interval, or use NTHA fallback |
| event-disjoint target | MLP finetune  |            0.01 |        0.00420686 |           0.00840183 |           2.635   |                2.39098 |                2.5 | Fail          | collect labels, widen interval, or use NTHA fallback |
| event-disjoint target | XGB direct    |            0.01 |        0.00471411 |           0.0108865  |           2.59612 |                2.2943  |                2.5 | Fail          | collect labels, widen interval, or use NTHA fallback |
| event-disjoint target | MLP scratch   |            0.01 |        0.00609658 |           0.0178935  |           2.5065  |                2.09934 |                2.5 | Fail          | collect labels, widen interval, or use NTHA fallback |
| main event-held-out   | MLP scratch   |            0.01 |        0.00145524 |           0.00282931 |           2.97704 |                2.76693 |                2.5 | Pass          | eligible; select by loss/PFU                         |
| main event-held-out   | LGBM direct   |            0.01 |        0.00214868 |           0.00467116 |           2.85547 |                2.59927 |                2.5 | Pass          | eligible; select by loss/PFU                         |
| main event-held-out   | Ridge direct  |            0.01 |        0.00194077 |           0.00518648 |           2.88763 |                2.56314 |                2.5 | Pass          | eligible; select by loss/PFU                         |
| main event-held-out   | XGB direct    |            0.01 |        0.00313875 |           0.00617257 |           2.73292 |                2.50212 |                2.5 | Pass          | eligible; select by loss/PFU                         |
| main event-held-out   | MLP finetune  |            0.01 |        0.0027536  |           0.00635469 |           2.77576 |                2.49181 |                2.5 | Fail          | collect labels, widen interval, or use NTHA fallback |

## False-safe reliability winners by protocol at 1% IDR

| protocol | highest beta_FS model | beta_FS | false-safe | false-unsafe |
| --- | --- | ---: | ---: | ---: |
| event-disjoint target | Ridge direct | 2.95 | 0.0016 | 0.1155 |
| main event-held-out | MLP scratch | 2.98 | 0.0015 | 0.0796 |

## Boundary

The reliability index is an empirical diagnostic for screening-rule false-safe probability, not a replacement for component/system-level code calibration or PBEE/FEMA P-58 loss assessment. Because the rates are event-equal and clustered, the conservative beta_FS column uses an event-bootstrap upper 95% false-safe bound rather than an independent Bernoulli assumption.

## Generated files

- Tables: `R:\NAS_DRIVE\IMUT\1-Research_Output\1-Papers\1_In_Preparation\2026-Claude\projects\P4_SeismicFoundationModel\outputs\high_target\r24_structural_safety_tables`
- Figure: `R:\NAS_DRIVE\IMUT\1-Research_Output\1-Papers\1_In_Preparation\2026-Claude\projects\P4_SeismicFoundationModel\outputs\figures\high_target\fig_r24_false_safe_reliability_index.pdf`
- LaTeX fragments: `R:\NAS_DRIVE\IMUT\1-Research_Output\1-Papers\1_In_Preparation\2026-Claude\projects\P4_SeismicFoundationModel\submission\structural_safety_2026-06-01\latex_source_flat`