# R30 conditional-coverage diagnostic by event-level PGA bin

This diagnostic uses the R28 true event-disjoint N=2000 prediction export and the test split only.
Coverage is aggregated event-equal within each event-level PGA bin to avoid row-level pseudo-replication.

- Worst median bin coverage: XGB direct / high PGA = 42.6%.
- Highest median bin coverage: LGBM direct / mid PGA = 69.9%.
- The result is a descriptive stress check, not a formal conditional conformal guarantee.

## Summary table

| model_label   | pga_bin   |   median_coverage |   q05_coverage |   q95_coverage |   min_rep_coverage |   median_event_count |
|:--------------|:----------|------------------:|---------------:|---------------:|-------------------:|---------------------:|
| Ridge direct  | low PGA   |          0.488368 |       0.378646 |       0.521684 |           0.352083 |                    3 |
| Ridge direct  | mid PGA   |          0.443229 |       0.411302 |       0.496458 |           0.405208 |                    2 |
| Ridge direct  | high PGA  |          0.569932 |       0.541888 |       0.589555 |           0.539807 |                    3 |
| LGBM direct   | low PGA   |          0.592535 |       0.508385 |       0.634774 |           0.485417 |                    3 |
| LGBM direct   | mid PGA   |          0.698698 |       0.669063 |       0.73612  |           0.6625   |                    2 |
| LGBM direct   | high PGA  |          0.436862 |       0.397689 |       0.497871 |           0.392649 |                    3 |
| XGB direct    | low PGA   |          0.528299 |       0.459931 |       0.561979 |           0.439931 |                    3 |
| XGB direct    | mid PGA   |          0.680208 |       0.595    |       0.721458 |           0.56875  |                    2 |
| XGB direct    | high PGA  |          0.426136 |       0.386489 |       0.498463 |           0.375137 |                    3 |
| MLP scratch   | low PGA   |          0.681944 |       0.613837 |       0.748316 |           0.599306 |                    3 |
| MLP scratch   | mid PGA   |          0.534375 |       0.432734 |       0.682266 |           0.415625 |                    2 |
| MLP scratch   | high PGA  |          0.676191 |       0.629545 |       0.728035 |           0.629007 |                    3 |