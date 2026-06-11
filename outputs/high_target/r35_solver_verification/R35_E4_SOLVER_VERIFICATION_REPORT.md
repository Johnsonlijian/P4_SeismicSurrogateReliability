# R35-E4 solver verification and degrading/P-Delta sensitivity

Scope: 16 protocol systems x 187 disjoint-test records (2992 pairs).

## Step-halving (dt 0.01 -> 0.005)

- Median relative peak-IDR difference: 1.20e-05
- p95: 4.92e-04; max: 2.34e-03
- 1% IDR exceedance classification agreement: 100.00% (0 flips of 2992)

## Cross-solver spot checks (RK45, rtol 1e-8)

- Median relative difference: 3.64e-06
- Max: 3.55e-05 over 12 pairs

## Degrading (delta=0.10) + P-Delta variant

- Spearman rho of peak IDR (pairs with IDR > 0.5%): 0.872
- 1% exceedance Jaccard overlap: 0.672
- Fraction of baseline-unsafe pairs still unsafe: 97.8%
- Baseline unsafe pairs: 92; variant unsafe: 132
