"""R35-E1c: partition-ensemble eligibility resolved by threshold.

One fit per (partition, rep, N, model); metrics evaluated at IDR thresholds
0.5%, 1%, 2% from the same predictions. Answers: where does the ensemble
filter ever certify stably - i.e., is instability driven by evidence mass
(exceedance prevalence) rather than by the filter itself?
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd

import r35_harness as H

OUT = H.ROOT / "outputs" / "high_target" / "r35_partition_stability"
ROUND = H.ROOT / "rounds" / "R34_figure_style_normalization_2026-06-10"

MODELS = ["ridge_direct", "lgbm_direct", "xgb_direct", "scratch_mlp"]
BUDGETS = [500, 2000]
THRESHOLDS = [0.005, 0.01, 0.02]
N_REPS = 5
N_PARTITIONS = 24
BETA_TARGET = 2.5
PFU_MAX = 0.30
ORIGINAL_SEED = 20260613


def main() -> None:
    t0 = time.time()
    records = H.load_records()
    labels = H.load_labels()
    _, target_events, _, _, _ = H.split_events(records)
    partition_seeds = [ORIGINAL_SEED] + [9100 + i for i in range(N_PARTITIONS - 1)]
    rows = []
    for p_i, pseed in enumerate(partition_seeds):
        fit_gm, test_gm, _, _ = H.disjoint_partition(records, target_events,
                                                     seed=pseed)
        _, X_pool, y_pool = H.merge_features(labels, records, gm_ids=fit_gm,
                                             source_period="target")
        test_df, X_test, y_test = H.merge_features(labels, records,
                                                   gm_ids=test_gm,
                                                   source_period="target")
        ev_test = test_df["event_id"].to_numpy()
        rng_eval = np.random.default_rng(7000 + pseed)
        rng_boot = np.random.default_rng(8700 + pseed)
        for rep in range(N_REPS):
            rep_seed = int(rng_eval.integers(1, 10 ** 8))
            for N in BUDGETS:
                if N > len(X_pool):
                    continue
                rng = np.random.default_rng(rep_seed)
                sel = rng.choice(np.arange(len(X_pool)), size=N, replace=False)
                X_sel, y_sel = X_pool[sel], y_pool[sel]
                half = N // 2
                perm = rng.permutation(N)
                tr, ca = perm[:half], perm[half:]
                for model in MODELS:
                    pred_calib, pred_test = H.fit_and_predict(
                        model, X_sel[tr], y_sel[tr], X_sel[ca], X_test, rep_seed)
                    q = H.conformal_quantile(np.abs(y_sel[ca] - pred_calib), 0.10)
                    for thr in THRESHOLDS:
                        met = H.decision_metrics(y_test, pred_test, q, ev_test,
                                                 rng_boot,
                                                 log_thr=math.log10(thr))
                        met.update({"partition_seed": pseed, "rep": rep,
                                    "N": N, "model": model,
                                    "threshold_idr": thr})
                        rows.append(met)
        print(f"[75c] partition {p_i + 1}/{len(partition_seeds)} "
              f"({time.time() - t0:.0f}s)", flush=True)

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "partition_thresholds_detail.csv", index=False)
    med = (detail.groupby(["partition_seed", "model", "N", "threshold_idr"])
           .agg(beta=("beta_fs_cons", "median"),
                pfu=("p_false_unsafe", "median"),
                cond=("conditional_miss_event_mean", "median"),
                unsafe=("n_unsafe_rows", "median"),
                ev_unsafe=("n_events_with_unsafe", "median"))
           .reset_index())
    med["guarded_pass"] = (med["beta"] >= BETA_TARGET) & (med["pfu"] <= PFU_MAX)
    summary = (med.groupby(["model", "N", "threshold_idr"])
               .agg(guarded_pass_fraction=("guarded_pass", "mean"),
                    beta_median=("beta", "median"),
                    cond_median=("cond", "median"),
                    unsafe_rows_median=("unsafe", "median"),
                    events_with_unsafe_median=("ev_unsafe", "median"))
               .reset_index())
    summary.to_csv(OUT / "partition_thresholds_summary.csv", index=False)

    lines = ["# R35-E1c threshold-resolved partition-ensemble eligibility", "",
             f"24 partitions, 5 reps, guarded filter (beta* = {BETA_TARGET}, "
             f"P_FU,max = {PFU_MAX}), level 0.90.", "",
             "| model | N | tau | guarded pass | beta med | cond. miss med | unsafe rows med | events w/ unsafe med |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for _, r in summary.sort_values(["threshold_idr", "model", "N"]).iterrows():
        lines.append(
            f"| {r['model']} | {int(r['N'])} | {100 * r['threshold_idr']:g}% | "
            f"{100 * r['guarded_pass_fraction']:.0f}% | {r['beta_median']:.2f} | "
            f"{r['cond_median']:.3f} | {int(r['unsafe_rows_median'])} | "
            f"{r['events_with_unsafe_median']:.0f} |")
    text = "\n".join(lines) + "\n"
    (OUT / "R35_E1C_THRESHOLDS_REPORT.md").write_text(text, encoding="utf-8")
    (ROUND / "R35_E1C_THRESHOLDS_REPORT.md").write_text(text, encoding="utf-8")
    print(f"[75c] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
