"""R35-E1b: partition-ensemble eligibility at the widened 0.975 interval.

Companion to 75_r35_partition_stability.py (level 0.90): does interval
widening restore supermajority (>=80% of partitions) admissibility, and at
what false-unsafe cost?
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import r35_harness as H

OUT = H.ROOT / "outputs" / "high_target" / "r35_partition_stability"
ROUND = H.ROOT / "rounds" / "R34_figure_style_normalization_2026-06-10"

MODELS = ["ridge_direct", "lgbm_direct", "xgb_direct", "scratch_mlp"]
BUDGETS = [500, 2000]
N_REPS = 5
N_PARTITIONS = 24
BETA_TARGET = 2.5
PFU_MAX = 0.30
ALPHA_WIDE = 0.025
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
        rng_eval = np.random.default_rng(7000 + pseed)  # same rep seeds as E1
        rng_boot = np.random.default_rng(8500 + pseed)
        for rep in range(N_REPS):
            rep_seed = int(rng_eval.integers(1, 10 ** 8))
            for N in BUDGETS:
                if N > len(X_pool):
                    continue
                for model in MODELS:
                    met = H.run_cell(model, X_pool, y_pool, X_test, y_test,
                                     ev_test, N, rep_seed, rng_boot,
                                     alpha=ALPHA_WIDE)
                    met.update({"partition_seed": pseed, "rep": rep, "N": N,
                                "model": model, "alpha": ALPHA_WIDE})
                    rows.append(met)
        print(f"[75b] partition {p_i + 1}/{len(partition_seeds)} "
              f"({time.time() - t0:.0f}s)", flush=True)

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "partition_stability_wide_detail.csv", index=False)
    med = (detail.groupby(["partition_seed", "model", "N"])
           .agg(beta=("beta_fs_cons", "median"),
                pfu=("p_false_unsafe", "median"),
                cond=("conditional_miss_event_mean", "median"))
           .reset_index())
    med["plain_pass"] = med["beta"] >= BETA_TARGET
    med["guarded_pass"] = med["plain_pass"] & (med["pfu"] <= PFU_MAX)
    summary = (med.groupby(["model", "N"])
               .agg(plain_pass_fraction=("plain_pass", "mean"),
                    guarded_pass_fraction=("guarded_pass", "mean"),
                    beta_median=("beta", "median"),
                    beta_p05=("beta", lambda s: float(np.nanquantile(s, 0.05))),
                    beta_p95=("beta", lambda s: float(np.nanquantile(s, 0.95))),
                    pfu_median=("pfu", "median"),
                    cond_median=("cond", "median"))
               .reset_index())
    summary.to_csv(OUT / "partition_stability_wide_summary.csv", index=False)

    lines = ["# R35-E1b partition ensemble at widened interval (level 0.975)",
             "",
             "| model | N | plain pass | guarded pass | beta med [p05, p95] | P_FU med | cond. miss med |",
             "| --- | ---: | ---: | ---: | --- | ---: | ---: |"]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['model']} | {int(r['N'])} | {100 * r['plain_pass_fraction']:.0f}% | "
            f"{100 * r['guarded_pass_fraction']:.0f}% | "
            f"{r['beta_median']:.2f} [{r['beta_p05']:.2f}, {r['beta_p95']:.2f}] | "
            f"{r['pfu_median']:.3f} | {r['cond_median']:.3f} |")
    text = "\n".join(lines) + "\n"
    (OUT / "R35_E1B_WIDE_REPORT.md").write_text(text, encoding="utf-8")
    (ROUND / "R35_E1B_WIDE_REPORT.md").write_text(text, encoding="utf-8")
    print(f"[75b] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
