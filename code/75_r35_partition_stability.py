"""R35-E1: filter-verdict stability over >= 24 random event partitions.

Every eligibility verdict in the submitted/v25 manuscript is conditional on a
single 7-fit/8-test partition (seed 20260613) of the 15 target events. This
experiment regenerates the full pipeline for 24 partitions (the original plus
23 fresh seeds), 4 focus models, budgets {500, 2000}, 5 replicates each, and
reports: pass fractions for the plain and guarded filters, beta_FS,cons
spread, and whether the LGBM N=1000->2000 eligibility reversal (here proxied
by the N=500 vs N=2000 contrast) exceeds partition variability.
"""
from __future__ import annotations

import time
from pathlib import Path

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
ORIGINAL_SEED = 20260613


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    records = H.load_records()
    labels = H.load_labels()
    _, target_events, _, _, _ = H.split_events(records)

    partition_seeds = [ORIGINAL_SEED] + [9100 + i for i in range(N_PARTITIONS - 1)]
    rows = []
    for p_i, pseed in enumerate(partition_seeds):
        fit_gm, test_gm, fit_events, test_events = H.disjoint_partition(
            records, target_events, seed=pseed)
        pool_df, X_pool, y_pool = H.merge_features(labels, records,
                                                   gm_ids=fit_gm,
                                                   source_period="target")
        test_df, X_test, y_test = H.merge_features(labels, records,
                                                   gm_ids=test_gm,
                                                   source_period="target")
        ev_test = test_df["event_id"].to_numpy()
        rng_eval = np.random.default_rng(7000 + pseed)
        rng_boot = np.random.default_rng(8000 + pseed)
        for rep in range(N_REPS):
            rep_seed = int(rng_eval.integers(1, 10 ** 8))
            for N in BUDGETS:
                if N > len(X_pool):
                    continue
                for model in MODELS:
                    met = H.run_cell(model, X_pool, y_pool, X_test, y_test,
                                     ev_test, N, rep_seed, rng_boot)
                    met.update({
                        "partition_seed": pseed,
                        "partition_index": p_i,
                        "is_original_partition": pseed == ORIGINAL_SEED,
                        "rep": rep, "N": N, "model": model,
                        "n_fit_events": len(fit_events),
                        "n_test_events": len(test_events),
                    })
                    rows.append(met)
        print(f"[75] partition {p_i + 1}/{len(partition_seeds)} done "
              f"({time.time() - t0:.0f}s)", flush=True)

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "partition_stability_detail.csv", index=False)

    # Per partition x model x N: median over reps, then verdicts.
    med = (detail.groupby(["partition_seed", "model", "N"])
           .agg(beta=("beta_fs_cons", "median"),
                pfu=("p_false_unsafe", "median"),
                cond=("conditional_miss_event_mean", "median"),
                unsafe_rows=("n_unsafe_rows", "median"))
           .reset_index())
    med["plain_pass"] = med["beta"] >= BETA_TARGET
    med["guarded_pass"] = med["plain_pass"] & (med["pfu"] <= PFU_MAX)
    med.to_csv(OUT / "partition_stability_partition_medians.csv", index=False)

    summary = (med.groupby(["model", "N"])
               .agg(plain_pass_fraction=("plain_pass", "mean"),
                    guarded_pass_fraction=("guarded_pass", "mean"),
                    beta_median=("beta", "median"),
                    beta_p05=("beta", lambda s: float(np.quantile(s, 0.05))),
                    beta_p95=("beta", lambda s: float(np.quantile(s, 0.95))),
                    pfu_median=("pfu", "median"),
                    cond_median=("cond", "median"),
                    partitions=("partition_seed", "nunique"))
               .reset_index())
    summary.to_csv(OUT / "partition_stability_summary.csv", index=False)

    # Budget contrast per partition (does beta drop from N=500 to N=2000?).
    wide = med.pivot_table(index=["partition_seed", "model"], columns="N",
                           values="beta").reset_index()
    wide["delta_beta_2000_minus_500"] = wide[2000] - wide[500]
    contrast = (wide.groupby("model")
                .agg(frac_beta_drops=("delta_beta_2000_minus_500",
                                      lambda s: float((s < 0).mean())),
                     delta_median=("delta_beta_2000_minus_500", "median"),
                     delta_p05=("delta_beta_2000_minus_500",
                                lambda s: float(np.quantile(s, 0.05))),
                     delta_p95=("delta_beta_2000_minus_500",
                                lambda s: float(np.quantile(s, 0.95))))
                .reset_index())
    contrast.to_csv(OUT / "partition_budget_contrast.csv", index=False)

    lines = ["# R35-E1 partition stability", "",
             f"{len(partition_seeds)} partitions (original {ORIGINAL_SEED} + "
             f"{len(partition_seeds) - 1} fresh), {N_REPS} reps, budgets "
             f"{BUDGETS}, beta* = {BETA_TARGET}, P_FU,max = {PFU_MAX}.", "",
             "## Pass fractions over partitions (per-partition rep-medians)", "",
             "| model | N | plain pass | guarded pass | beta med [p05, p95] | P_FU med | cond. miss med |",
             "| --- | ---: | ---: | ---: | --- | ---: | ---: |"]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['model']} | {int(r['N'])} | {100 * r['plain_pass_fraction']:.0f}% | "
            f"{100 * r['guarded_pass_fraction']:.0f}% | "
            f"{r['beta_median']:.2f} [{r['beta_p05']:.2f}, {r['beta_p95']:.2f}] | "
            f"{r['pfu_median']:.3f} | {r['cond_median']:.3f} |")
    lines += ["", "## N=500 -> N=2000 beta contrast per partition", "",
              "| model | fraction of partitions with beta drop | median delta | [p05, p95] |",
              "| --- | ---: | ---: | --- |"]
    for _, r in contrast.iterrows():
        lines.append(f"| {r['model']} | {100 * r['frac_beta_drops']:.0f}% | "
                     f"{r['delta_median']:.2f} | [{r['delta_p05']:.2f}, "
                     f"{r['delta_p95']:.2f}] |")
    text = "\n".join(lines) + "\n"
    (OUT / "R35_E1_PARTITION_STABILITY_REPORT.md").write_text(text, encoding="utf-8")
    (ROUND / "R35_E1_PARTITION_STABILITY_REPORT.md").write_text(text, encoding="utf-8")
    print(f"[75] done in {time.time() - t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
