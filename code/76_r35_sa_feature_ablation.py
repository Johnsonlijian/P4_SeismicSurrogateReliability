"""R35-E2: Sa(T1)/Sa_avg feature ablation.

Reviewer kill-issue: the submitted feature set omits spectral ordinates, so
the residual-geometry conclusions could be artifacts of an information-starved
surrogate. This experiment recomputes the pipeline with two paired feature
sets - base (8 IM + 6 structural) and spectral (base + Sa(T1) + Sa_avg over
0.2T1-3T1, Newmark linear SDOF, zeta=5%) - using identical replicate seeds so
every contrast is paired.

Scope:
- event-disjoint protocol (original partition): 4 focus models x budgets
  {50,100,250,500,1000,2000} x 10 reps;
- main event-held-out protocol: 7 families (incl. source-pretrained MLP)
  at N=500 x 10 reps.
Outputs: paired summaries, filter-verdict flip table, RMSE gains.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

import r35_harness as H

OUT = H.ROOT / "outputs" / "high_target" / "r35_sa_ablation"
ROUND = H.ROOT / "rounds" / "R34_figure_style_normalization_2026-06-10"

FOCUS = ["ridge_direct", "lgbm_direct", "xgb_direct", "scratch_mlp"]
ALL7 = ["ridge_direct", "rf_direct", "hgb_direct", "xgb_direct",
        "lgbm_direct", "scratch_mlp", "pretrained_finetune"]
BUDGETS = [50, 100, 250, 500, 1000, 2000]
N_REPS = 10
BETA_TARGET = 2.5
PFU_MAX = 0.30


def run_protocol(name, X_pool, y_pool, X_test, y_test, ev_test, models,
                 budgets, featureset, rep_seeds, rng_boot,
                 foundation=None, sc=None) -> list[dict]:
    rows = []
    for rep, rep_seed in enumerate(rep_seeds):
        for N in budgets:
            if N > len(X_pool):
                continue
            for model in models:
                met = H.run_cell(model, X_pool, y_pool, X_test, y_test,
                                 ev_test, N, rep_seed, rng_boot,
                                 foundation=foundation, sc=sc)
                met.update({"protocol": name, "featureset": featureset,
                            "rep": rep, "N": N, "model": model})
                rows.append(met)
    return rows


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    records = H.load_records()
    labels = H.load_labels()
    spectra = H.compute_spectra(cache_path=OUT / "spectra_cache.csv")
    print(f"[76] spectra ready ({time.time() - t0:.0f}s)", flush=True)

    source_events, target_events, source_gm, pool_gm, test_gm = H.split_events(records)
    fit_gm_d, test_gm_d, _, _ = H.disjoint_partition(records, target_events)

    rep_seeds = [int(s) for s in
                 np.random.default_rng(20260603).integers(1, 10 ** 8, N_REPS)]
    rows: list[dict] = []

    for featureset, spec in [("base", None), ("spectral", spectra)]:
        # --- event-disjoint, focus models, all budgets.
        _, X_pool, y_pool = H.merge_features(labels, records, gm_ids=fit_gm_d,
                                             source_period="target", spectra=spec)
        test_df, X_test, y_test = H.merge_features(labels, records,
                                                   gm_ids=test_gm_d,
                                                   source_period="target",
                                                   spectra=spec)
        ev_test = test_df["event_id"].to_numpy()
        rng_boot = np.random.default_rng(31000)
        rows += run_protocol("event-disjoint target", X_pool, y_pool, X_test,
                             y_test, ev_test, FOCUS, BUDGETS, featureset,
                             rep_seeds, rng_boot)
        print(f"[76] disjoint/{featureset} done ({time.time() - t0:.0f}s)", flush=True)

        # --- main event-held-out, 7 families, N=500.
        _, X_pre, y_pre = H.merge_features(labels, records, gm_ids=source_gm,
                                           source_period="seen", spectra=spec)
        _, X_val, y_val = H.merge_features(labels, records, gm_ids=pool_gm,
                                           source_period="seen", spectra=spec)
        foundation, sc, val_r2 = H.train_mod.fit_foundation(
            X_pre, y_pre, X_val, y_val, seed=20260601)
        print(f"[76] foundation/{featureset} val R2 = {val_r2:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        _, X_pool_m, y_pool_m = H.merge_features(labels, records,
                                                 gm_ids=pool_gm,
                                                 source_period="target",
                                                 spectra=spec)
        test_df_m, X_test_m, y_test_m = H.merge_features(labels, records,
                                                         gm_ids=test_gm,
                                                         source_period="target",
                                                         spectra=spec)
        ev_test_m = test_df_m["event_id"].to_numpy()
        rng_boot_m = np.random.default_rng(32000)
        rows += run_protocol("main event-held-out", X_pool_m, y_pool_m,
                             X_test_m, y_test_m, ev_test_m, ALL7, [500],
                             featureset, rep_seeds, rng_boot_m,
                             foundation=foundation, sc=sc)
        print(f"[76] main/{featureset} done ({time.time() - t0:.0f}s)", flush=True)

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "sa_ablation_detail.csv", index=False)

    summary = (detail.groupby(["protocol", "featureset", "model", "N"])
               .agg(rmse_median=("rmse_log", "median"),
                    coverage_median=("coverage_event_equal", "median"),
                    beta_median=("beta_fs_cons", "median"),
                    pfu_median=("p_false_unsafe", "median"),
                    cond_median=("conditional_miss_event_mean", "median"),
                    q_median=("q_log10", "median"),
                    reps=("rep", "nunique"))
               .reset_index())
    summary["plain_filter"] = np.where(summary["beta_median"] >= BETA_TARGET,
                                       "Pass", "Fail")
    summary["guarded_filter"] = np.where(
        (summary["beta_median"] >= BETA_TARGET)
        & (summary["pfu_median"] <= PFU_MAX), "Pass",
        np.where(summary["beta_median"] >= BETA_TARGET, "Vacuous/Fail-guard",
                 "Fail"))
    summary.to_csv(OUT / "sa_ablation_summary.csv", index=False)

    # Paired verdict-flip table on the event-disjoint protocol.
    dis = summary[summary["protocol"].eq("event-disjoint target")]
    base = dis[dis["featureset"].eq("base")].set_index(["model", "N"])
    spec_s = dis[dis["featureset"].eq("spectral")].set_index(["model", "N"])
    flips = []
    for key in base.index:
        b, s = base.loc[key], spec_s.loc[key]
        flips.append({
            "model": key[0], "N": key[1],
            "rmse_base": b["rmse_median"], "rmse_spectral": s["rmse_median"],
            "rmse_gain_pct": 100 * (b["rmse_median"] - s["rmse_median"]) / b["rmse_median"],
            "beta_base": b["beta_median"], "beta_spectral": s["beta_median"],
            "cond_base": b["cond_median"], "cond_spectral": s["cond_median"],
            "filter_base": b["plain_filter"], "filter_spectral": s["plain_filter"],
            "verdict_flip": b["plain_filter"] != s["plain_filter"],
        })
    flips_df = pd.DataFrame(flips).sort_values(["model", "N"])
    flips_df.to_csv(OUT / "sa_ablation_verdict_flips.csv", index=False)

    lines = ["# R35-E2 Sa(T1)/Sa_avg feature ablation", "",
             f"Paired reps (N_reps={N_REPS}, identical seeds). beta* = "
             f"{BETA_TARGET}; spectral features: Sa(T1), Sa_avg(0.2T1-3T1), "
             "Newmark linear SDOF zeta=5% from the stored 1000-record array.", "",
             "## Event-disjoint verdict comparison", "",
             "| model | N | RMSE base->spec (gain%) | beta base->spec | cond.miss base->spec | filter base->spec | flip |",
             "| --- | ---: | --- | --- | --- | --- | --- |"]
    for _, r in flips_df.iterrows():
        lines.append(
            f"| {r['model']} | {int(r['N'])} | {r['rmse_base']:.3f}->"
            f"{r['rmse_spectral']:.3f} ({r['rmse_gain_pct']:+.1f}%) | "
            f"{r['beta_base']:.2f}->{r['beta_spectral']:.2f} | "
            f"{r['cond_base']:.2f}->{r['cond_spectral']:.2f} | "
            f"{r['filter_base']}->{r['filter_spectral']} | "
            f"{'YES' if r['verdict_flip'] else 'no'} |")
    lines += ["", "## Main protocol (N=500, 7 families)", "",
              "| model | featureset | RMSE | coverage | beta | cond. miss |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    main_s = summary[summary["protocol"].eq("main event-held-out")]
    for _, r in main_s.sort_values(["model", "featureset"]).iterrows():
        lines.append(f"| {r['model']} | {r['featureset']} | "
                     f"{r['rmse_median']:.3f} | {r['coverage_median']:.3f} | "
                     f"{r['beta_median']:.2f} | {r['cond_median']:.3f} |")
    text = "\n".join(lines) + "\n"
    (OUT / "R35_E2_SA_ABLATION_REPORT.md").write_text(text, encoding="utf-8")
    (ROUND / "R35_E2_SA_ABLATION_REPORT.md").write_text(text, encoding="utf-8")
    print(f"[76] done in {time.time() - t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
