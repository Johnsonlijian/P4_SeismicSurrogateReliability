"""R35-E3: empirical coverage of the event-bootstrap U95 bound at 8 events.

Reviewer kill-issue: the 'conservative' percentile-bootstrap upper bound rests
on 8 test events and may undercover for skewed rare-event means. This study
treats all 37 non-fit events (the 8 original disjoint test events plus the 29
source events, never seen by the direct models) as the ground-truth
population: models are fitted/calibrated on the original 7 fit events, the
population event-equal P_FS at 1% IDR is computed over the 37 events, and
500 random 8-event test subsets per replicate measure how often the
subset-computed bound covers the population value.

Bounds compared: plain U95; U95 with a row-level rule-of-three floor applied
when the subset shows zero false-safe rows; U99.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd

import r35_harness as H

OUT = H.ROOT / "outputs" / "high_target" / "r35_bound_coverage"
ROUND = H.ROOT / "rounds" / "R34_figure_style_normalization_2026-06-10"

MODELS = ["ridge_direct", "lgbm_direct", "xgb_direct", "scratch_mlp"]
BUDGETS = [500, 2000]
N_REPS = 3
N_DRAWS = 500
N_BOOT = 2000
SUBSET_EVENTS = 8


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    records = H.load_records()
    labels = H.load_labels()
    _, target_events, _, _, _ = H.split_events(records)
    fit_gm, _, fit_events, _ = H.disjoint_partition(records, target_events)

    pool_df, X_pool, y_pool = H.merge_features(labels, records, gm_ids=fit_gm,
                                               source_period="target")
    pop_records = records[~records["event_id"].isin(fit_events)]
    pop_gm = sorted(pop_records["gm_id"].to_numpy(int).tolist())
    pop_df, X_popu, y_popu = H.merge_features(labels, records, gm_ids=pop_gm,
                                              source_period="target")
    ev_pop = pop_df["event_id"].to_numpy()
    events_unique = np.array(sorted(pd.unique(ev_pop)))
    print(f"[77] population: {len(events_unique)} events, {len(X_popu)} rows",
          flush=True)

    rep_seeds = [int(s) for s in
                 np.random.default_rng(20260615).integers(1, 10 ** 8, N_REPS)]
    rows = []
    for model in MODELS:
        for N in BUDGETS:
            for rep, rep_seed in enumerate(rep_seeds):
                rng = np.random.default_rng(rep_seed)
                sel = rng.choice(np.arange(len(X_pool)), size=N, replace=False)
                X_sel, y_sel = X_pool[sel], y_pool[sel]
                half = N // 2
                perm = rng.permutation(N)
                tr, ca = perm[:half], perm[half:]
                pred_calib, pred_pop = H.fit_and_predict(
                    model, X_sel[tr], y_sel[tr], X_sel[ca], X_popu, rep_seed)
                q = H.conformal_quantile(np.abs(y_sel[ca] - pred_calib), 0.10)
                fs = (y_popu > H.LOG1PCT) & ((pred_pop + q) <= H.LOG1PCT)
                per_event = (pd.DataFrame({"f": fs.astype(float), "e": ev_pop})
                             .groupby("e").agg(rate=("f", "mean"),
                                               rows=("f", "size")))
                per_event = per_event.reindex(events_unique)
                rates = per_event["rate"].to_numpy(float)
                rowcounts = per_event["rows"].to_numpy(float)
                pop_pfs = float(rates.mean())

                from scipy.stats import t as t_dist
                t95 = float(t_dist.ppf(0.95, SUBSET_EVENTS - 1))
                rng_draw = np.random.default_rng(50_000 + rep_seed)
                covered_u95 = covered_floor = covered_u99 = 0
                covered_t = covered_t_floor = 0
                u95_list = []
                for _ in range(N_DRAWS):
                    idx = rng_draw.choice(len(rates), size=SUBSET_EVENTS,
                                          replace=False)
                    sub = rates[idx]
                    boots = rng_draw.choice(sub, size=(N_BOOT, SUBSET_EVENTS),
                                            replace=True).mean(axis=1)
                    u95 = float(np.quantile(boots, 0.95))
                    u99 = float(np.quantile(boots, 0.99))
                    n_rows_sub = float(rowcounts[idx].sum())
                    floor = 3.0 / n_rows_sub if float(sub.sum()) == 0.0 else 0.0
                    u95f = max(u95, floor)
                    # Student-t upper bound on the event-rate mean.
                    ub_t = float(sub.mean()
                                 + t95 * sub.std(ddof=1) / np.sqrt(SUBSET_EVENTS))
                    ub_t_floor = max(ub_t, floor)
                    covered_u95 += u95 >= pop_pfs
                    covered_floor += u95f >= pop_pfs
                    covered_u99 += u99 >= pop_pfs
                    covered_t += ub_t >= pop_pfs
                    covered_t_floor += ub_t_floor >= pop_pfs
                    u95_list.append(u95)
                rows.append({
                    "model": model, "N": N, "rep": rep,
                    "q_log10": float(q),
                    "population_p_fs": pop_pfs,
                    "population_beta_cons_equiv": H.beta_from_p(pop_pfs),
                    "coverage_u95": covered_u95 / N_DRAWS,
                    "coverage_u95_floor": covered_floor / N_DRAWS,
                    "coverage_u99": covered_u99 / N_DRAWS,
                    "coverage_t": covered_t / N_DRAWS,
                    "coverage_t_floor": covered_t_floor / N_DRAWS,
                    "median_u95": float(np.median(u95_list)),
                })
            print(f"[77] {model} N={N} done ({time.time() - t0:.0f}s)",
                  flush=True)

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "bound_coverage_detail.csv", index=False)
    summary = (detail.groupby(["model", "N"])
               .agg(pop_pfs_median=("population_p_fs", "median"),
                    cov_u95=("coverage_u95", "mean"),
                    cov_u95_floor=("coverage_u95_floor", "mean"),
                    cov_u99=("coverage_u99", "mean"),
                    cov_t=("coverage_t", "mean"),
                    cov_t_floor=("coverage_t_floor", "mean"),
                    median_u95=("median_u95", "median"))
               .reset_index())
    summary.to_csv(OUT / "bound_coverage_summary.csv", index=False)

    lines = ["# R35-E3 bound-coverage study", "",
             f"Population = {len(events_unique)} non-fit events; "
             f"{N_DRAWS} random {SUBSET_EVENTS}-event subsets per rep, "
             f"{N_REPS} reps; nominal target 95%.", "",
             "| model | N | population P_FS | U95 coverage | U95+floor | U99 | t-bound | t+floor |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for _, r in summary.iterrows():
        lines.append(f"| {r['model']} | {int(r['N'])} | "
                     f"{r['pop_pfs_median']:.4f} | {100 * r['cov_u95']:.1f}% | "
                     f"{100 * r['cov_u95_floor']:.1f}% | "
                     f"{100 * r['cov_u99']:.1f}% | "
                     f"{100 * r['cov_t']:.1f}% | "
                     f"{100 * r['cov_t_floor']:.1f}% |")
    text = "\n".join(lines) + "\n"
    (OUT / "R35_E3_BOUND_COVERAGE_REPORT.md").write_text(text, encoding="utf-8")
    (ROUND / "R35_E3_BOUND_COVERAGE_REPORT.md").write_text(text, encoding="utf-8")
    print(f"[77] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
