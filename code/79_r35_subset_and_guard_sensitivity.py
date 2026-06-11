"""R35-E5/E6: story-subset sensitivity and guard-threshold sensitivity.

E5: the protocol population is uniformly T1 = 1.8 s; the 3-story half of the
grid at that period is physically implausible (parametric stress cells).
Recompute the 1% IDR filter quantities separately for 3-story and 6-story
rows from the cached event-disjoint exports (no retraining).

E6: guarded-filter verdicts at P_FU,max in {0.15, 0.30, 0.50} from the cached
budget summary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import r35_harness as H

OUT = H.ROOT / "outputs" / "high_target" / "r35_subset_guard"
ROUND = H.ROOT / "rounds" / "R34_figure_style_normalization_2026-06-10"
PRED = H.ROOT / "outputs" / "high_target" / "r28_event_disjoint_large_budget" / "event_disjoint_large_budget_predictions.csv"
BUDGET_SUMMARY = H.ROOT / "outputs" / "high_target" / "r28_gate_large_budget_sensitivity" / "true_budget_gate_summary.csv"

MODELS = ["ridge_direct", "lgbm_direct", "xgb_direct", "scratch_mlp"]
BETA_TARGET = 2.5


def e5_story_subsets() -> pd.DataFrame:
    usecols = ["rep", "N", "model", "split", "gm_id", "n_story",
               "y_true_log", "y_pred_log", "q_value_log"]
    chunks = []
    for chunk in pd.read_csv(PRED, usecols=usecols, chunksize=2_000_000):
        sel = chunk[(chunk["split"].eq("test")) & chunk["model"].isin(MODELS)
                    & chunk["N"].isin([500, 2000])]
        if len(sel):
            chunks.append(sel)
    data = pd.concat(chunks, ignore_index=True)
    rec = H.load_records()[["gm_id", "event_id"]].drop_duplicates("gm_id")
    data = data.merge(rec, on="gm_id", how="left")

    rows = []
    rng = np.random.default_rng(20260616)
    for (model, n, rep), g in data.groupby(["model", "N", "rep"]):
        q = float(g["q_value_log"].iloc[0])
        for subset, gg in [("all", g), ("3-story", g[g["n_story"].eq(3)]),
                           ("6-story", g[g["n_story"].eq(6)])]:
            met = H.decision_metrics(gg["y_true_log"].to_numpy(float),
                                     gg["y_pred_log"].to_numpy(float), q,
                                     gg["event_id"].to_numpy(), rng)
            met.update({"model": model, "N": int(n), "rep": int(rep),
                        "subset": subset})
            rows.append(met)
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "story_subset_detail.csv", index=False)
    summary = (detail.groupby(["model", "N", "subset"])
               .agg(beta_median=("beta_fs_cons", "median"),
                    pfu_median=("p_false_unsafe", "median"),
                    cond_median=("conditional_miss_event_mean", "median"),
                    exceed_rows=("n_unsafe_rows", "median"))
               .reset_index())
    summary["filter"] = np.where(summary["beta_median"] >= BETA_TARGET,
                                 "Pass", "Fail")
    summary.to_csv(OUT / "story_subset_summary.csv", index=False)
    return summary


def e6_guard_sensitivity() -> pd.DataFrame:
    s = pd.read_csv(BUDGET_SUMMARY)
    s = s[s["interval_level"].eq(0.9)].copy()
    rows = []
    for pfu_max in [0.15, 0.30, 0.50]:
        for _, r in s.iterrows():
            plain = r["beta_false_safe_cons_median"] >= BETA_TARGET
            guarded = plain and (r["p_false_unsafe_median"] <= pfu_max)
            rows.append({"model_label": r["model_label"],
                         "N": int(r["target_label_budget"]),
                         "pfu_max": pfu_max,
                         "plain": "Pass" if plain else "Fail",
                         "guarded": ("Pass" if guarded else
                                     ("Vacuous/guard-fail" if plain else "Fail"))})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "guard_sensitivity.csv", index=False)
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = e5_story_subsets()
    guard = e6_guard_sensitivity()

    lines = ["# R35-E5/E6 story-subset and guard sensitivity", "",
             "## E5: 1% IDR filter quantities by story subset "
             "(event-disjoint, medians over 10 reps)", "",
             "| model | N | subset | beta_FS,cons | P_FU | cond. miss | unsafe rows | filter |",
             "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |"]
    for _, r in summary.sort_values(["model", "N", "subset"]).iterrows():
        lines.append(f"| {r['model']} | {int(r['N'])} | {r['subset']} | "
                     f"{r['beta_median']:.2f} | {r['pfu_median']:.3f} | "
                     f"{r['cond_median']:.3f} | {int(r['exceed_rows'])} | "
                     f"{r['filter']} |")
    lines += ["", "## E6: guarded verdicts vs P_FU,max", "",
              "Differences relative to P_FU,max = 0.30:", ""]
    base = guard[guard["pfu_max"].eq(0.30)].set_index(["model_label", "N"])["guarded"]
    for pfu_max in [0.15, 0.50]:
        cmp_ = guard[guard["pfu_max"].eq(pfu_max)].set_index(["model_label", "N"])["guarded"]
        diff = cmp_[cmp_ != base]
        if diff.empty:
            lines.append(f"- P_FU,max = {pfu_max}: identical verdict set.")
        else:
            for (ml, n), v in diff.items():
                lines.append(f"- P_FU,max = {pfu_max}: {ml} N={n}: "
                             f"{base.loc[(ml, n)]} -> {v}")
    text = "\n".join(lines) + "\n"
    (OUT / "R35_E5_E6_REPORT.md").write_text(text, encoding="utf-8")
    (ROUND / "R35_E5_E6_REPORT.md").write_text(text, encoding="utf-8")
    print(f"[79] done -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
