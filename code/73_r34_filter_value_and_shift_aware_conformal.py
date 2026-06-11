"""R34 upgrades U1-U3: filter decision value, shift-aware conformal, vacuous-pass guard.

All computations reuse cached prediction/decision exports; no model retraining.

U2  Filter value/regret: across the threshold x cost surface, compare three
    selection policies -- (a) lowest-RMSE model, (b) filter-constrained
    minimum-loss model (beta_FS,cons >= 2.5), (c) oracle minimum-loss model --
    on realized event-equal expected loss and false-safe exposure.

U1  Shift-aware conformal baseline at N=2000 (event-disjoint, level 0.90):
    standard split conformal vs event-weighted calibration quantile vs
    Mondrian (event-median-PGA tercile) calibration. Reports event-equal
    coverage, false-safe probability at 1% IDR, conservative index and
    false-unsafe burden. Answers "why not existing conformal-under-shift?".

U3  Vacuous-pass guard: re-evaluate the label-budget eligibility map with a
    minimum-discrimination condition P_FU <= 0.30 added to the filter.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[1]
HT = ROOT / "outputs" / "high_target"
OUT = HT / "r34_filter_value"
ROUND = ROOT / "rounds" / "R34_figure_style_normalization_2026-06-10"

RISK_DETAIL = HT / "r22_decision_risk" / "decision_risk_sensitivity_detail.csv"
RELIABILITY = HT / "r24_structural_safety_tables" / "false_safe_reliability_detail.csv"
CLUSTER = HT / "r21_full_residual_cluster" / "full_event_cluster_summary_n500.csv"
PREDICTIONS = HT / "r28_event_disjoint_large_budget" / "event_disjoint_large_budget_predictions.csv"
RECORDS = HT / "recorded_nsmp_full" / "nsmp_recorded_records.csv"
BUDGET_SUMMARY = HT / "r28_gate_large_budget_sensitivity" / "true_budget_gate_summary.csv"

BETA_TARGET = 2.5
PFU_MAX = 0.30
LEVEL = 0.90
THRESHOLD_IDR = 0.01
LOG_THR = math.log10(THRESHOLD_IDR)
MODELS = ["ridge_direct", "lgbm_direct", "xgb_direct", "scratch_mlp"]
NORMAL = NormalDist()
BOOT_N = 2000


def beta_from_p(p: float) -> float:
    p = min(max(float(p), 1e-9), 1 - 1e-9)
    return float(-NORMAL.inv_cdf(p))


# ------------------------------------------------------------------ U2 ------
def filter_value_regret() -> pd.DataFrame:
    detail = pd.read_csv(RISK_DETAIL)
    rel = pd.read_csv(RELIABILITY)
    cluster = pd.read_csv(CLUSTER)

    proto_map = {"main_full_residual": "main event-held-out",
                 "event_disjoint_full_residual": "event-disjoint target"}
    cluster["protocol_label"] = cluster["protocol"].map(proto_map)
    rmse_best = (cluster.sort_values("rmse_event_mean")
                 .groupby("protocol_label").head(1)
                 .set_index("protocol_label")["model"].to_dict())

    gate = rel.set_index(["protocol", "model", "threshold_idr"])[
        "beta_false_safe_cons"].to_dict()

    rows = []
    for (protocol, thr, cost), g in detail.groupby(
            ["protocol", "threshold_idr", "cost_ratio_false_safe"]):
        g = g.set_index("model")
        oracle = g.loc[g["expected_loss"].idxmin()]
        rmse_model = rmse_best[protocol]
        rmse_row = g.loc[rmse_model]
        eligible = [m for m in g.index
                    if gate.get((protocol, m, thr), -np.inf) >= BETA_TARGET]
        if eligible:
            ge = g.loc[eligible]
            filt = ge.loc[ge["expected_loss"].idxmin()]
            filt_model, filt_loss = filt.name, float(filt["expected_loss"])
            filt_fs = float(filt["false_safe_rate"])
            filt_fu = float(filt["false_unsafe_rate"])
        else:
            filt_model, filt_loss, filt_fs, filt_fu = "none", np.nan, np.nan, np.nan
        rows.append({
            "protocol": protocol,
            "threshold_idr": thr,
            "cost_ratio": cost,
            "oracle_model": oracle.name,
            "oracle_loss": float(oracle["expected_loss"]),
            "rmse_model": rmse_model,
            "rmse_loss": float(rmse_row["expected_loss"]),
            "rmse_false_safe": float(rmse_row["false_safe_rate"]),
            "filter_model": filt_model,
            "filter_loss": filt_loss,
            "filter_false_safe": filt_fs,
            "filter_false_unsafe": filt_fu,
            "regret_rmse": float(rmse_row["expected_loss"] - oracle["expected_loss"]),
            "regret_filter": (filt_loss - float(oracle["expected_loss"])
                              if filt_model != "none" else np.nan),
        })
    cells = pd.DataFrame(rows)
    cells.to_csv(OUT / "filter_value_regret_cells.csv", index=False)
    return cells


def summarize_u2(cells: pd.DataFrame) -> list[str]:
    lines = ["## U2 Filter decision value (threshold x cost surface, N=500 trace)",
             "",
             "Policies: RMSE-best model; filter-constrained minimum-loss model "
             f"(beta_FS,cons >= {BETA_TARGET}); oracle minimum-loss model. "
             "Losses are event-equal expected screening losses.",
             ""]
    lines.append("| protocol | cells | infeasible | mean regret RMSE-pick | mean regret filter-pick | worst regret RMSE-pick | worst regret filter-pick | worst-cell loss RMSE | worst-cell loss filter | max P_FS RMSE-pick | max P_FS filter-pick |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for protocol, g in cells.groupby("protocol"):
        feasible = g[g["filter_model"] != "none"]
        lines.append(
            f"| {protocol} | {len(g)} | {int((g['filter_model'] == 'none').sum())} | "
            f"{g['regret_rmse'].mean():.4f} | {feasible['regret_filter'].mean():.4f} | "
            f"{g['regret_rmse'].max():.4f} | {feasible['regret_filter'].max():.4f} | "
            f"{g['rmse_loss'].max():.3f} | {feasible['filter_loss'].max():.3f} | "
            f"{g['rmse_false_safe'].max():.4f} | {feasible['filter_false_safe'].max():.4f} |"
        )
    lines.append("")
    hi_cost = cells[cells["cost_ratio"] >= 25]
    for protocol, g in hi_cost.groupby("protocol"):
        feasible = g[g["filter_model"] != "none"]
        lines.append(
            f"- High-cost cells (C >= 25), {protocol}: RMSE-pick mean regret "
            f"{g['regret_rmse'].mean():.4f} vs filter-pick {feasible['regret_filter'].mean():.4f}; "
            f"worst-case loss {g['rmse_loss'].max():.3f} vs {feasible['filter_loss'].max():.3f}."
        )
    lines.append("")
    return lines


# ------------------------------------------------------------------ U1 ------
def order_stat_quantile(values: np.ndarray, level: float) -> float:
    vals = np.sort(np.asarray(values, float))
    n = len(vals)
    k = min(max(int(math.ceil(level * (n + 1))), 1), n)
    return float(vals[k - 1])


def weighted_quantile(values: np.ndarray, weights: np.ndarray, level: float) -> float:
    """Conservative weighted quantile with effective-sample finite correction."""
    order = np.argsort(values)
    v, w = np.asarray(values, float)[order], np.asarray(weights, float)[order]
    w = w / w.sum()
    n_eff = 1.0 / np.sum(w ** 2)
    level_adj = min(level * (n_eff + 1.0) / n_eff, 1.0)
    cum = np.cumsum(w)
    idx = int(np.searchsorted(cum, level_adj, side="left"))
    return float(v[min(idx, len(v) - 1)])


def event_equal_mean(flags: np.ndarray, events: np.ndarray) -> float:
    df = pd.DataFrame({"f": flags.astype(float), "e": events})
    return float(df.groupby("e")["f"].mean().mean())


def event_boot_upper(flags: np.ndarray, events: np.ndarray,
                     rng: np.random.Generator) -> float:
    means = (pd.DataFrame({"f": flags.astype(float), "e": events})
             .groupby("e")["f"].mean().to_numpy())
    draws = rng.choice(means, size=(BOOT_N, len(means)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.95))


def load_n2000() -> pd.DataFrame:
    usecols = ["rep", "N", "model", "split", "gm_id",
               "y_true_log", "y_pred_log", "residual_abs_log"]
    chunks = []
    for chunk in pd.read_csv(PREDICTIONS, usecols=usecols, chunksize=2_000_000):
        sel = chunk[(chunk["N"] == 2000) & chunk["model"].isin(MODELS)]
        if len(sel):
            chunks.append(sel)
    data = pd.concat(chunks, ignore_index=True)
    rec = pd.read_csv(RECORDS, usecols=["gm_id", "event_id", "pga_g"])
    rec = rec.drop_duplicates("gm_id")
    return data.merge(rec, on="gm_id", how="left")


def shift_aware_conformal() -> pd.DataFrame:
    data = load_n2000()
    rng_base = np.random.default_rng(20260610)
    rows = []
    for (model, rep), g in data.groupby(["model", "rep"]):
        cal = g[g["split"].eq("calibration")]
        test = g[g["split"].eq("test")].copy()
        if cal.empty or test.empty:
            continue
        cal_resid = cal["residual_abs_log"].to_numpy(float)

        # Method 1: standard split conformal.
        q_std = order_stat_quantile(cal_resid, LEVEL)

        # Method 2: event-weighted calibration (each event equal total weight).
        ev_counts = cal.groupby("event_id")["residual_abs_log"].transform("size")
        weights = 1.0 / ev_counts.to_numpy(float)
        q_evw = weighted_quantile(cal_resid, weights, LEVEL)

        # Method 3: Mondrian by event-median PGA tercile (bins from calibration).
        ev_pga_cal = cal.groupby("event_id")["pga_g"].median()
        edges = np.quantile(ev_pga_cal.to_numpy(float), [1 / 3, 2 / 3])
        cal_bins = np.digitize(cal["event_id"].map(ev_pga_cal).to_numpy(float), edges)
        q_bin = {}
        for b in (0, 1, 2):
            vals = cal_resid[cal_bins == b]
            q_bin[b] = (order_stat_quantile(vals, LEVEL)
                        if len(vals) >= 5 else q_std)
        ev_pga_test = test.groupby("event_id")["pga_g"].median()
        test_bins = np.digitize(test["event_id"].map(ev_pga_test).to_numpy(float), edges)
        q_mondrian = np.array([q_bin[b] for b in test_bins])

        y_true = test["y_true_log"].to_numpy(float)
        y_pred = test["y_pred_log"].to_numpy(float)
        abs_res = test["residual_abs_log"].to_numpy(float)
        events = test["event_id"].to_numpy()
        truth_unsafe = y_true > LOG_THR

        for method, q in [("standard split", q_std),
                          ("event-weighted", q_evw),
                          ("Mondrian PGA-tercile", q_mondrian)]:
            qv = np.broadcast_to(np.asarray(q, float), y_pred.shape)
            covered = abs_res <= qv
            predicted_safe = (y_pred + qv) <= LOG_THR
            false_safe = truth_unsafe & predicted_safe
            false_unsafe = (~truth_unsafe) & (~predicted_safe)
            rng = np.random.default_rng(int(rng_base.integers(0, 2 ** 31 - 1)))
            pfs = event_equal_mean(false_safe, events)
            pfs_hi = event_boot_upper(false_safe, events, rng)
            rows.append({
                "model": model, "rep": int(rep), "method": method,
                "q_mean_log": float(np.mean(qv)),
                "coverage_event_equal": event_equal_mean(covered, events),
                "p_false_safe": pfs,
                "beta_false_safe_cons": beta_from_p(pfs_hi),
                "p_false_unsafe": event_equal_mean(false_unsafe, events),
            })
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "shift_aware_conformal_detail.csv", index=False)
    summary = (detail.groupby(["model", "method"])
               .agg(coverage_median=("coverage_event_equal", "median"),
                    p_fs_median=("p_false_safe", "median"),
                    beta_cons_median=("beta_false_safe_cons", "median"),
                    p_fu_median=("p_false_unsafe", "median"),
                    q_median=("q_mean_log", "median"),
                    reps=("rep", "nunique"))
               .reset_index())
    summary.to_csv(OUT / "shift_aware_conformal_summary.csv", index=False)
    return summary


def summarize_u1(summary: pd.DataFrame) -> list[str]:
    label = {"ridge_direct": "Ridge direct", "lgbm_direct": "LGBM direct",
             "xgb_direct": "XGB direct", "scratch_mlp": "MLP scratch"}
    method_order = ["standard split", "event-weighted", "Mondrian PGA-tercile"]
    lines = ["## U1 Shift-aware conformal baseline "
             "(event-disjoint, N=2000, level 0.90, medians over 10 reps)",
             "",
             "| model | method | event-equal coverage | P_FS at 1% IDR | "
             "beta_FS,cons | P_FU | median q (log10) |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for m in MODELS:
        for method in method_order:
            r = summary[(summary["model"].eq(m)) & (summary["method"].eq(method))]
            if r.empty:
                continue
            r = r.iloc[0]
            lines.append(
                f"| {label[m]} | {method} | {r['coverage_median']:.3f} | "
                f"{r['p_fs_median']:.4f} | {r['beta_cons_median']:.2f} | "
                f"{r['p_fu_median']:.3f} | {r['q_median']:.3f} |"
            )
    lines.append("")
    return lines


# ------------------------------------------------------------------ U3 ------
def vacuous_pass_guard() -> list[str]:
    s = pd.read_csv(BUDGET_SUMMARY)
    s = s[s["interval_level"].eq(0.9)].copy()
    s["plain_gate"] = np.where(s["beta_false_safe_cons_median"] >= BETA_TARGET,
                               "Pass", "Fail")
    s["guarded_gate"] = np.where(
        s["plain_gate"].eq("Fail"), "Fail",
        np.where(s["p_false_unsafe_median"] <= PFU_MAX, "Pass", "Vacuous pass"))
    flips = s[s["guarded_gate"].eq("Vacuous pass")]
    lines = [f"## U3 Vacuous-pass guard (add P_FU <= {PFU_MAX:.2f} to the filter)",
             "",
             "| model | N | beta_FS,cons | P_FU | plain filter | guarded filter |",
             "| --- | ---: | ---: | ---: | --- | --- |"]
    for _, r in s.sort_values(["model_label", "target_label_budget"]).iterrows():
        lines.append(
            f"| {r['model_label']} | {int(r['target_label_budget'])} | "
            f"{r['beta_false_safe_cons_median']:.2f} | "
            f"{r['p_false_unsafe_median']:.3f} | {r['plain_gate']} | "
            f"{r['guarded_gate']} |")
    lines.append("")
    if len(flips):
        worst = flips.sort_values("p_false_unsafe_median", ascending=False).iloc[0]
        lines.append(
            f"- {len(flips)} pass cell(s) become vacuous under the guard; the most "
            f"extreme is {worst['model_label']} at N={int(worst['target_label_budget'])} "
            f"with P_FU = {worst['p_false_unsafe_median']:.3f} (flags nearly every "
            "case as unsafe, so the false-safe pass carries no screening value).")
    lines.append("")
    return lines


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    parts = ["# R34 upgrade results: filter value, shift-aware conformal, vacuous-pass guard",
             "", "Date: 2026-06-10. All results derive from cached prediction/decision "
             "exports; no model retraining.", ""]
    cells = filter_value_regret()
    parts += summarize_u2(cells)
    print("[73] U2 done", flush=True)
    summary = shift_aware_conformal()
    parts += summarize_u1(summary)
    print("[73] U1 done", flush=True)
    parts += vacuous_pass_guard()
    print("[73] U3 done", flush=True)
    text = "\n".join(parts) + "\n"
    (OUT / "R34_UPGRADE_RESULTS.md").write_text(text, encoding="utf-8")
    (ROUND / "R34_UPGRADE_RESULTS.md").write_text(text, encoding="utf-8")
    print(f"[73] wrote {ROUND / 'R34_UPGRADE_RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()
