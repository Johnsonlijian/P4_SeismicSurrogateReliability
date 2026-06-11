"""R35 pre-manuscript computations from cached exports (no retraining).

A. Event metadata table (magnitude, components, PGA range, protocol role).
B. Conditional miss rate P(predicted safe | truly unsafe), dual to P_FS,
   at all thresholds for both protocols (N=500 trace), event-equal and
   event-conditional variants, plus raw unsafe-row/event counts.
C. Budget-wise conditional miss at 1% IDR from the true event-disjoint
   N=50..2000 exports (median over reps).
D. Conformal risk control (CRC) baseline at N=2000: calibrate the interval
   inflation on the calibration set to certify row-level false-safe risk
   <= alpha* = Phi(-2.5), then measure realized test risk under event-
   disjoint shift. Shows whether an exchangeability-based guarantee
   survives event-level separation (answers the RCPS/CRC novelty attack).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HT = ROOT / "outputs" / "high_target"
OUT = HT / "r35_conditional_miss"
ROUND = ROOT / "rounds" / "R34_figure_style_normalization_2026-06-10"

RECORDS = HT / "recorded_nsmp_full" / "nsmp_recorded_records.csv"
SPLIT_META = HT / "split_manifest" / "recorded_event_split_meta.json"
EVENT_META = HT / "event_disjoint_conformal_stress" / "event_disjoint_meta.json"
RISK_DETAIL = HT / "r22_decision_risk" / "decision_risk_sensitivity_detail.csv"
MAIN_TRACE = HT / "r21_full_residual_trace" / "residual_trace_samples.csv"
DISJ_TRACE = HT / "event_disjoint_conformal_stress" / "event_disjoint_residual_samples.csv"
PREDICTIONS = HT / "r28_event_disjoint_large_budget" / "event_disjoint_large_budget_predictions.csv"

NORMAL = NormalDist()
ALPHA_STAR = NORMAL.cdf(-2.5)          # 0.0062, the row-level risk target
THRESHOLDS = [0.005, 0.0075, 0.010, 0.015, 0.020, 0.030, 0.040]
MODELS = ["ridge_direct", "lgbm_direct", "xgb_direct", "scratch_mlp"]
LOG1PCT = math.log10(0.01)


def read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# ------------------------------------------------------------------- A ------
def event_metadata() -> pd.DataFrame:
    rec = pd.read_csv(RECORDS, usecols=["gm_id", "event_id", "event_title",
                                        "magnitude", "pga_g"])
    split = read_json(SPLIT_META)
    dis = read_json(EVENT_META)
    target_events = set(map(str, split.get("target_events", [])))
    fit_events = set(map(str, dis.get("target_fit_events", [])))
    test_events = set(map(str, dis.get("target_test_events", [])))

    g = rec.groupby("event_id").agg(
        event_title=("event_title", "first"),
        magnitude=("magnitude", "max"),
        n_components=("gm_id", "nunique"),
        pga_min_g=("pga_g", "min"),
        pga_median_g=("pga_g", "median"),
        pga_max_g=("pga_g", "max"),
    ).reset_index()
    def role(e: str) -> str:
        e = str(e)
        if e in test_events:
            return "target-test (disjoint)"
        if e in fit_events:
            return "target-fit/cal (disjoint)"
        if e in target_events:
            return "target (holdout)"
        return "source"
    g["protocol_role"] = g["event_id"].map(role)
    g = g.sort_values(["protocol_role", "magnitude"], ascending=[True, False])
    g.to_csv(OUT / "event_metadata_table.csv", index=False)
    return g


# ------------------------------------------------------------------- B ------
def conditional_miss_n500() -> pd.DataFrame:
    detail = pd.read_csv(RISK_DETAIL)
    base = detail[detail["cost_ratio_false_safe"].eq(1)].copy()
    base["conditional_miss_eventequal"] = (
        base["false_safe_rate"] / base["true_exceed_rate"]).replace([np.inf], np.nan)

    # Event-conditional variant from raw traces: per event with >=1 unsafe row,
    # miss fraction among unsafe rows; mean over such events.
    usecols = ["family", "model", "split", "N", "gm_id",
               "y_true_log", "y_pred_log", "q_value_log"]
    rec = pd.read_csv(RECORDS, usecols=["gm_id", "event_id"]).drop_duplicates("gm_id")
    rows = []
    for protocol, path in [("main event-held-out", MAIN_TRACE),
                           ("event-disjoint target", DISJ_TRACE)]:
        df = pd.read_csv(path, usecols=usecols)
        df = df[(df["N"] == 500) & df["split"].eq("test")].merge(rec, on="gm_id")
        df["upper"] = df["y_pred_log"] + df["q_value_log"]
        for model, g in df.groupby("model"):
            for thr in THRESHOLDS:
                log_thr = math.log10(thr)
                unsafe = g[g["y_true_log"] > log_thr]
                n_unsafe = len(unsafe)
                ev_with_unsafe = unsafe["event_id"].nunique()
                if n_unsafe:
                    miss = unsafe["upper"] <= log_thr
                    ev_cond = (unsafe.assign(miss=miss)
                               .groupby("event_id")["miss"].mean().mean())
                    pooled = float(miss.mean())
                else:
                    ev_cond, pooled = np.nan, np.nan
                rows.append({
                    "protocol": protocol, "model": model,
                    "threshold_idr": thr,
                    "n_unsafe_rows": int(n_unsafe),
                    "n_events_with_unsafe": int(ev_with_unsafe),
                    "conditional_miss_pooled": pooled,
                    "conditional_miss_event_mean": float(ev_cond) if n_unsafe else np.nan,
                })
    raw = pd.DataFrame(rows)
    merged = base.merge(raw, on=["protocol", "model", "threshold_idr"], how="left")
    keep = ["protocol", "model", "model_label", "threshold_idr",
            "true_exceed_rate", "false_safe_rate", "conditional_miss_eventequal",
            "conditional_miss_pooled", "conditional_miss_event_mean",
            "n_unsafe_rows", "n_events_with_unsafe"]
    merged = merged[keep].sort_values(["protocol", "threshold_idr", "model"])
    merged.to_csv(OUT / "conditional_miss_n500.csv", index=False)
    return merged


# ------------------------------------------------------------------- C/D ----
def load_n2000_and_budgets() -> pd.DataFrame:
    usecols = ["rep", "N", "model", "split", "gm_id",
               "y_true_log", "y_pred_log", "residual_abs_log", "q_value_log"]
    chunks = []
    for chunk in pd.read_csv(PREDICTIONS, usecols=usecols, chunksize=2_000_000):
        sel = chunk[chunk["model"].isin(MODELS)]
        if len(sel):
            chunks.append(sel)
    data = pd.concat(chunks, ignore_index=True)
    rec = pd.read_csv(RECORDS, usecols=["gm_id", "event_id"]).drop_duplicates("gm_id")
    return data.merge(rec, on="gm_id", how="left")


def budget_conditional_and_crc(data: pd.DataFrame) -> None:
    # C: conditional miss by budget at 1% IDR.
    rows = []
    test = data[data["split"].eq("test")]
    for (model, n, rep), g in test.groupby(["model", "N", "rep"]):
        q = float(g["q_value_log"].iloc[0])
        unsafe = g[g["y_true_log"] > LOG1PCT]
        if len(unsafe) == 0:
            continue
        miss = (unsafe["y_pred_log"] + q) <= LOG1PCT
        ev_cond = (unsafe.assign(m=miss).groupby("event_id")["m"].mean().mean())
        rows.append({"model": model, "N": int(n), "rep": int(rep),
                     "conditional_miss_event_mean": float(ev_cond),
                     "n_unsafe_rows": int(len(unsafe))})
    cond = pd.DataFrame(rows)
    summary = (cond.groupby(["model", "N"])
               .agg(conditional_miss_median=("conditional_miss_event_mean", "median"),
                    conditional_miss_p05=("conditional_miss_event_mean",
                                          lambda s: float(np.quantile(s, 0.05))),
                    conditional_miss_p95=("conditional_miss_event_mean",
                                          lambda s: float(np.quantile(s, 0.95))),
                    unsafe_rows_median=("n_unsafe_rows", "median"))
               .reset_index())
    summary.to_csv(OUT / "conditional_miss_by_budget.csv", index=False)

    # D: CRC baseline at N=2000 — calibrate lambda for row-level false-safe
    # risk <= ALPHA_STAR with the (n+1) finite-sample correction, then measure
    # realized event-disjoint test risk.
    crc_rows = []
    d2k = data[data["N"].eq(2000)]
    for (model, rep), g in d2k.groupby(["model", "rep"]):
        cal = g[g["split"].eq("calibration")]
        tst = g[g["split"].eq("test")]
        if cal.empty or tst.empty:
            continue
        n = len(cal)
        cal_unsafe = cal["y_true_log"].to_numpy() > LOG1PCT
        margins = (LOG1PCT - cal["y_pred_log"].to_numpy())
        # False-safe occurs iff lambda <= margin and the case is unsafe.
        unsafe_margins = np.sort(margins[cal_unsafe])
        # Risk(lambda) = (#unsafe with margin >= lambda)/n ; choose smallest
        # lambda with (n*Risk + 1)/(n+1) <= alpha.
        lam = None
        candidates = np.unique(np.concatenate([[0.0], unsafe_margins,
                                               [unsafe_margins.max() + 1e-6]
                                               if len(unsafe_margins) else [[0.0], [1.0]][1]]))
        for c in candidates:
            risk = float(np.sum(unsafe_margins >= c)) / n
            if (n * risk + 1.0) / (n + 1.0) <= ALPHA_STAR:
                lam = float(c)
                break
        if lam is None:
            lam = float(unsafe_margins.max() + 1e-6) if len(unsafe_margins) else 0.0
        y_t = tst["y_true_log"].to_numpy()
        up_t = tst["y_pred_log"].to_numpy() + lam
        t_unsafe = y_t > LOG1PCT
        fs = t_unsafe & (up_t <= LOG1PCT)
        fu = (~t_unsafe) & (up_t > LOG1PCT)
        ev = tst["event_id"].to_numpy()
        fs_evmean = pd.DataFrame({"f": fs, "e": ev}).groupby("e")["f"].mean().mean()
        crc_rows.append({
            "model": model, "rep": int(rep), "lambda_log10": lam,
            "alpha_target_rowlevel": ALPHA_STAR,
            "test_false_safe_rowlevel": float(fs.mean()),
            "test_false_safe_eventequal": float(fs_evmean),
            "test_false_unsafe_rowlevel": float(fu.mean()),
            "cal_unsafe_rows": int(cal_unsafe.sum()),
        })
    crc = pd.DataFrame(crc_rows)
    crc.to_csv(OUT / "crc_baseline_detail.csv", index=False)
    crc_summary = (crc.groupby("model")
                   .agg(lambda_median=("lambda_log10", "median"),
                        test_fs_rowlevel_median=("test_false_safe_rowlevel", "median"),
                        test_fs_eventequal_median=("test_false_safe_eventequal", "median"),
                        test_fu_rowlevel_median=("test_false_unsafe_rowlevel", "median"),
                        violation_fraction=("test_false_safe_rowlevel",
                                            lambda s: float((s > ALPHA_STAR).mean())),
                        reps=("rep", "nunique"))
                   .reset_index())
    crc_summary.to_csv(OUT / "crc_baseline_summary.csv", index=False)

    # Markdown report.
    lines = ["# R35 conditional miss, CRC baseline, event metadata", "",
             f"Row-level CRC target alpha* = Phi(-2.5) = {ALPHA_STAR:.4f} "
             "(false-safe risk at 1% IDR), calibrated with the (n+1) "
             "finite-sample correction on the event-disjoint calibration set "
             "(N=2000), evaluated on held-out disjoint test events.", "",
             "## CRC baseline (medians over 10 reps)", "",
             "| model | lambda (log10) | test FS row-level | test FS event-equal | test FU row-level | reps violating alpha* |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for _, r in crc_summary.iterrows():
        lines.append(f"| {r['model']} | {r['lambda_median']:.3f} | "
                     f"{r['test_fs_rowlevel_median']:.4f} | "
                     f"{r['test_fs_eventequal_median']:.4f} | "
                     f"{r['test_fu_rowlevel_median']:.3f} | "
                     f"{100*r['violation_fraction']:.0f}% |")
    lines += ["", "## Conditional miss by budget (1% IDR, event-disjoint, median over reps)",
              "", "| model | N | conditional miss (median) | p05-p95 |",
              "| --- | ---: | ---: | --- |"]
    for _, r in summary.iterrows():
        lines.append(f"| {r['model']} | {int(r['N'])} | "
                     f"{r['conditional_miss_median']:.3f} | "
                     f"[{r['conditional_miss_p05']:.3f}, {r['conditional_miss_p95']:.3f}] |")
    (OUT / "R35_PRECOMPUTE_REPORT.md").write_text("\n".join(lines) + "\n",
                                                  encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = event_metadata()
    print(f"[74] events: {len(meta)}", flush=True)
    cond = conditional_miss_n500()
    print(f"[74] N500 conditional rows: {len(cond)}", flush=True)
    data = load_n2000_and_budgets()
    budget_conditional_and_crc(data)
    print(f"[74] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
