"""R24 Structural Safety tables and false-safe reliability index.

This script turns the residual-trace outputs into manuscript-ready evidence:

- protocol/dataset summary,
- model settings,
- event-clustered metric summary,
- decision-risk and false-safe reliability-index summary,
- an additional multi-panel figure for false-safe reliability.

No raw NSMP records are redistributed. Outputs are derived tables and figures.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "high_target" / "r24_structural_safety_tables"
FIG_DIR = ROOT / "outputs" / "figures" / "high_target"
DOCS = ROOT / "docs"
LATEX = ROOT / "submission" / "structural_safety_2026-06-01" / "latex_source_flat"

MAIN_TRACE = ROOT / "outputs" / "high_target" / "r21_full_residual_trace" / "residual_trace_samples.csv"
EVENT_TRACE = ROOT / "outputs" / "high_target" / "event_disjoint_conformal_stress" / "event_disjoint_residual_samples.csv"
RECORDS = ROOT / "outputs" / "high_target" / "recorded_nsmp_full" / "nsmp_recorded_records.csv"
R21_SUMMARY = ROOT / "outputs" / "high_target" / "r21_full_residual_cluster" / "full_event_cluster_summary_n500.csv"
R23_METRICS = ROOT / "outputs" / "high_target" / "r23_residual_mechanism" / "residual_mechanism_metrics.csv"
R22_DETAIL = ROOT / "outputs" / "high_target" / "r22_decision_risk" / "decision_risk_sensitivity_detail.csv"
R22_WINNERS = ROOT / "outputs" / "high_target" / "r22_decision_risk" / "decision_risk_sensitivity_winners.csv"

POP_META = ROOT / "outputs" / "high_target" / "recorded_nsmp_full" / "nsmp_recorded_population_meta.json"
MDOF_META = ROOT / "outputs" / "high_target" / "nonlinear_mdof_grid_full" / "nonlinear_mdof_meta.json"
SPLIT_META = ROOT / "outputs" / "high_target" / "split_manifest" / "recorded_event_split_meta.json"
EVENT_META = ROOT / "outputs" / "high_target" / "event_disjoint_conformal_stress" / "event_disjoint_meta.json"

N_TARGET = 500
THRESHOLDS = np.array([0.005, 0.0075, 0.010, 0.015, 0.020, 0.030, 0.040])
COSTS = [1, 10, 50, 100]
NORMAL = NormalDist()
BOOTSTRAP_REPS = 3000
BOOTSTRAP_SEED = 20260601
GATE_BETA_TARGET = 2.5

MODEL_LABELS = {
    "xgb_direct": "XGB direct",
    "lgbm_direct": "LGBM direct",
    "hgb_direct": "HGB direct",
    "rf_direct": "RF direct",
    "ridge_direct": "Ridge direct",
    "pretrained_finetune": "MLP finetune",
    "scratch_mlp": "MLP scratch",
}

MODEL_SHORT = {
    "xgb_direct": "XGB",
    "lgbm_direct": "LGBM",
    "hgb_direct": "HGB",
    "rf_direct": "RF",
    "ridge_direct": "Ridge",
    "pretrained_finetune": "FT",
    "scratch_mlp": "Scratch",
}

PALETTE = {
    "xgb_direct": "#2F5F8A",
    "lgbm_direct": "#F58518",
    "hgb_direct": "#54A24B",
    "rf_direct": "#9E77B4",
    "ridge_direct": "#6B6B6B",
    "pretrained_finetune": "#4C78A8",
    "scratch_mlp": "#D62728",
}

PROTOCOL_MAP = {
    "main_full_residual": "main event-held-out",
    "event_disjoint_full_residual": "event-disjoint target",
}


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.0,
        "axes.titlesize": 8.2,
        "axes.labelsize": 7.2,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 6.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def beta_false_safe(p: float) -> float:
    """Map empirical false-safe probability to a reliability-style index."""
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return float(-NORMAL.inv_cdf(p))


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = BOOTSTRAP_REPS) -> tuple[float, float]:
    """Event-bootstrap interval for an event-equal mean."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    if vals.size == 1:
        return float(vals[0]), float(vals[0])
    draws = rng.integers(0, vals.size, size=(n_boot, vals.size))
    means = vals[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def tex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def write_latex_table(path: Path, caption: str, label: str, columns: list[str], rows: list[list[object]], widths: str) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        rf"\caption{{{tex_escape(caption)}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{widths}}}",
        r"\toprule",
        " & ".join(tex_escape(c) for c in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(tex_escape(x) for x in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_traces() -> pd.DataFrame:
    frames = []
    for protocol, path in [
        ("main event-held-out", MAIN_TRACE),
        ("event-disjoint target", EVENT_TRACE),
    ]:
        df = pd.read_csv(path)
        df["protocol"] = protocol
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data = data[data["N"].eq(N_TARGET)].copy()
    records = pd.read_csv(RECORDS)[["gm_id", "event_id", "event_title", "magnitude"]].drop_duplicates("gm_id")
    data = data.merge(records, on="gm_id", how="left")
    data["upper_log"] = data["y_pred_log"].astype(float) + data["q_value_log"].astype(float)
    return data


def build_table1(traces: pd.DataFrame) -> pd.DataFrame:
    pop = read_json(POP_META)
    mdof = read_json(MDOF_META)
    split = read_json(SPLIT_META)
    ev = read_json(EVENT_META)

    test_counts = (
        traces[traces["split"].eq("test")]
        .groupby("protocol")
        .agg(
            test_rows_per_all_reps=("sample_index", "size"),
            test_events=("event_id", "nunique"),
            test_components=("gm_id", "nunique"),
            test_systems=("system_id", "nunique"),
            reps=("rep", "nunique"),
        )
        .reset_index()
    )
    per_model = (
        traces[traces["split"].eq("test")]
        .groupby(["protocol", "model"])
        .size()
        .groupby("protocol")
        .median()
        .astype(int)
        .to_dict()
    )
    counts = test_counts.set_index("protocol").to_dict("index")

    rows = [
        {
            "protocol": "main event-held-out",
            "purpose": "Source-to-target transfer with target-event holdout; target calibration/test are component-level splits within target events.",
            "source_events": split.get("source_event_count", ""),
            "target_fit_or_cal_events": split.get("target_pool_event_count", ""),
            "target_test_events": split.get("target_test_event_count", ""),
            "target_fit_or_cal_components": split.get("target_pool_component_count", ""),
            "target_test_components": split.get("target_test_component_count", ""),
            "event_overlap_cal_test": split.get("target_pool_test_event_overlap_count", ""),
            "N_target_labels": N_TARGET,
            "test_rows_per_model_all_reps": per_model.get("main event-held-out", ""),
            "test_events_in_trace": counts.get("main event-held-out", {}).get("test_events", ""),
            "systems": mdof.get("n_systems", ""),
        },
        {
            "protocol": "event-disjoint target",
            "purpose": "Stricter stress test where target fit/calibration events and target-test events are mutually disjoint.",
            "source_events": ev.get("source_event_count", ""),
            "target_fit_or_cal_events": ev.get("target_fit_event_count", ""),
            "target_test_events": ev.get("target_test_event_count", ""),
            "target_fit_or_cal_components": ev.get("target_fit_components", ""),
            "target_test_components": ev.get("target_test_components", ""),
            "event_overlap_cal_test": 0,
            "N_target_labels": N_TARGET,
            "test_rows_per_model_all_reps": per_model.get("event-disjoint target", ""),
            "test_events_in_trace": counts.get("event-disjoint target", {}).get("test_events", ""),
            "systems": mdof.get("n_systems", ""),
        },
    ]
    table = pd.DataFrame(rows)
    table["recorded_population"] = (
        f"{pop.get('n_events_used', '')} events, {pop.get('n_components', '')} components, "
        f"Mw {pop.get('magnitude_min', '')}-{pop.get('magnitude_max', '')}"
    )
    return table


def build_table2() -> pd.DataFrame:
    rows = [
        {
            "model_label": "Ridge direct",
            "training_mode": "Target-label direct regression",
            "main_settings": "StandardScaler; Ridge alpha=1.0",
            "engineering_role": "Linear conservative baseline for finite target labels",
        },
        {
            "model_label": "RF direct",
            "training_mode": "Target-label direct regression",
            "main_settings": "RandomForestRegressor; 200 trees; min leaf=1",
            "engineering_role": "Non-parametric tree baseline",
        },
        {
            "model_label": "HGB direct",
            "training_mode": "Target-label direct regression",
            "main_settings": "HistGradientBoosting; 200 iterations; learning rate=0.05; max leaves=15; L2=0.01",
            "engineering_role": "Sklearn boosting baseline",
        },
        {
            "model_label": "XGB direct",
            "training_mode": "Target-label direct regression",
            "main_settings": "XGBoost; 200 trees; depth=3; learning rate=0.05; subsample=0.9; colsample=0.9; lambda=1.0",
            "engineering_role": "High-capacity boosting comparator",
        },
        {
            "model_label": "LGBM direct",
            "training_mode": "Target-label direct regression",
            "main_settings": "LightGBM; 200 trees; depth=3; learning rate=0.05; subsample=0.9; colsample=0.9; lambda=1.0",
            "engineering_role": "High-capacity boosting comparator",
        },
        {
            "model_label": "MLP scratch",
            "training_mode": "Target labels only",
            "main_settings": "MLP (64, 32), ReLU, Adam; lr=1e-3; max_iter=400; batch <=32",
            "engineering_role": "Same neural architecture without source pretraining",
        },
        {
            "model_label": "MLP finetune",
            "training_mode": "Source pretraining plus target fine-tuning",
            "main_settings": "Source MLP (64, 32), early stopping; target partial_fit 120 epochs; lr=3e-4",
            "engineering_role": "Foundation/surrogate-transfer candidate",
        },
    ]
    table = pd.DataFrame(rows)
    table["features"] = "8 ground-motion scalars + 6 structural/system scalars"
    return table


def fmt_ci(mean: float, lo: float, hi: float) -> str:
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def build_table3() -> pd.DataFrame:
    r21 = pd.read_csv(R21_SUMMARY)
    r21["protocol"] = r21["protocol"].replace(PROTOCOL_MAP)
    r23 = pd.read_csv(R23_METRICS)
    table = r21.merge(
        r23[["protocol", "model_label", "q95_abs_log", "tail_ratio_q95_over_median", "test_calibration_residual_sd_ratio"]],
        on=["protocol", "model_label"],
        how="left",
    )
    table = table.sort_values(["protocol", "rmse_event_mean", "model_label"]).copy()
    out = pd.DataFrame(
        {
            "protocol": table["protocol"],
            "model": table["model_label"],
            "event_count": table["event_count"].astype(int),
            "row_count": table["row_count"].astype(int),
            "RMSE_log_event_mean_CI": [
                fmt_ci(r.rmse_event_mean, r.rmse_ci95_lo, r.rmse_ci95_hi) for r in table.itertuples()
            ],
            "coverage_event_mean_CI": [
                fmt_ci(r.coverage_event_mean, r.coverage_ci95_lo, r.coverage_ci95_hi) for r in table.itertuples()
            ],
            "coverage_gap_vs_0p90": table["coverage_gap"].map(lambda x: f"{x:+.3f}"),
            "interval_score_90_CI": [
                fmt_ci(r.interval_score_90_event_mean, r.interval_score_90_ci95_lo, r.interval_score_90_ci95_hi)
                for r in table.itertuples()
            ],
            "q95_abs_log": table["q95_abs_log"].map(lambda x: f"{x:.3f}"),
            "tail_ratio": table["tail_ratio_q95_over_median"].map(lambda x: f"{x:.2f}"),
            "test_cal_resid_sd_ratio": table["test_calibration_residual_sd_ratio"].map(lambda x: f"{x:.2f}"),
        }
    )
    return out


def build_table4(reliability: pd.DataFrame) -> pd.DataFrame:
    winners = pd.read_csv(R22_WINNERS)
    focus = winners[
        winners["threshold_idr"].eq(0.01) & winners["cost_ratio_false_safe"].isin(COSTS)
    ].copy()
    rel_focus = reliability[reliability["threshold_idr"].eq(0.01)][
        ["protocol", "model_label", "threshold_idr", "false_safe_ci95_hi", "beta_false_safe_cons"]
    ].copy()
    focus = focus.merge(rel_focus, on=["protocol", "model_label", "threshold_idr"], how="left")
    focus["beta_false_safe"] = focus["false_safe_rate"].map(beta_false_safe)
    focus["beta_false_safe_cons"] = focus["beta_false_safe_cons"].fillna(
        focus["false_safe_ci95_hi"].fillna(focus["false_safe_rate"]).map(beta_false_safe)
    )
    out = focus.sort_values(["protocol", "cost_ratio_false_safe"]).copy()
    out = pd.DataFrame(
        {
            "protocol": out["protocol"],
            "IDR_threshold": out["threshold_idr"].map(lambda x: f"{100*x:.1f}%"),
            "false_safe_cost": out["cost_ratio_false_safe"].astype(int),
            "winner": out["model_label"],
            "true_exceed": out["true_exceed_rate"].map(lambda x: f"{x:.3f}"),
            "false_safe": out["false_safe_rate"].map(lambda x: f"{x:.4f}"),
            "false_safe_U95": out["false_safe_ci95_hi"].map(lambda x: f"{x:.4f}"),
            "beta_FS": out["beta_false_safe"].map(lambda x: f"{x:.2f}"),
            "beta_FS_cons": out["beta_false_safe_cons"].map(lambda x: f"{x:.2f}"),
            "false_unsafe": out["false_unsafe_rate"].map(lambda x: f"{x:.4f}"),
            "expected_loss": out["expected_loss"].map(lambda x: f"{x:.3f}"),
        }
    )
    return out


def compute_reliability_detail(traces: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    data = traces[(traces["N"].eq(N_TARGET)) & (traces["split"].eq("test"))].copy()
    for (protocol, model), g in data.groupby(["protocol", "model"]):
        for thr in THRESHOLDS:
            log_thr = np.log10(thr)
            event_rows = []
            for event_id, e in g.groupby("event_id"):
                truth_unsafe = e["y_true_log"].to_numpy(float) > log_thr
                predicted_safe = e["upper_log"].to_numpy(float) <= log_thr
                false_safe = truth_unsafe & predicted_safe
                false_unsafe = (~truth_unsafe) & (~predicted_safe)
                event_rows.append(
                    {
                        "event_id": event_id,
                        "event_false_safe_rate": float(np.mean(false_safe)),
                        "event_false_unsafe_rate": float(np.mean(false_unsafe)),
                        "event_true_exceed_rate": float(np.mean(truth_unsafe)),
                    }
                )
            ev = pd.DataFrame(event_rows)
            false_safe_rate = float(ev["event_false_safe_rate"].mean())
            fs_lo, fs_hi = bootstrap_mean_ci(ev["event_false_safe_rate"].to_numpy(float), rng)
            fu_lo, fu_hi = bootstrap_mean_ci(ev["event_false_unsafe_rate"].to_numpy(float), rng)
            rows.append(
                {
                    "protocol": protocol,
                    "model": model,
                    "model_label": MODEL_LABELS.get(model, model),
                    "threshold_idr": float(thr),
                    "event_count": int(ev["event_id"].nunique()),
                    "true_exceed_rate": float(ev["event_true_exceed_rate"].mean()),
                    "false_safe_rate": false_safe_rate,
                    "false_safe_ci95_lo": fs_lo,
                    "false_safe_ci95_hi": fs_hi,
                    "beta_false_safe": beta_false_safe(false_safe_rate),
                    "beta_false_safe_cons": beta_false_safe(fs_hi),
                    "false_unsafe_rate": float(ev["event_false_unsafe_rate"].mean()),
                    "false_unsafe_ci95_lo": fu_lo,
                    "false_unsafe_ci95_hi": fu_hi,
                    "worst_event_false_safe_rate": float(ev["event_false_safe_rate"].max()),
                    "p90_event_false_safe_rate": float(ev["event_false_safe_rate"].quantile(0.90)),
                }
            )
    return pd.DataFrame(rows)


def build_table5_residual_variance(traces: pd.DataFrame) -> pd.DataFrame:
    """Decompose signed residual variance into event-mean and within-event components."""
    data = traces[(traces["N"].eq(N_TARGET)) & (traces["split"].eq("test"))].copy()
    data["signed_residual_log"] = data["y_true_log"].astype(float) - data["y_pred_log"].astype(float)
    rows = []
    for (protocol, model), g in data.groupby(["protocol", "model"]):
        ev = (
            g.groupby("event_id")
            .agg(
                event_mean_residual=("signed_residual_log", "mean"),
                event_size=("signed_residual_log", "size"),
                event_rmse=("signed_residual_log", lambda x: float(np.sqrt(np.mean(np.square(x))))),
                event_within_var=("signed_residual_log", lambda x: float(np.var(x, ddof=1)) if len(x) > 1 else 0.0),
            )
            .reset_index()
        )
        between = float(np.var(ev["event_mean_residual"], ddof=1)) if len(ev) > 1 else 0.0
        within = float(ev["event_within_var"].mean())
        total = between + within
        rows.append(
            {
                "protocol": protocol,
                "model": model,
                "model_label": MODEL_LABELS.get(model, model),
                "event_count": int(ev["event_id"].nunique()),
                "between_event_residual_var": between,
                "within_event_residual_var": within,
                "between_event_share": between / total if total > 0 else np.nan,
                "p90_abs_event_mean_residual": float(ev["event_mean_residual"].abs().quantile(0.90)),
                "mean_event_rmse": float(ev["event_rmse"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["protocol", "between_event_share", "model_label"], ascending=[True, False, True])


def build_table6_gate_summary(reliability: pd.DataFrame, beta_target: float = GATE_BETA_TARGET) -> pd.DataFrame:
    focus_models = ["xgb_direct", "lgbm_direct", "ridge_direct", "pretrained_finetune", "scratch_mlp"]
    focus = reliability[
        reliability["threshold_idr"].eq(0.01) & reliability["model"].isin(focus_models)
    ].copy()
    focus["gate_target_beta"] = beta_target
    focus["gate_status"] = np.where(focus["beta_false_safe_cons"] >= beta_target, "Pass", "Fail")
    focus["action"] = np.where(
        focus["gate_status"].eq("Pass"),
        "eligible; select by loss/PFU",
        "collect labels, widen interval, or use NTHA fallback",
    )
    focus = focus.sort_values(["protocol", "beta_false_safe_cons"], ascending=[True, False])
    return focus[
        [
            "protocol",
            "model_label",
            "threshold_idr",
            "false_safe_rate",
            "false_safe_ci95_hi",
            "beta_false_safe",
            "beta_false_safe_cons",
            "gate_target_beta",
            "gate_status",
            "action",
        ]
    ]


def pareto_mask(df: pd.DataFrame) -> pd.Series:
    """Return True for non-dominated points: higher beta and lower false-unsafe."""
    vals = df[["beta_false_safe", "false_unsafe_rate"]].to_numpy(float)
    keep = []
    for i, (beta_i, fu_i) in enumerate(vals):
        dominated = False
        for j, (beta_j, fu_j) in enumerate(vals):
            if i == j:
                continue
            if (beta_j >= beta_i and fu_j <= fu_i) and (beta_j > beta_i or fu_j < fu_i):
                dominated = True
                break
        keep.append(not dominated)
    return pd.Series(keep, index=df.index)


def draw_reliability_figure(reliability: pd.DataFrame, detail: pd.DataFrame) -> None:
    selected = ["xgb_direct", "lgbm_direct", "hgb_direct", "ridge_direct", "pretrained_finetune", "scratch_mlp"]
    fig = plt.figure(figsize=(7.4, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95], hspace=0.42, wspace=0.42)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    for ax, protocol in [(ax_a, "event-disjoint target"), (ax_b, "main event-held-out")]:
        sub = reliability[reliability["protocol"].eq(protocol) & reliability["model"].isin(selected)].copy()
        for model, g in sub.groupby("model"):
            g = g.sort_values("threshold_idr")
            if "beta_false_safe_cons" in g:
                ax.fill_between(
                    100 * g["threshold_idr"],
                    g["beta_false_safe_cons"],
                    g["beta_false_safe"],
                    color=PALETTE.get(model, "#777777"),
                    alpha=0.075,
                    linewidth=0,
                )
            ax.plot(
                100 * g["threshold_idr"],
                g["beta_false_safe"],
                "-o",
                lw=1.25,
                ms=3.0,
                color=PALETTE.get(model, "#777777"),
                label=MODEL_SHORT.get(model, model),
            )
        ax.set_xscale("log")
        ax.set_xticks([0.5, 1.0, 2.0, 4.0])
        ax.set_xticklabels(["0.5", "1", "2", "4"])
        ax.set_xlabel("IDR threshold (%)")
        ax.set_ylabel(r"False-safe reliability index, $\beta_{FS}$")
        ax.set_title(protocol, loc="left", fontweight="bold")
        ax.grid(axis="y", color="#dddddd", lw=0.6)
        ax.axhline(2.0, color="#999999", lw=0.7, ls=":")
    ax_a.legend(ncol=3, loc="lower right", fontsize=5.8)

    focus = reliability[reliability["threshold_idr"].eq(0.01) & reliability["model"].isin(selected)].copy()
    for protocol, marker in [("main event-held-out", "o"), ("event-disjoint target", "s")]:
        sub = focus[focus["protocol"].eq(protocol)].copy()
        sub["pareto"] = pareto_mask(sub)
        for _, r in sub.iterrows():
            ax_c.scatter(
                r["false_unsafe_rate"],
                r["beta_false_safe"],
                s=80 if r["pareto"] else 42,
                marker=marker,
                color=PALETTE.get(r["model"], "#777777"),
                edgecolor="black" if r["pareto"] else "white",
                linewidth=0.6,
                alpha=0.92 if r["pareto"] else 0.65,
            )
            ax_c.text(
                r["false_unsafe_rate"] + 0.003,
                r["beta_false_safe"] + 0.015,
                MODEL_SHORT.get(r["model"], r["model"]),
                fontsize=5.8,
            )
    ax_c.set_xlabel("False-unsafe rate at 1% IDR")
    ax_c.set_ylabel(r"$\beta_{FS}$ at 1% IDR")
    ax_c.set_title("False-safe/false-unsafe frontier", loc="left", fontweight="bold")
    ax_c.grid(axis="both", color="#dddddd", lw=0.6)
    ax_c.text(0.02, 0.04, "outline = non-dominated within protocol", transform=ax_c.transAxes, fontsize=5.7, color="#555555")

    worst = focus.pivot_table(
        index="model", columns="protocol", values="worst_event_false_safe_rate", aggfunc="first"
    ).reindex(selected)
    mat = worst[["main event-held-out", "event-disjoint target"]].to_numpy(float)
    im = ax_d.imshow(mat, cmap="YlOrRd", vmin=0.0, vmax=max(0.08, float(np.nanmax(mat))), aspect="auto")
    ax_d.set_yticks(np.arange(len(selected)))
    ax_d.set_yticklabels([MODEL_SHORT[m] for m in selected])
    ax_d.set_xticks([0, 1])
    ax_d.set_xticklabels(["main", "event-\ndisjoint"])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            ax_d.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=6.2, color="white" if val > 0.045 else "#222222")
    ax_d.set_title("Worst-event concentration", loc="left", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax_d, fraction=0.046, pad=0.02)
    cbar.set_label("Worst-event false-safe rate")

    fig.suptitle(
        "False-safe reliability index for interval-based seismic screening",
        x=0.02,
        y=0.988,
        ha="left",
        fontsize=9.4,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.954,
        r"$\beta_{FS}=-\Phi^{-1}(P_{FS})$ converts false-safe probability into a reliability-style scale; faint bands use event-bootstrap upper 95% false-safe bounds.",
        ha="left",
        fontsize=6.6,
        color="#444444",
    )
    for ax, letter in zip([ax_a, ax_b, ax_c, ax_d], "abcd"):
        ax.text(-0.14, 1.08, letter, transform=ax.transAxes, fontsize=10.5, fontweight="bold", ha="left", va="top")

    base = FIG_DIR / "fig_r24_false_safe_reliability_index"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def draw_gate_figure(reliability: pd.DataFrame, beta_target: float = GATE_BETA_TARGET) -> None:
    selected = ["xgb_direct", "lgbm_direct", "ridge_direct", "pretrained_finetune", "scratch_mlp"]
    winners = pd.read_csv(R22_WINNERS)
    rel_gate = reliability[["protocol", "model_label", "threshold_idr", "beta_false_safe_cons"]].copy()
    winners = winners.merge(rel_gate, on=["protocol", "model_label", "threshold_idr"], how="left")
    winners["gate_pass"] = winners["beta_false_safe_cons"] >= beta_target

    fig = plt.figure(figsize=(7.4, 6.35))
    gs = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.46)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    for ax, protocol in zip(axes[:2], ["event-disjoint target", "main event-held-out"]):
        sub = reliability[reliability["protocol"].eq(protocol) & reliability["model"].isin(selected)].copy()
        for model, g in sub.groupby("model"):
            g = g.sort_values("threshold_idr")
            ax.plot(
                100 * g["threshold_idr"],
                g["beta_false_safe_cons"],
                "-o",
                lw=1.25,
                ms=3.0,
                color=PALETTE.get(model, "#777777"),
                label=MODEL_SHORT.get(model, model),
            )
        for beta, style, alpha in [(2.0, ":", 0.55), (2.5, "--", 0.75), (3.0, "-.", 0.55)]:
            ax.axhline(beta, color="#444444", lw=0.75, ls=style, alpha=alpha)
            ax.text(4.15, beta + 0.02, rf"$\beta^*={beta:.1f}$", fontsize=5.6, va="bottom", color="#444444")
        ax.set_xscale("log")
        ax.set_xticks([0.5, 1.0, 2.0, 4.0])
        ax.set_xticklabels(["0.5", "1", "2", "4"])
        ax.set_xlabel("IDR threshold (%)")
        ax.set_ylabel(r"Conservative filter index, $\beta_{FS,cons}$")
        ax.set_title(protocol, loc="left", fontweight="bold")
        ax.grid(axis="y", color="#dddddd", lw=0.6)
        ax.set_ylim(1.55, 5.0)
    axes[0].legend(ncol=3, loc="lower right", fontsize=5.7)

    thresholds = sorted(winners["threshold_idr"].dropna().unique())
    costs = sorted(winners["cost_ratio_false_safe"].dropna().unique())
    cmap_models = {
        "XGB direct": "#4C78A8",
        "LGBM direct": "#F58518",
        "HGB direct": "#54A24B",
        "Ridge direct": "#6B6B6B",
        "MLP finetune": "#72B7B2",
        "MLP scratch": "#D62728",
    }
    for ax, protocol in zip(axes[2:], ["event-disjoint target", "main event-held-out"]):
        sub = winners[winners["protocol"].eq(protocol)].copy()
        for i, cost in enumerate(costs):
            for j, thr in enumerate(thresholds):
                cell = sub[sub["cost_ratio_false_safe"].eq(cost) & sub["threshold_idr"].eq(thr)]
                if cell.empty:
                    continue
                r = cell.iloc[0]
                color = cmap_models.get(str(r["model_label"]), "#CCCCCC")
                rect = Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, alpha=0.82, edgecolor="white", lw=0.8)
                ax.add_patch(rect)
                if not bool(r["gate_pass"]):
                    hatch = Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="none", edgecolor="#222222", hatch="////", lw=0.0)
                    ax.add_patch(hatch)
                ax.text(
                    j,
                    i,
                    f"{MODEL_SHORT.get(str(r['model']), str(r['model_label']).split()[0])}\n{r['expected_loss']:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.4,
                    color="white" if r["expected_loss"] > 0.10 else "#111111",
                )
        ax.set_xlim(-0.5, len(thresholds) - 0.5)
        ax.set_ylim(len(costs) - 0.5, -0.5)
        ax.set_xticks(range(len(thresholds)))
        ax.set_xticklabels([f"{100*t:g}%" for t in thresholds], rotation=30, ha="right")
        ax.set_yticks(range(len(costs)))
        ax.set_yticklabels([str(int(c)) for c in costs])
        ax.set_xlabel("IDR threshold")
        ax.set_ylabel("False-safe cost ratio")
        short_protocol = "event-disjoint" if protocol == "event-disjoint target" else "main"
        ax.set_title(f"{short_protocol}: loss winner + filter", loc="left", fontweight="bold")
        ax.text(
            0.01,
            -0.18,
            rf"hatching = winner fails illustrative $\beta^*={beta_target:.1f}$ filter",
            transform=ax.transAxes,
            fontsize=5.7,
            color="#444444",
        )

    fig.suptitle(
        "False-safe reliability filter and constrained decision surface",
        x=0.02,
        y=0.99,
        ha="left",
        fontsize=9.4,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.955,
        r"The filter uses $\beta_{FS,cons}=-\Phi^{-1}(P_{FS}^{U95})$; target lines are illustrative diagnostic thresholds, not code targets.",
        ha="left",
        fontsize=6.5,
        color="#444444",
    )
    for ax, letter in zip(axes, "abcd"):
        ax.text(-0.14, 1.08, letter, transform=ax.transAxes, fontsize=10.5, fontweight="bold", ha="left", va="top")

    base = FIG_DIR / "fig_r26_false_safe_reliability_gate"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_report(
    table1: pd.DataFrame,
    table3: pd.DataFrame,
    table4: pd.DataFrame,
    table5: pd.DataFrame,
    table6: pd.DataFrame,
    reliability: pd.DataFrame,
) -> None:
    lines = [
        "# R24 Structural Safety tables and false-safe reliability index",
        "",
        "## Manuscript contribution strengthened",
        "",
        "The analysis upgrades the manuscript from a metric-comparison note to a reliability-oriented article. It maps interval screening errors to a false-safe probability and a reliability-style index, while preserving event-level separation and event-equal aggregation.",
        "",
        "## Dataset/protocol facts used",
        "",
        table1.to_markdown(index=False),
        "",
        "## Main metric summary",
        "",
        table3.to_markdown(index=False),
        "",
        "## Decision-risk reliability summary at 1% IDR",
        "",
        table4.to_markdown(index=False),
        "",
        "## Event-level residual variance decomposition",
        "",
        table5.to_markdown(index=False),
        "",
        "## False-safe filter decision summary",
        "",
        table6.to_markdown(index=False),
        "",
        "## False-safe reliability winners by protocol at 1% IDR",
        "",
        "| protocol | highest beta_FS model | beta_FS | false-safe | false-unsafe |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    focus = reliability[reliability["threshold_idr"].eq(0.01)].copy()
    for protocol, g in focus.groupby("protocol"):
        r = g.loc[g["beta_false_safe"].idxmax()]
        lines.append(
            f"| {protocol} | {r['model_label']} | {r['beta_false_safe']:.2f} | "
            f"{r['false_safe_rate']:.4f} | {r['false_unsafe_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "The reliability index is an empirical diagnostic for screening-rule false-safe probability, not a replacement for component/system-level code calibration or PBEE/FEMA P-58 loss assessment. Because the rates are event-equal and clustered, the conservative beta_FS column uses an event-bootstrap upper 95% false-safe bound rather than an independent Bernoulli assumption.",
            "",
            "## Generated files",
            "",
            f"- Tables: `{OUT}`",
            f"- Figure: `{FIG_DIR / 'fig_r24_false_safe_reliability_index.pdf'}`",
            f"- LaTeX fragments: `{LATEX}`",
        ]
    )
    (OUT / "R24_STRUCTURAL_SAFETY_TABLES_AND_RELIABILITY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def write_all_latex_fragments(
    table1: pd.DataFrame,
    table2: pd.DataFrame,
    table3: pd.DataFrame,
    table4: pd.DataFrame,
    table5: pd.DataFrame,
    table6: pd.DataFrame,
) -> None:
    LATEX.mkdir(parents=True, exist_ok=True)
    rows1 = [
        [
            r["protocol"],
            r["source_events"],
            r["target_fit_or_cal_events"],
            r["target_test_events"],
            r["target_fit_or_cal_components"],
            r["target_test_components"],
            r["event_overlap_cal_test"],
            r["N_target_labels"],
            r["systems"],
        ]
        for _, r in table1.iterrows()
    ]
    write_latex_table(
        LATEX / "table1_protocol_summary.tex",
        "Dataset and separation protocol used for the reliability audit.",
        "tab:protocol_summary",
        ["Protocol", "Source events", "Cal/fit events", "Test events", "Cal/fit comps.", "Test comps.", "Event overlap", "N", "Systems"],
        rows1,
        "lrrrrrrrr",
    )

    rows2 = [
        [r["model_label"], r["training_mode"], r["main_settings"], r["engineering_role"]]
        for _, r in table2.iterrows()
    ]
    write_latex_table(
        LATEX / "table2_model_settings.tex",
        "Model families and training definitions. All methods use the same ground-motion and structural feature set.",
        "tab:model_settings",
        ["Model", "Training definition", "Key settings", "Role in comparison"],
        rows2,
        "p{0.16\\textwidth}p{0.22\\textwidth}p{0.34\\textwidth}p{0.22\\textwidth}",
    )

    selected = table3[
        table3["model"].isin(["XGB direct", "LGBM direct", "Ridge direct", "MLP scratch", "MLP finetune"])
    ].copy()
    rows3 = [
        [
            r["protocol"],
            r["model"],
            r["RMSE_log_event_mean_CI"],
            r["coverage_event_mean_CI"],
            r["interval_score_90_CI"],
            r["q95_abs_log"],
            r["test_cal_resid_sd_ratio"],
        ]
        for _, r in selected.iterrows()
    ]
    write_latex_table(
        LATEX / "table3_metric_summary.tex",
        "Event-clustered predictive and interval metrics at the N=500 target-label budget.",
        "tab:metric_summary",
        ["Protocol", "Model", "RMSE", "Coverage", "Interval score", "q95 |resid|", "SD ratio"],
        rows3,
        "llccccc",
    )

    rows4 = [
        [
            r["protocol"],
            r["false_safe_cost"],
            r["winner"],
            r["true_exceed"],
            r["false_safe"],
            r["false_safe_U95"],
            r["beta_FS"],
            r["beta_FS_cons"],
            r["false_unsafe"],
            r["expected_loss"],
        ]
        for _, r in table4.iterrows()
    ]
    write_latex_table(
        LATEX / "table4_decision_reliability.tex",
        "Decision-risk winners and false-safe reliability index at the 1% IDR threshold. U95 is the event-bootstrap upper 95% bound for event-equal false-safe probability; beta cons uses that upper bound.",
        "tab:decision_reliability",
        ["Protocol", "C", "Winner", "True exceed", "False-safe", "U95", "beta FS", "beta cons", "False-unsafe", "Loss"],
        rows4,
        "llcccccccc",
    )

    selected5 = table5[
        table5["model_label"].isin(["XGB direct", "LGBM direct", "Ridge direct", "MLP scratch", "MLP finetune"])
    ].copy()
    rows5 = [
        [
            r["protocol"],
            r["model_label"],
            f"{r['between_event_residual_var']:.4f}",
            f"{r['within_event_residual_var']:.4f}",
            f"{100*r['between_event_share']:.1f}",
            f"{r['p90_abs_event_mean_residual']:.3f}",
            f"{r['mean_event_rmse']:.3f}",
        ]
        for _, r in selected5.iterrows()
    ]
    write_latex_table(
        LATEX / "table5_variance_decomposition.tex",
        "Event-level residual variance decomposition. Between-event variance is the variance of event-mean signed residuals; within-event variance is averaged event-wise, matching the event-equal aggregation used elsewhere.",
        "tab:variance_decomposition",
        ["Protocol", "Model", "Between var", "Within var", "Between share (%)", "P90 |event mean|", "Mean event RMSE"],
        rows5,
        "llccccc",
    )

    rows6 = [
        [
            r["protocol"],
            r["model_label"],
            "1.0%",
            f"{r['false_safe_rate']:.4f}",
            f"{r['false_safe_ci95_hi']:.4f}",
            f"{r['beta_false_safe']:.2f}",
            f"{r['beta_false_safe_cons']:.2f}",
            f"{r['gate_target_beta']:.1f}",
            r["gate_status"],
            r["action"],
        ]
        for _, r in table6.iterrows()
    ]
    write_latex_table(
        LATEX / "table6_gate_summary.tex",
        "False-safe reliability filter summary at the 1% IDR threshold. The beta target is an illustrative diagnostic filter, not a code-calibrated target reliability.",
        "tab:gate_summary",
        ["Protocol", "Model", "Threshold", "P_FS", "U95", "beta FS", "beta cons", "beta target", "Filter", "Action"],
        rows6,
        "llcccccccp{0.24\\textwidth}",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    traces = load_traces()
    table1 = build_table1(traces)
    table2 = build_table2()
    table3 = build_table3()
    reliability = compute_reliability_detail(traces)
    table4 = build_table4(reliability)
    table5 = build_table5_residual_variance(traces)
    table6 = build_table6_gate_summary(reliability)

    table1.to_csv(OUT / "table1_protocol_summary.csv", index=False)
    table2.to_csv(OUT / "table2_model_settings.csv", index=False)
    table3.to_csv(OUT / "table3_metric_summary.csv", index=False)
    table4.to_csv(OUT / "table4_decision_reliability_summary.csv", index=False)
    table5.to_csv(OUT / "table5_residual_variance_decomposition.csv", index=False)
    table6.to_csv(OUT / "table6_false_safe_gate_summary.csv", index=False)
    reliability.to_csv(OUT / "false_safe_reliability_detail.csv", index=False)

    draw_reliability_figure(reliability, pd.read_csv(R22_DETAIL))
    draw_gate_figure(reliability)
    write_all_latex_fragments(table1, table2, table3, table4, table5, table6)
    write_report(table1, table3, table4, table5, table6, reliability)

    # Copy the new display figure into the flat LaTeX source folder.
    for suffix in [".pdf", ".png", ".svg"]:
        for stem, fig_name in [
            ("fig_r24_false_safe_reliability_index", "Figure_5"),
            ("fig_r26_false_safe_reliability_gate", "Figure_6"),
        ]:
            src = FIG_DIR / f"{stem}{suffix}"
            if src.exists():
                dst = LATEX / f"{fig_name}{suffix}"
                dst.write_bytes(src.read_bytes())

    print(f"[64] wrote R24 tables to {OUT}")
    print(f"[64] wrote figure to {FIG_DIR / 'fig_r24_false_safe_reliability_index.pdf'}")


if __name__ == "__main__":
    main()

