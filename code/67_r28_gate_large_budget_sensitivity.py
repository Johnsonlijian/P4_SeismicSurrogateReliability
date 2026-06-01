"""R28 false-safe gate stability from true event-disjoint N=1000/2000 exports."""
from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm

PROJECT = Path(r"R:\NAS_DRIVE\IMUT\1-Research_Output\1-Papers\1_In_Preparation\2026-Claude\projects\P4_SeismicFoundationModel")
TRACE = PROJECT / "outputs" / "high_target" / "r28_event_disjoint_large_budget" / "event_disjoint_large_budget_predictions.csv"
RECORDS = PROJECT / "outputs" / "high_target" / "recorded_nsmp_full" / "nsmp_recorded_records.csv"
OUT = PROJECT / "outputs" / "high_target" / "r28_gate_large_budget_sensitivity"
FIGOUT = PROJECT / "outputs" / "figures" / "high_target"
LATEX = PROJECT / "submission" / "structural_safety_2026-06-01" / "latex_source_flat"

THRESHOLD_IDR = 0.01
LOG_THR = math.log10(THRESHOLD_IDR)
BETA_TARGET = 2.5
BUDGETS = [50, 100, 250, 500, 1000, 2000]
INTERVAL_LEVELS = [0.90, 0.95, 0.975]
MODELS = ["ridge_direct", "lgbm_direct", "xgb_direct", "scratch_mlp"]
MODEL_LABELS = {
    "ridge_direct": "Ridge direct",
    "lgbm_direct": "LGBM direct",
    "xgb_direct": "XGB direct",
    "scratch_mlp": "MLP scratch",
}
COLORS = {
    "ridge_direct": "#2C7BB6",
    "lgbm_direct": "#1A9850",
    "xgb_direct": "#D73027",
    "scratch_mlp": "#7B3294",
}


def conformal_quantile(values: np.ndarray, level: float) -> float:
    vals = np.sort(np.asarray(values, dtype=float))
    n = len(vals)
    if n == 0:
        raise ValueError("empty residual array")
    k = int(math.ceil(level * (n + 1)))
    k = min(max(k, 1), n)
    return float(vals[k - 1])


def beta_from_probability(p: float) -> float:
    p = min(max(float(p), 1e-9), 1 - 1e-9)
    return float(norm.isf(p))


def event_bootstrap_upper(flags: np.ndarray, event_id: np.ndarray, rng: np.random.Generator, n_boot: int = 2000) -> tuple[float, float]:
    df = pd.DataFrame({"flag": flags.astype(float), "event_id": event_id})
    event_means = df.groupby("event_id", observed=True)["flag"].mean().to_numpy(dtype=float)
    point = float(event_means.mean())
    draws = rng.choice(event_means, size=(n_boot, len(event_means)), replace=True).mean(axis=1)
    return point, float(np.quantile(draws, 0.95))


def metrics_for(test: pd.DataFrame, q_value: float, rng: np.random.Generator) -> dict[str, float]:
    y_true = test["y_true_log"].to_numpy(dtype=float)
    y_pred = test["y_pred_log"].to_numpy(dtype=float)
    event_id = test["event_id"].to_numpy()
    truth_unsafe = y_true > LOG_THR
    predicted_safe = (y_pred + q_value) <= LOG_THR
    false_safe = truth_unsafe & predicted_safe
    false_unsafe = (~truth_unsafe) & (~predicted_safe)
    pfs, pfs_hi = event_bootstrap_upper(false_safe, event_id, rng)
    pfu, pfu_hi = event_bootstrap_upper(false_unsafe, event_id, rng)
    return {
        "p_false_safe": pfs,
        "p_false_safe_upper95": pfs_hi,
        "beta_false_safe": beta_from_probability(pfs),
        "beta_false_safe_cons": beta_from_probability(pfs_hi),
        "p_false_unsafe": pfu,
        "p_false_unsafe_upper95": pfu_hi,
        "rmse_log": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "q_value_log": float(q_value),
        "event_count": int(pd.Series(event_id).nunique()),
        "test_rows": int(len(test)),
    }


def summarize(detail: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    keys = ["beta_false_safe_cons", "p_false_safe", "p_false_unsafe", "rmse_log", "q_value_log"]
    agg = {}
    for key in keys:
        agg[f"{key}_median"] = (key, "median")
        agg[f"{key}_p05"] = (key, lambda s: float(np.quantile(s, 0.05)))
        agg[f"{key}_p95"] = (key, lambda s: float(np.quantile(s, 0.95)))
    agg["replicate_count"] = ("rep", "nunique")
    out = detail.groupby(group_cols, observed=True).agg(**agg).reset_index()
    out["gate"] = np.where(out["beta_false_safe_cons_median"] >= BETA_TARGET, "Pass", "Fail")
    return out


def build_label_budget(trace: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    rng_base = np.random.default_rng(20260628)
    test = trace[trace["split"].eq("test")].copy()
    for (model, n_budget, rep), g in test.groupby(["model", "N", "rep"], observed=True):
        if model not in MODELS or int(n_budget) not in BUDGETS:
            continue
        q_value = float(g["q_value_log"].iloc[0])
        met = metrics_for(g, q_value, np.random.default_rng(int(rng_base.integers(0, 2**31 - 1))))
        met.update({
            "analysis": "true event-disjoint target-label budget",
            "model": model,
            "model_label": MODEL_LABELS[model],
            "target_label_budget": int(n_budget),
            "rep": int(rep),
            "interval_level": 0.90,
        })
        rows.append(met)
    detail = pd.DataFrame(rows)
    summary = summarize(detail, ["analysis", "model", "model_label", "target_label_budget", "interval_level"])
    return detail, summary


def build_interval_sensitivity(trace: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    rng_base = np.random.default_rng(20260629)
    max_budget = max(BUDGETS)
    for (model, rep), g in trace[trace["N"].eq(max_budget)].groupby(["model", "rep"], observed=True):
        if model not in MODELS:
            continue
        calib = g[g["split"].eq("calibration")]["residual_abs_log"].to_numpy(dtype=float)
        test = g[g["split"].eq("test")].copy()
        for level in INTERVAL_LEVELS:
            q_value = conformal_quantile(calib, level)
            met = metrics_for(test, q_value, np.random.default_rng(int(rng_base.integers(0, 2**31 - 1))))
            met.update({
                "analysis": "interval widening at N=2000",
                "model": model,
                "model_label": MODEL_LABELS[model],
                "target_label_budget": int(max_budget),
                "rep": int(rep),
                "interval_level": float(level),
                "calibration_residuals": int(len(calib)),
            })
            rows.append(met)
    detail = pd.DataFrame(rows)
    summary = summarize(detail, ["analysis", "model", "model_label", "target_label_budget", "interval_level"])
    return detail, summary


def write_latex(label_summary: pd.DataFrame, interval_summary: pd.DataFrame) -> None:
    LATEX.mkdir(parents=True, exist_ok=True)
    order = {m: i for i, m in enumerate(MODELS)}
    rows = []
    work = label_summary.copy()
    work["model_order"] = work["model"].map(order)
    for _, r in work.sort_values(["model_order", "target_label_budget"]).iterrows():
        rows.append(
            f"{r['model_label']} & {int(r['target_label_budget'])} & {r['beta_false_safe_cons_median']:.2f} & "
            f"{100*r['p_false_safe_median']:.2f} & {100*r['p_false_unsafe_median']:.1f} & {r['gate']} \\\\"
        )
    table7 = """\\begin{table}[t]
\\caption{True event-disjoint target-label-budget stability of the false-safe gate at the 1\\% IDR threshold. N=1000 and N=2000 are regenerated prediction exports rather than extrapolations from the N=500 trace.}
\\label{tab:label_budget_gate_stability}
\\centering
\\scriptsize
\\begin{tabular}{lrrrrl}
\\toprule
Model & N & $\\beta_{FS,cons}$ & $P_{FS}$ (\\%) & $P_{FU}$ (\\%) & Gate \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\end{table}
"""
    (LATEX / "table7_label_budget_gate_stability.tex").write_text(table7, encoding="utf-8")

    rows = []
    work = interval_summary.copy()
    work["model_order"] = work["model"].map(order)
    for _, r in work.sort_values(["model_order", "interval_level"]).iterrows():
        rows.append(
            f"{r['model_label']} & {int(r['target_label_budget'])} & {r['interval_level']:.3f} & {r['q_value_log_median']:.3f} & "
            f"{r['beta_false_safe_cons_median']:.2f} & {100*r['p_false_safe_median']:.2f} & "
            f"{100*r['p_false_unsafe_median']:.1f} & {r['gate']} \\\\"
        )
    table8 = """\\begin{table}[t]
\\caption{Interval-widening sensitivity at N=2000 under event-disjoint testing. Wider conformal intervals can recover false-safe gate eligibility, but they increase false-unsafe screening burden.}
\\label{tab:interval_widening_sensitivity}
\\centering
\\small
\\begin{tabular}{lrrrrrrl}
\\toprule
Model & N & Level & $q$ & $\\beta_{FS,cons}$ & $P_{FS}$ (\\%) & $P_{FU}$ (\\%) & Gate \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\end{table}
"""
    (LATEX / "table8_interval_widening_sensitivity.tex").write_text(table8, encoding="utf-8")


def draw_figure(label_summary: pd.DataFrame, interval_summary: pd.DataFrame) -> None:
    FIGOUT.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.2,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.2,
        "legend.fontsize": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig = plt.figure(figsize=(7.6, 6.8), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, left=0.08, right=0.98, bottom=0.12, top=0.91, wspace=0.38, hspace=0.42)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    for model in MODELS:
        s = label_summary[label_summary["model"].eq(model)].sort_values("target_label_budget")
        x = s["target_label_budget"].to_numpy(dtype=float)
        y = s["beta_false_safe_cons_median"].to_numpy(dtype=float)
        lo = s["beta_false_safe_cons_p05"].to_numpy(dtype=float)
        hi = s["beta_false_safe_cons_p95"].to_numpy(dtype=float)
        ax1.plot(x, y, marker="o", lw=1.7, color=COLORS[model], label=MODEL_LABELS[model])
        ax1.fill_between(x, lo, hi, color=COLORS[model], alpha=0.13, linewidth=0)
        ax2.plot(x, 100 * s["p_false_unsafe_median"], marker="s", lw=1.5, color=COLORS[model])
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xticks(BUDGETS)
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.grid(True, color="#e6e6e6", linewidth=0.6)
    ax1.axhline(BETA_TARGET, color="#222222", lw=1.0, ls="--")
    ax1.text(55, BETA_TARGET + 0.05, r"gate target $\beta^*=2.5$", fontsize=7.2)
    ax1.set_title("a  True budget gate stability")
    ax1.set_xlabel("Target labels N")
    ax1.set_ylabel(r"Conservative gate index, $\beta_{FS,cons}$")
    ax1.legend(frameon=False, ncol=1, loc="lower right")
    ax2.set_title("b  False-unsafe burden")
    ax2.set_xlabel("Target labels N")
    ax2.set_ylabel(r"Median $P_{FU}$ (%)")

    heat = label_summary.pivot(index="model_label", columns="target_label_budget", values="beta_false_safe_cons_median")
    order = [MODEL_LABELS[m] for m in MODELS]
    heat = heat.loc[order, BUDGETS]
    im = ax3.imshow(heat.to_numpy(), cmap="RdYlBu", vmin=1.7, vmax=3.2, aspect="auto")
    ax3.set_title("c  Gate eligibility map")
    ax3.set_xticks(range(len(BUDGETS)), [str(n) for n in BUDGETS], rotation=30, ha="right")
    ax3.set_yticks(range(len(order)), order)
    ax3.set_xlabel("Target labels N")
    for i, label in enumerate(order):
        for j, n in enumerate(BUDGETS):
            val = float(heat.loc[label, n])
            ax3.text(j, i, f"{val:.2f}\n{'P' if val >= BETA_TARGET else 'F'}", ha="center", va="center", fontsize=6.2, color="#111111")
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.03)

    for model in MODELS:
        s = interval_summary[interval_summary["model"].eq(model)].sort_values("interval_level")
        ax4.plot(s["interval_level"], s["beta_false_safe_cons_median"], marker="o", lw=1.7, color=COLORS[model], label=MODEL_LABELS[model])
    ax4.axhline(BETA_TARGET, color="#222222", lw=1.0, ls="--")
    ax4.set_title("d  N=2000 interval sensitivity")
    ax4.set_xlabel("Conformal interval level")
    ax4.set_ylabel(r"Median $\beta_{FS,cons}$")
    ax4.set_xticks(INTERVAL_LEVELS, ["0.90", "0.95", "0.975"])
    ax4.grid(True, color="#e6e6e6", linewidth=0.6)

    fig.suptitle("True event-disjoint false-safe gate stability at the 1% IDR threshold", x=0.08, ha="left", fontsize=11, fontweight="bold")
    base = FIGOUT / "fig_r28_true_budget_gate_sensitivity"
    for ext, dpi in [("pdf", None), ("svg", None), ("png", 450), ("tiff", 600)]:
        fig.savefig(base.with_suffix(f".{ext}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    for ext in ["pdf", "png", "svg"]:
        (LATEX / f"Figure_7.{ext}").write_bytes(base.with_suffix(f".{ext}").read_bytes())


def write_report(label_summary: pd.DataFrame, interval_summary: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = [
        "# R28 true event-disjoint gate stability",
        "",
        "Date: 2026-06-01",
        "",
        "R28 replaces the R27 calibration-subset stress test with true regenerated event-disjoint prediction exports for N=1000 and N=2000.",
        "",
        "## Main budget-gate summary",
        "",
    ]
    for _, r in label_summary.sort_values(["model_label", "target_label_budget"]).iterrows():
        lines.append(f"- {r['model_label']} N={int(r['target_label_budget'])}: beta_FS_cons={r['beta_false_safe_cons_median']:.2f}, PFS={100*r['p_false_safe_median']:.2f}%, PFU={100*r['p_false_unsafe_median']:.1f}%, gate={r['gate']}.")
    lines += ["", "## N=2000 interval sensitivity", ""]
    for _, r in interval_summary.sort_values(["model_label", "interval_level"]).iterrows():
        lines.append(f"- {r['model_label']} level {r['interval_level']:.3f}: beta_FS_cons={r['beta_false_safe_cons_median']:.2f}, PFU={100*r['p_false_unsafe_median']:.1f}%, gate={r['gate']}.")
    (OUT / "R28_GATE_LARGE_BUDGET_SENSITIVITY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    trace = pd.read_csv(TRACE)
    records = pd.read_csv(RECORDS)[["gm_id", "event_id"]].drop_duplicates("gm_id")
    trace = trace.merge(records, on="gm_id", how="left")
    trace = trace[trace["model"].isin(MODELS) & trace["N"].isin(BUDGETS)].copy()
    label_detail, label_summary = build_label_budget(trace)
    interval_detail, interval_summary = build_interval_sensitivity(trace)
    label_detail.to_csv(OUT / "true_budget_gate_detail.csv", index=False)
    label_summary.to_csv(OUT / "true_budget_gate_summary.csv", index=False)
    interval_detail.to_csv(OUT / "n2000_interval_sensitivity_detail.csv", index=False)
    interval_summary.to_csv(OUT / "n2000_interval_sensitivity_summary.csv", index=False)
    write_latex(label_summary, interval_summary)
    draw_figure(label_summary, interval_summary)
    write_report(label_summary, interval_summary)
    print(f"[67] wrote R28 gate large-budget outputs to {OUT}")


if __name__ == "__main__":
    main()
