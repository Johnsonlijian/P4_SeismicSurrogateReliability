"""R27 gate stability and interval sensitivity analyses.

This script is intentionally evidence-bound. It does not invent N=1000 traces.
It uses the event-disjoint N=500 residual trace to stress-test calibration-label
budget and interval widening under fixed trained surrogate predictions.
"""
from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm

PROJECT = Path(r"R:\NAS_DRIVE\IMUT\1-Research_Output\1-Papers\1_In_Preparation\2026-Claude\projects\P4_SeismicFoundationModel")
TRACE = PROJECT / "outputs" / "high_target" / "event_disjoint_conformal_stress" / "event_disjoint_residual_samples.csv"
RECORDS = PROJECT / "outputs" / "high_target" / "recorded_nsmp_full" / "nsmp_recorded_records.csv"
OUT = PROJECT / "outputs" / "high_target" / "r27_gate_stability_sensitivity"
FIGOUT = PROJECT / "outputs" / "figures" / "high_target"
LATEX = PROJECT / "submission" / "structural_safety_2026-06-01" / "latex_source_flat"

THRESHOLD_IDR = 0.01
BETA_TARGET = 2.5
N_BUDGETS = [50, 100, 250]
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
        raise ValueError("empty calibration residual set")
    k = int(math.ceil(level * (n + 1)))
    k = min(max(k, 1), n)
    return float(vals[k - 1])


def beta_from_probability(p: float) -> float:
    p = min(max(float(p), 1e-9), 1 - 1e-9)
    return float(norm.isf(p))


def event_bootstrap_upper(flags: np.ndarray, gm_id: np.ndarray, rng: np.random.Generator, n_boot: int = 1000) -> tuple[float, float]:
    df = pd.DataFrame({"flag": flags.astype(float), "gm_id": gm_id})
    event_means = df.groupby("gm_id", observed=True)["flag"].mean().to_numpy(dtype=float)
    if len(event_means) == 0:
        return float("nan"), float("nan")
    point = float(event_means.mean())
    draws = rng.choice(event_means, size=(n_boot, len(event_means)), replace=True).mean(axis=1)
    upper = float(np.quantile(draws, 0.95))
    return point, upper


def metrics_for(test: pd.DataFrame, q_value: float, rng: np.random.Generator) -> dict[str, float]:
    log_thr = math.log10(THRESHOLD_IDR)
    y_true = test["y_true_log"].to_numpy(dtype=float)
    y_pred = test["y_pred_log"].to_numpy(dtype=float)
    event_key = "event_id" if "event_id" in test.columns else "gm_id"
    event_id = test[event_key].to_numpy()
    truth_unsafe = y_true > log_thr
    predicted_safe = (y_pred + q_value) <= log_thr
    false_safe = truth_unsafe & predicted_safe
    false_unsafe = (~truth_unsafe) & (~predicted_safe)
    pfs, pfs_upper = event_bootstrap_upper(false_safe, event_id, rng)
    pfu, pfu_upper = event_bootstrap_upper(false_unsafe, event_id, rng)
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    return {
        "p_false_safe": pfs,
        "p_false_safe_upper95": pfs_upper,
        "beta_false_safe": beta_from_probability(pfs),
        "beta_false_safe_cons": beta_from_probability(pfs_upper),
        "p_false_unsafe": pfu,
        "p_false_unsafe_upper95": pfu_upper,
        "rmse_log": rmse,
        "q_value_log": float(q_value),
        "event_count": int(pd.Series(event_id).nunique()),
    }


def summarize_replicates(rows: list[dict[str, float]], keys: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    group_cols = [c for c in ["analysis", "model", "model_label", "calibration_budget", "interval_level"] if c in df.columns]
    agg = {}
    for k in keys:
        agg[f"{k}_median"] = (k, "median")
        agg[f"{k}_p05"] = (k, lambda s: float(np.quantile(s, 0.05)))
        agg[f"{k}_p95"] = (k, lambda s: float(np.quantile(s, 0.95)))
    agg["replicate_count"] = ("rep", "count")
    out = df.groupby(group_cols, observed=True).agg(**agg).reset_index()
    if "beta_false_safe_cons_median" in out.columns:
        out["gate"] = np.where(out["beta_false_safe_cons_median"] >= BETA_TARGET, "Pass", "Fail")
    return out


def make_label_budget(trace: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    base_rng = np.random.default_rng(20260601)
    for model in MODELS:
        mdf = trace[trace["model"] == model]
        for rep, rdf in mdf.groupby("rep", observed=True):
            calib = rdf[rdf["split"] == "calibration"]["residual_abs_log"].to_numpy(dtype=float)
            test = rdf[rdf["split"] == "test"]
            if len(calib) == 0 or len(test) == 0:
                continue
            for n_budget in N_BUDGETS:
                if n_budget > len(calib):
                    continue
                draw_count = 1 if n_budget == len(calib) else 24
                for draw in range(draw_count):
                    seed = int(base_rng.integers(0, 2**31 - 1))
                    rng = np.random.default_rng(seed)
                    if n_budget == len(calib):
                        sample = calib
                    else:
                        sample = rng.choice(calib, size=n_budget, replace=False)
                    q_value = conformal_quantile(sample, 0.90)
                    met = metrics_for(test, q_value, rng)
                    met.update({
                        "analysis": "calibration-label budget",
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "rep": int(rep),
                        "draw": int(draw),
                        "calibration_budget": int(n_budget),
                        "interval_level": 0.90,
                    })
                    rows.append(met)
    detail = pd.DataFrame(rows)
    summary = summarize_replicates(
        rows,
        ["beta_false_safe_cons", "p_false_safe", "p_false_unsafe", "rmse_log", "q_value_log"],
    )
    return detail, summary


def make_interval_sensitivity(trace: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    base_rng = np.random.default_rng(20260602)
    for model in MODELS:
        mdf = trace[trace["model"] == model]
        for rep, rdf in mdf.groupby("rep", observed=True):
            calib = rdf[rdf["split"] == "calibration"]["residual_abs_log"].to_numpy(dtype=float)
            test = rdf[rdf["split"] == "test"]
            if len(calib) == 0 or len(test) == 0:
                continue
            for level in INTERVAL_LEVELS:
                rng = np.random.default_rng(int(base_rng.integers(0, 2**31 - 1)))
                q_value = conformal_quantile(calib, level)
                met = metrics_for(test, q_value, rng)
                met.update({
                    "analysis": "interval widening",
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "rep": int(rep),
                    "draw": 0,
                    "calibration_budget": int(len(calib)),
                    "interval_level": float(level),
                })
                rows.append(met)
    detail = pd.DataFrame(rows)
    summary = summarize_replicates(
        rows,
        ["beta_false_safe_cons", "p_false_safe", "p_false_unsafe", "q_value_log"],
    )
    pass_rates = detail.assign(pass_gate=detail["beta_false_safe_cons"] >= BETA_TARGET).groupby(
        ["analysis", "model", "model_label", "calibration_budget", "interval_level"], observed=True
    )["pass_gate"].mean().reset_index(name="gate_pass_replicate_rate")
    summary = summary.merge(pass_rates, on=["analysis", "model", "model_label", "calibration_budget", "interval_level"], how="left")
    return detail, summary


def write_latex_tables(label_summary: pd.DataFrame, interval_summary: pd.DataFrame) -> None:
    LATEX.mkdir(parents=True, exist_ok=True)
    label_rows = []
    for _, r in label_summary.sort_values(["model_label", "calibration_budget"]).iterrows():
        label_rows.append(
            f"{r['model_label']} & {int(r['calibration_budget'])} & "
            f"{r['beta_false_safe_cons_median']:.2f} & "
            f"{100*r['p_false_safe_median']:.2f} & "
            f"{100*r['p_false_unsafe_median']:.1f} & "
            f"{r['gate']} \\\\"
        )
    label_table = """\\begin{table}[t]
\\caption{Calibration-label-budget stability of the false-safe gate at the 1\\% IDR threshold. Values are medians across event-disjoint replicate stress tests under fixed trained predictions; the 90\\% split-conformal interval is recalibrated with each calibration budget.}
\\label{tab:label_budget_gate_stability}
\\centering
\\small
\\begin{tabular}{lrrrrl}
\\toprule
Model & Budget & $\\beta_{FS,cons}$ & $P_{FS}$ (\\%) & $P_{FU}$ (\\%) & Gate \\\\
\\midrule
""" + "\n".join(label_rows) + """
\\bottomrule
\\end{tabular}
\\end{table}
"""
    (LATEX / "table7_label_budget_gate_stability.tex").write_text(label_table, encoding="utf-8")

    interval_rows = []
    for _, r in interval_summary.sort_values(["model_label", "interval_level"]).iterrows():
        interval_rows.append(
            f"{r['model_label']} & {r['interval_level']:.3f} & "
            f"{r['q_value_log_median']:.3f} & "
            f"{r['beta_false_safe_cons_median']:.2f} & "
            f"{100*r['p_false_safe_median']:.2f} & "
            f"{100*r['p_false_unsafe_median']:.1f} & "
            f"{r['gate']} \\\\"
        )
    interval_table = """\\begin{table}[t]
\\caption{Interval-widening sensitivity of gate decisions at the 1\\% IDR threshold. Wider conformal intervals reduce false-safe risk but increase the false-unsafe screening burden.}
\\label{tab:interval_widening_sensitivity}
\\centering
\\small
\\begin{tabular}{lrrrrrl}
\\toprule
Model & Level & $q$ & $\\beta_{FS,cons}$ & $P_{FS}$ (\\%) & $P_{FU}$ (\\%) & Gate \\\\
\\midrule
""" + "\n".join(interval_rows) + """
\\bottomrule
\\end{tabular}
\\end{table}
"""
    (LATEX / "table8_interval_widening_sensitivity.tex").write_text(interval_table, encoding="utf-8")


def draw_figure(label_summary: pd.DataFrame, interval_summary: pd.DataFrame) -> None:
    FIGOUT.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig = plt.figure(figsize=(7.4, 6.7), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, left=0.08, right=0.98, bottom=0.12, top=0.92, wspace=0.38, hspace=0.42)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    for model in MODELS:
        s = label_summary[label_summary["model"] == model].sort_values("calibration_budget")
        x = s["calibration_budget"].to_numpy(dtype=float)
        y = s["beta_false_safe_cons_median"].to_numpy(dtype=float)
        lo = s["beta_false_safe_cons_p05"].to_numpy(dtype=float)
        hi = s["beta_false_safe_cons_p95"].to_numpy(dtype=float)
        ax1.plot(x, y, marker="o", lw=1.8, color=COLORS[model], label=MODEL_LABELS[model])
        ax1.fill_between(x, lo, hi, color=COLORS[model], alpha=0.14, linewidth=0)
        ax2.plot(x, 100 * s["p_false_unsafe_median"], marker="s", lw=1.6, color=COLORS[model])

    ax1.axhline(BETA_TARGET, color="#222222", lw=1.0, ls="--")
    ax1.text(55, BETA_TARGET + 0.05, r"gate target $\beta^*=2.5$", fontsize=7.5, color="#222222")
    ax1.set_title("a  False-safe gate stability")
    ax1.set_xlabel("Calibration labels used for interval recalibration")
    ax1.set_ylabel(r"Conservative gate index, $\beta_{FS,cons}$")
    ax1.set_xticks(N_BUDGETS)
    ax1.set_xlim(40, 260)
    ax1.grid(True, color="#e6e6e6", linewidth=0.6)
    ax1.legend(frameon=False, ncol=1, loc="lower right")

    ax2.set_title("b  False-unsafe burden")
    ax2.set_xlabel("Calibration labels used for interval recalibration")
    ax2.set_ylabel(r"Median $P_{FU}$ (%)")
    ax2.set_xticks(N_BUDGETS)
    ax2.set_xlim(40, 260)
    ax2.grid(True, color="#e6e6e6", linewidth=0.6)

    heat = label_summary.pivot(index="model_label", columns="calibration_budget", values="beta_false_safe_cons_median")
    order = [MODEL_LABELS[m] for m in MODELS]
    heat = heat.loc[order, N_BUDGETS]
    im = ax3.imshow(heat.to_numpy(), cmap="RdYlBu", vmin=1.7, vmax=3.2, aspect="auto")
    ax3.set_title("c  Gate eligibility map")
    ax3.set_xticks(range(len(N_BUDGETS)), [str(n) for n in N_BUDGETS])
    ax3.set_yticks(range(len(order)), order)
    ax3.set_xlabel("Calibration budget")
    for i, label in enumerate(order):
        for j, n in enumerate(N_BUDGETS):
            val = heat.loc[label, n]
            txt = "P" if val >= BETA_TARGET else "F"
            ax3.text(j, i, f"{val:.2f}\n{txt}", ha="center", va="center", fontsize=7.2, color="#111111")
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.03)

    for model in MODELS:
        s = interval_summary[interval_summary["model"] == model].sort_values("interval_level")
        ax4.plot(s["interval_level"], s["beta_false_safe_cons_median"], marker="o", lw=1.8, color=COLORS[model], label=MODEL_LABELS[model])
    ax4.axhline(BETA_TARGET, color="#222222", lw=1.0, ls="--")
    ax4.set_title("d  Interval-widening sensitivity")
    ax4.set_xlabel("Conformal interval level")
    ax4.set_ylabel(r"Median $\beta_{FS,cons}$")
    ax4.set_xticks(INTERVAL_LEVELS, ["0.90", "0.95", "0.975"])
    ax4.grid(True, color="#e6e6e6", linewidth=0.6)

    fig.suptitle("Event-disjoint false-safe gate stability at the 1% IDR threshold", x=0.08, ha="left", fontsize=11, fontweight="bold")
    base = FIGOUT / "fig_r27_label_budget_interval_sensitivity"
    for ext, dpi in [("pdf", None), ("svg", None), ("png", 450), ("tiff", 600)]:
        fig.savefig(base.with_suffix(f".{ext}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    for ext in ["pdf", "png", "svg"]:
        (LATEX / f"Figure_7.{ext}").write_bytes(base.with_suffix(f".{ext}").read_bytes())


def write_report(label_summary: pd.DataFrame, interval_summary: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = [
        "# R27 gate stability and interval sensitivity",
        "",
        "Date: 2026-06-01",
        "",
        "This round addresses the main external critique after R26: the manuscript had a clear false-safe gate but needed stability evidence.",
        "",
        "## Evidence boundary",
        "",
        "- The analysis uses the existing event-disjoint N=500 residual trace.",
        "- Calibration-label budgets Calibration-label budgets 50/100/250 are stress tests obtained by recalibrating the interval from subsets of the available calibration residuals under fixed trained predictions.",
        "- N=500 as a fresh calibration split and N=1000/2000 are not imputed because no corresponding event-disjoint prediction trace exists in the current project outputs.",
        "- Interval widening is evaluated at conformal levels 0.90/0.95/0.975.",
        "",
        "## Main conclusions",
        "",
    ]
    best = label_summary.sort_values("beta_false_safe_cons_median", ascending=False).head(4)
    for _, r in best.iterrows():
        lines.append(f"- {r['model_label']} at N={int(r['calibration_budget'])}: beta_FS_cons={r['beta_false_safe_cons_median']:.2f}, gate={r['gate']}.")
    lines += ["", "## Interval sensitivity summary", ""]
    for _, r in interval_summary.sort_values(["model_label", "interval_level"]).iterrows():
        lines.append(f"- {r['model_label']} level {r['interval_level']:.3f}: beta_FS_cons={r['beta_false_safe_cons_median']:.2f}, PFU={100*r['p_false_unsafe_median']:.1f}%, gate={r['gate']}.")
    (OUT / "R27_GATE_STABILITY_SENSITIVITY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not TRACE.exists():
        raise FileNotFoundError(TRACE)
    OUT.mkdir(parents=True, exist_ok=True)
    FIGOUT.mkdir(parents=True, exist_ok=True)
    trace = pd.read_csv(TRACE)
    if RECORDS.exists() and "event_id" not in trace.columns:
        records = pd.read_csv(RECORDS)[["gm_id", "event_id"]].drop_duplicates("gm_id")
        trace = trace.merge(records, on="gm_id", how="left")
    trace = trace[trace["model"].isin(MODELS)].copy()
    label_detail, label_summary = make_label_budget(trace)
    interval_detail, interval_summary = make_interval_sensitivity(trace)
    label_detail.to_csv(OUT / "label_budget_gate_stability_detail.csv", index=False)
    label_summary.to_csv(OUT / "label_budget_gate_stability_summary.csv", index=False)
    interval_detail.to_csv(OUT / "interval_widening_sensitivity_detail.csv", index=False)
    interval_summary.to_csv(OUT / "interval_widening_sensitivity_summary.csv", index=False)
    write_latex_tables(label_summary, interval_summary)
    draw_figure(label_summary, interval_summary)
    write_report(label_summary, interval_summary)
    print("Wrote R27 gate stability outputs to", OUT)
    print("Wrote Figure_7 and tables to", LATEX)


if __name__ == "__main__":
    main()




