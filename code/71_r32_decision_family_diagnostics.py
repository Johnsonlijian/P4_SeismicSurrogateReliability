from pathlib import Path
import shutil
import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import ticker

ROOT = Path(__file__).resolve().parents[1]
DECISION_DETAIL = ROOT / "outputs/high_target/r22_decision_risk/decision_risk_sensitivity_detail.csv"
RELIABILITY = ROOT / "outputs/high_target/r24_structural_safety_tables/false_safe_reliability_detail.csv"
PRED = ROOT / "outputs/high_target/r28_event_disjoint_large_budget/event_disjoint_large_budget_predictions.csv"
REC = ROOT / "outputs/high_target/recorded_nsmp_full/nsmp_recorded_records.csv"

OUT = ROOT / "outputs/high_target/r32_decision_family_diagnostics"
FIG = ROOT / "outputs/figures/high_target"
SUB = ROOT / "submission/structural_safety_2026-06-01"
SUPP = SUB / "supplementary_R32"
for d in [OUT, FIG, SUPP]:
    d.mkdir(parents=True, exist_ok=True)

BETA_TARGET = 2.5
THRESHOLD_STRESS = 0.01
MODEL_ORDER = ["Ridge direct", "LGBM direct", "XGB direct", "MLP scratch"]
MODEL_COLORS = {
    "Ridge direct": "#2F5C8A",
    "HGB direct": "#8C6D31",
    "RF direct": "#4C78A8",
    "SVR direct": "#B279A2",
    "LGBM direct": "#1B8A5A",
    "XGB direct": "#C47A1B",
    "MLP scratch": "#7B4FA3",
    "MLP transfer": "#D45087",
    "NTHA fallback": "#6F6F6F",
}
ACTION_SHORT = {
    "Ridge direct": "Ridge",
    "HGB direct": "HGB",
    "RF direct": "RF",
    "SVR direct": "SVR",
    "LGBM direct": "LGBM",
    "XGB direct": "XGB",
    "MLP scratch": "MLP",
    "MLP transfer": "MLP-T",
    "NTHA fallback": "NTHA",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 10,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "figure.titlesize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.linewidth": 0.8,
})
ND = NormalDist()

def beta_from_p(p):
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return -ND.inv_cdf(p)


def save_figure(fig, stem):
    paths = []
    for ext in ["pdf", "svg", "png", "tiff"]:
        path = FIG / f"{stem}.{ext}"
        kwargs = {"bbox_inches": "tight"}
        if ext in ["png", "tiff"]:
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
        paths.append(path)
    return paths


def decision_map():
    dec = pd.read_csv(DECISION_DETAIL)
    rel = pd.read_csv(RELIABILITY)
    dec = dec[dec["protocol"].eq("event-disjoint target")].copy()
    rel = rel[rel["protocol"].eq("event-disjoint target")].copy()
    rel = rel[["model_label", "threshold_idr", "beta_false_safe_cons", "false_safe_ci95_hi", "false_unsafe_rate"]]
    df = dec.merge(rel, on=["model_label", "threshold_idr"], how="left", suffixes=("", "_rel"), validate="many_to_one")
    df["eligible"] = df["beta_false_safe_cons"] >= BETA_TARGET
    # Approximate downstream workload for an interval threshold screen: true positives are routed onward,
    # false-unsafe cases are routed onward unnecessarily, and false-safe cases are missed.
    df["fallback_workload"] = (df["true_exceed_rate"] - df["false_safe_rate"] + df["false_unsafe_rate"]).clip(0, 1)

    rows = []
    for (tau, cost), g in df.groupby(["threshold_idr", "cost_ratio_false_safe"], sort=True):
        eligible = g[g["eligible"]].copy()
        if eligible.empty:
            rows.append({
                "threshold_idr": tau,
                "cost_ratio_false_safe": cost,
                "selected_action": "NTHA fallback",
                "selected_model": "NTHA fallback",
                "expected_loss": np.nan,
                "fallback_workload": 1.0,
                "beta_false_safe_cons": np.nan,
                "eligible_model_count": 0,
            })
        else:
            best = eligible.sort_values(["expected_loss", "fallback_workload", "model_label"]).iloc[0]
            rows.append({
                "threshold_idr": tau,
                "cost_ratio_false_safe": cost,
                "selected_action": best["model_label"],
                "selected_model": best["model_label"],
                "expected_loss": best["expected_loss"],
                "fallback_workload": best["fallback_workload"],
                "beta_false_safe_cons": best["beta_false_safe_cons"],
                "eligible_model_count": int(eligible["model_label"].nunique()),
            })
    selected = pd.DataFrame(rows)
    selected.to_csv(OUT / "r32_decision_impact_selected_actions.csv", index=False)
    df.to_csv(OUT / "r32_decision_impact_model_detail.csv", index=False)

    taus = sorted(selected["threshold_idr"].unique())
    costs = sorted(selected["cost_ratio_false_safe"].unique())
    preferred_actions = ["Ridge direct", "HGB direct", "RF direct", "SVR direct", "LGBM direct", "XGB direct", "MLP scratch", "MLP transfer"]
    observed_actions = [a for a in preferred_actions if a in set(selected["selected_action"])]
    extra_actions = sorted(set(selected["selected_action"]) - set(observed_actions) - {"NTHA fallback"})
    actions = observed_actions + extra_actions + ["NTHA fallback"]
    action_to_id = {a: i for i, a in enumerate(actions)}
    mat = np.full((len(costs), len(taus)), np.nan)
    work = np.full_like(mat, np.nan, dtype=float)
    for _, r in selected.iterrows():
        i = costs.index(r["cost_ratio_false_safe"])
        j = taus.index(r["threshold_idr"])
        mat[i, j] = action_to_id[r["selected_action"]]
        work[i, j] = r["fallback_workload"]

    cmap = ListedColormap([MODEL_COLORS[a] for a in actions])
    norm = BoundaryNorm(np.arange(-0.5, len(actions) + 0.5, 1), cmap.N)

    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    fig.subplots_adjust(left=0.12, right=0.985, top=0.78, bottom=0.24)
    im = ax.imshow(mat, origin="lower", cmap=cmap, norm=norm, aspect="auto")
    xx, yy = np.meshgrid(np.arange(len(taus)), np.arange(len(costs)))
    levels = [0.05, 0.10, 0.20, 0.40, 0.70, 0.95]
    valid_levels = [lv for lv in levels if np.nanmin(work) <= lv <= np.nanmax(work)]
    if valid_levels:
        cs = ax.contour(xx, yy, work, levels=valid_levels, colors="white", linewidths=1.0, alpha=0.9)
        ax.clabel(cs, fmt=lambda x: f"{x:.0%}", fontsize=7, inline=True, colors="white")

    for i, cost in enumerate(costs):
        for j, tau in enumerate(taus):
            row = selected[(selected["cost_ratio_false_safe"].eq(cost)) & (selected["threshold_idr"].eq(tau))].iloc[0]
            label = f"{ACTION_SHORT.get(row['selected_action'], row['selected_action'])}\n{row['fallback_workload']:.0%}"
            ax.text(j, i, label, ha="center", va="center", color="white", fontsize=7.0, fontweight="bold")

    ax.set_xticks(np.arange(len(taus)))
    ax.set_xticklabels([f"{100*t:.1f}%" for t in taus])
    ax.set_yticks(np.arange(len(costs)))
    ax.set_yticklabels([f"{c:g}:1" for c in costs])
    ax.set_xlabel("Drift threshold used by the interval screen")
    ax.set_ylabel("False-safe cost ratio")
    fig.text(0.12, 0.95, "Reliability-efficiency decision map", ha="left", va="top", fontsize=13.5, fontweight="bold")
    fig.text(0.12, 0.885,
             "Cells select the loss-minimizing action among models satisfying beta_FS,cons >= 2.5; "
             "labels and contours show fallback workload.",
             ha="left", va="top", fontsize=8.0, color="#333333")
    handles = [plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=MODEL_COLORS.get(a, "#999999"), markersize=9, label=ACTION_SHORT.get(a, a)) for a in actions]
    ax.legend(handles=handles, ncol=min(5, len(actions)), loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    ax.set_xlim(-0.5, len(taus) - 0.5)
    ax.set_ylim(-0.5, len(costs) - 0.5)
    fig_paths = save_figure(fig, "fig_r32_decision_impact_map")
    plt.close(fig)
    return selected, fig_paths


def family_diagnostic():
    usecols = ["rep", "N", "model_label", "split", "gm_id", "n_story", "T1_s", "pattern", "y_true_log", "y_pred_log", "q_value_log", "covered"]
    pred = pd.read_csv(PRED, usecols=usecols)
    pred = pred[pred["split"].eq("test") & pred["N"].eq(2000) & pred["model_label"].isin(MODEL_ORDER)].copy()
    rec = pd.read_csv(REC, usecols=["gm_id", "event_id"])
    pred = pred.merge(rec, on="gm_id", how="left", validate="many_to_one")
    if pred["event_id"].isna().any():
        raise RuntimeError("Some gm_id values do not map to event_id in the NSMP record registry.")

    tau_log = math.log10(THRESHOLD_STRESS)
    pred["upper_log"] = pred["y_pred_log"] + pred["q_value_log"]
    pred["true_unsafe"] = pred["y_true_log"] > tau_log
    pred["pred_safe"] = pred["upper_log"] <= tau_log
    pred["false_safe"] = pred["pred_safe"] & pred["true_unsafe"]
    pred["false_unsafe"] = (~pred["pred_safe"]) & (~pred["true_unsafe"])
    pred["family"] = pred.apply(lambda r: f"{int(r['n_story'])}F {str(r['pattern']).replace('_', '-')} T={float(r['T1_s']):.1f}s", axis=1)

    event = pred.groupby(["model_label", "rep", "family", "event_id"], observed=True).agg(
        false_safe_rate=("false_safe", "mean"),
        false_unsafe_rate=("false_unsafe", "mean"),
        coverage=("covered", "mean"),
        true_exceed_rate=("true_unsafe", "mean"),
        n_cases=("false_safe", "size"),
    ).reset_index()
    repfam = event.groupby(["model_label", "rep", "family"], observed=True).agg(
        false_safe_rate=("false_safe_rate", "mean"),
        false_safe_rate_q95_event=("false_safe_rate", lambda s: float(np.quantile(s, 0.95))),
        false_unsafe_rate=("false_unsafe_rate", "mean"),
        coverage=("coverage", "mean"),
        true_exceed_rate=("true_exceed_rate", "mean"),
        event_count=("event_id", "nunique"),
    ).reset_index()
    summary = repfam.groupby(["model_label", "family"], observed=True).agg(
        false_safe_rate_median=("false_safe_rate", "median"),
        false_safe_rate_q95_rep=("false_safe_rate", lambda s: float(np.quantile(s, 0.95))),
        event_q95_false_safe_rate_median=("false_safe_rate_q95_event", "median"),
        false_unsafe_rate_median=("false_unsafe_rate", "median"),
        coverage_median=("coverage", "median"),
        true_exceed_rate_median=("true_exceed_rate", "median"),
        event_count_median=("event_count", "median"),
    ).reset_index()
    summary["beta_fs_cons_family"] = summary["false_safe_rate_q95_rep"].map(beta_from_p)
    summary["pass_beta_2p5"] = summary["beta_fs_cons_family"] >= BETA_TARGET
    summary["model_label"] = pd.Categorical(summary["model_label"], MODEL_ORDER, ordered=True)
    families = list(summary.sort_values(["family"])["family"].unique())
    summary["family"] = pd.Categorical(summary["family"], families, ordered=True)
    summary = summary.sort_values(["model_label", "family"])
    summary.to_csv(OUT / "r32_structural_family_stress_summary.csv", index=False)
    repfam.to_csv(OUT / "r32_structural_family_stress_rep_detail.csv", index=False)

    worst = summary.sort_values(["model_label", "beta_fs_cons_family"]).groupby("model_label", observed=True).head(1).copy()
    table = worst[["model_label", "family", "false_safe_rate_median", "false_safe_rate_q95_rep", "beta_fs_cons_family", "false_unsafe_rate_median", "coverage_median", "pass_beta_2p5"]]
    table.to_csv(OUT / "r32_structural_family_worst_cases.csv", index=False)
    table_tex = table.copy()
    table_tex["false_safe_rate_median"] = table_tex["false_safe_rate_median"].map(lambda x: f"{100*x:.2f}")
    table_tex["false_safe_rate_q95_rep"] = table_tex["false_safe_rate_q95_rep"].map(lambda x: f"{100*x:.2f}")
    table_tex["beta_fs_cons_family"] = table_tex["beta_fs_cons_family"].map(lambda x: f"{x:.2f}")
    table_tex["false_unsafe_rate_median"] = table_tex["false_unsafe_rate_median"].map(lambda x: f"{100*x:.1f}")
    table_tex["coverage_median"] = table_tex["coverage_median"].map(lambda x: f"{100*x:.1f}")
    table_tex["pass_beta_2p5"] = table_tex["pass_beta_2p5"].map(lambda x: "Pass" if x else "Flag")
    tex = table_tex.to_latex(index=False, escape=True, column_format="llrrrrrl")
    tex = tex.replace("model\\_label", "Model")
    tex = tex.replace("false\\_safe\\_rate\\_median", "Median FS (\\%)")
    tex = tex.replace("false\\_safe\\_rate\\_q95\\_rep", "Replicate q95 FS (\\%)")
    tex = tex.replace("beta\\_fs\\_cons\\_family", "$\\beta_{\\mathrm{FS,cons}}$")
    tex = tex.replace("false\\_unsafe\\_rate\\_median", "Median FU (\\%)")
    tex = tex.replace("coverage\\_median", "Coverage (\\%)")
    tex = tex.replace("pass\\_beta\\_2p5", "$\\beta\\ge2.5$")
    (OUT / "table_r32_structural_family_worst.tex").write_text(tex, encoding="utf-8")

    pivot_beta = summary.pivot(index="model_label", columns="family", values="beta_fs_cons_family").reindex(MODEL_ORDER)
    pivot_pass = summary.pivot(index="model_label", columns="family", values="pass_beta_2p5").reindex(MODEL_ORDER)
    pivot_fu = summary.pivot(index="model_label", columns="family", values="false_unsafe_rate_median").reindex(MODEL_ORDER)
    display_families = [
        f.replace(" soft-first-story ", "\nsoft\n").replace(" uniform ", "\nuniform\n")
        for f in families
    ]
    fig = plt.figure(figsize=(8.8, 6.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0], left=0.14, right=0.94, top=0.80, bottom=0.20, hspace=0.62)
    ax0 = fig.add_subplot(gs[0])
    data = pivot_beta.to_numpy(dtype=float)
    vmax = max(3.5, np.nanmax(data))
    im = ax0.imshow(data, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
    cbar = fig.colorbar(im, ax=ax0, fraction=0.026, pad=0.012)
    cbar.set_label(r"Family-stratified $\beta_{\mathrm{FS,cons}}$")
    ax0.set_xticks(np.arange(len(families)))
    ax0.set_xticklabels(display_families, rotation=0, ha="center")
    ax0.set_yticks(np.arange(len(MODEL_ORDER)))
    ax0.set_yticklabels(MODEL_ORDER)
    fig.text(0.14, 0.96, "Structural-family stratified stress diagnostic at 1.0% drift",
             ha="left", va="top", fontsize=13.2, fontweight="bold")
    fig.text(0.14, 0.905,
             "True R28 event-disjoint N=2000 predictions are stratified by family; this is not leave-family-out retraining.",
             ha="left", va="top", fontsize=8.0, color="#333333")
    for i in range(len(MODEL_ORDER)):
        for j in range(len(families)):
            val = data[i, j]
            ok = bool(pivot_pass.iloc[i, j])
            text = f"{val:.2f}\n{'P' if ok else 'F'}"
            ax0.text(j, i, text, ha="center", va="center", color="white" if val < 2.0 else "black", fontsize=6.2, fontweight="bold")
    ax1 = fig.add_subplot(gs[1])
    x = np.arange(len(families))
    for model in MODEL_ORDER:
        y = pivot_fu.loc[model].to_numpy(dtype=float) * 100.0
        ax1.plot(x, y, marker="o", lw=1.6, ms=3.5, color=MODEL_COLORS[model], label=model)
    ax1.set_xticks(x)
    ax1.set_xticklabels(display_families, rotation=0, ha="center")
    ax1.set_ylabel("Median false-unsafe workload (%)")
    ax1.set_xlabel("")
    ax1.grid(True, axis="y", color="#D7D7D7", lw=0.6)
    ax1.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.24), frameon=False)
    fig_paths = save_figure(fig, "fig_r32_structural_family_stress")
    plt.close(fig)
    return summary, repfam, table, fig_paths


def write_report(decision_selected, family_summary, family_table):
    report = f"""# R32 decision and structural-family diagnostic addendum

## Scope and evidence boundary

This addendum uses existing, already generated evidence from the R22/R24 decision-risk and false-safe reliability tables and the R28 true event-disjoint N=2000 prediction export. It does not introduce simulated placeholder values, fabricated observations, or AI-generated numerical results.

The structural-family analysis is a stratified diagnostic within the event-disjoint N=2000 stress test. It is not a true leave-family-out retraining experiment and should not be described as proof of extrapolation to unseen building inventories.

## Outputs

- `outputs/figures/high_target/fig_r32_decision_impact_map.*`
- `outputs/figures/high_target/fig_r32_structural_family_stress.*`
- `outputs/high_target/r32_decision_family_diagnostics/r32_decision_impact_selected_actions.csv`
- `outputs/high_target/r32_decision_family_diagnostics/r32_structural_family_stress_summary.csv`
- `submission/structural_safety_2026-06-01/supplementary_R32/Supplementary_R32_decision_family_diagnostics.pdf`

## Decision-map interpretation

The decision map selects the loss-minimizing action among models satisfying the conservative false-safe reliability threshold beta_FS,cons >= {BETA_TARGET}. If no surrogate satisfies the diagnostic filter for a threshold-cost cell, the action is shown as an NTHA fallback. The overlaid contours and cell percentages report the fraction of cases routed to downstream nonlinear analysis rather than treated as screened-safe.

## Structural-family diagnostic interpretation

The family diagnostic reports family-stratified conservative false-safe reliability and false-unsafe workload at a {100*THRESHOLD_STRESS:.1f}% drift threshold. The table below lists the weakest family for each evaluated model.

"""
    report += family_table.to_markdown(index=False)
    report += """

## Recommended manuscript use

Use these outputs as supplementary material or as a concise addendum during revision. Avoid inserting them into the already clean R31 main text unless a target-specific reason emerges, because the current main manuscript is internally stable and these diagnostics mainly strengthen reviewer reassurance rather than change the central claim.
"""
    (OUT / "R32_DECISION_FAMILY_DIAGNOSTICS_REPORT.md").write_text(report, encoding="utf-8")
    shutil.copy2(OUT / "R32_DECISION_FAMILY_DIAGNOSTICS_REPORT.md", SUB / "STRUCTURAL_SAFETY_R32_SUPPLEMENTARY_QA_REPORT.md")


def write_supplement():
    for stem in ["fig_r32_decision_impact_map", "fig_r32_structural_family_stress"]:
        shutil.copy2(FIG / f"{stem}.pdf", SUPP / f"{stem}.pdf")
        shutil.copy2(FIG / f"{stem}.png", SUPP / f"{stem}.png")
    shutil.copy2(OUT / "table_r32_structural_family_worst.tex", SUPP / "table_r32_structural_family_worst.tex")
    tex = r"""
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{caption}
\usepackage{amsmath}
\usepackage{siunitx}
\usepackage{hyperref}
\hypersetup{colorlinks=true, linkcolor=black, urlcolor=blue, citecolor=black}
\title{Supplementary R32: reliability--efficiency decision map and structural-family stratified diagnostic}
\author{Lijian REN}
\date{June 2, 2026}
\begin{document}
\maketitle

\section*{Scope}
This supplementary addendum supports the Structural Safety submission package by adding two final diagnostic views. The first maps reliability-constrained decision actions across drift thresholds and false-safe cost ratios. The second stratifies the already generated event-disjoint $N=2000$ prediction export by structural family at the 1.0\% drift threshold.

All numerical values are derived from the existing R22/R24 decision-risk and false-safe reliability tables and the R28 event-disjoint prediction export. The structural-family analysis is a stratified stress diagnostic, not a leave-family-out retraining experiment and not a claim of general inventory-level extrapolation.

\section*{Reliability--efficiency decision map}
Figure~\ref{fig:decision-map} selects the loss-minimizing action among models satisfying the conservative false-safe reliability criterion $\beta_{\mathrm{FS,cons}}\ge 2.5$. When no surrogate satisfies this diagnostic filter, the cell is assigned to nonlinear time-history analysis (NTHA) fallback. The contour lines and cell percentages report the approximate downstream workload routed onward rather than screened-safe.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.98\linewidth]{fig_r32_decision_impact_map.pdf}
\caption{Reliability--efficiency decision map over drift threshold and false-safe cost ratio. Colors denote the selected reliability-eligible action; contours and labels report fallback workload.}
\label{fig:decision-map}
\end{figure}

\section*{Structural-family stratified diagnostic}
Figure~\ref{fig:family-diagnostic} reports family-stratified conservative false-safe reliability and false-unsafe workload for the event-disjoint $N=2000$ stress test at a 1.0\% drift threshold. The panel is diagnostic: it identifies where the screening rule is most fragile within the tested structural families, but it should not be read as a leave-family-out or unseen-inventory generalization result.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.98\linewidth]{fig_r32_structural_family_stress.pdf}
\caption{Structural-family stratified stress diagnostic based on the true R28 event-disjoint $N=2000$ prediction export. P and F indicate whether the family-level conservative false-safe reliability index passes or flags the illustrative $\beta_{\mathrm{FS,cons}}\ge 2.5$ criterion.}
\label{fig:family-diagnostic}
\end{figure}

\section*{Weakest-family summary}
Table~\ref{tab:weakest-family} lists the weakest structural family for each evaluated model according to the family-stratified conservative false-safe reliability index.

\begin{table}[htbp]
\centering
\caption{Weakest structural family for each model under the R32 stratified diagnostic. FS = false-safe; FU = false-unsafe.}
\label{tab:weakest-family}
\resizebox{\linewidth}{!}{%
\input{table_r32_structural_family_worst.tex}
}
\end{table}

\section*{Use in the manuscript}
These diagnostics are best used as supplementary support or as revision-stage evidence. They do not change the main claim of the manuscript: the proposed filter is a necessary reliability diagnostic for surrogate-based drift-threshold screening, not a sufficient safety certificate.

\end{document}
"""
    (SUPP / "Supplementary_R32_decision_family_diagnostics.tex").write_text(tex, encoding="utf-8")


def main():
    decision_selected, decision_figs = decision_map()
    family_summary, repfam, family_table, family_figs = family_diagnostic()
    write_report(decision_selected, family_summary, family_table)
    write_supplement()
    print("R32 diagnostics generated")
    print(OUT)
    print(SUPP)
    print("Decision figures:", ", ".join(str(p) for p in decision_figs))
    print("Family figures:", ", ".join(str(p) for p in family_figs))

if __name__ == "__main__":
    main()
