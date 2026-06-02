"""R30 conditional-coverage diagnostic by event-level intensity bins.

This is a descriptive supplement to the marginal split-conformal coverage claim.
It uses the R28 regenerated event-disjoint prediction export and avoids row-level
pseudo-replication by assigning each held-out event to a PGA bin and aggregating
coverage event-equal within each model/rep/bin.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(r"R:\NAS_DRIVE\IMUT\1-Research_Output\1-Papers\1_In_Preparation\2026-Claude\projects\P4_SeismicFoundationModel")
PRED = ROOT / "outputs" / "high_target" / "r28_event_disjoint_large_budget" / "event_disjoint_large_budget_predictions.csv"
REC = ROOT / "outputs" / "high_target" / "recorded_nsmp_full" / "nsmp_recorded_records.csv"
OUT = ROOT / "outputs" / "high_target" / "r30_conditional_coverage_intensity"
FIG = ROOT / "outputs" / "figures" / "high_target"
LATEX = ROOT / "submission" / "structural_safety_2026-06-01" / "latex_source_flat"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

MODEL_ORDER = ["Ridge direct", "LGBM direct", "XGB direct", "MLP scratch"]
BIN_LABELS = ["low PGA", "mid PGA", "high PGA"]


def load_data() -> pd.DataFrame:
    usecols = ["rep", "N", "model_label", "split", "gm_id", "system_id", "covered"]
    pred = pd.read_csv(PRED, usecols=usecols)
    pred = pred[(pred["split"].eq("test")) & (pred["N"].eq(2000)) & (pred["model_label"].isin(MODEL_ORDER))].copy()
    rec = pd.read_csv(REC, usecols=["gm_id", "event_id", "pga_g"])
    pred = pred.merge(rec, on="gm_id", how="left", validate="many_to_one")
    if pred["event_id"].isna().any():
        raise RuntimeError("Missing event metadata after merge")
    event_pga = pred[["event_id", "pga_g"]].drop_duplicates().groupby("event_id", as_index=False)["pga_g"].median()
    event_pga["pga_bin"] = pd.qcut(event_pga["pga_g"], q=3, labels=BIN_LABELS, duplicates="drop")
    pred = pred.merge(event_pga[["event_id", "pga_bin"]], on="event_id", how="left", validate="many_to_one")
    return pred


def summarize(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_cov = (
        pred.groupby(["model_label", "rep", "pga_bin", "event_id"], observed=True)
        .agg(coverage=("covered", "mean"), rows=("covered", "size"), pga_g=("pga_g", "median"))
        .reset_index()
    )
    # Event-equal by first averaging events within each replicate/bin.
    rep_bin = (
        event_cov.groupby(["model_label", "rep", "pga_bin"], observed=True)
        .agg(event_equal_coverage=("coverage", "mean"), event_count=("event_id", "nunique"))
        .reset_index()
    )
    summary = (
        rep_bin.groupby(["model_label", "pga_bin"], observed=True)
        .agg(
            median_coverage=("event_equal_coverage", "median"),
            q05_coverage=("event_equal_coverage", lambda s: float(np.quantile(s, 0.05))),
            q95_coverage=("event_equal_coverage", lambda s: float(np.quantile(s, 0.95))),
            min_rep_coverage=("event_equal_coverage", "min"),
            median_event_count=("event_count", "median"),
        )
        .reset_index()
    )
    summary["model_label"] = pd.Categorical(summary["model_label"], MODEL_ORDER, ordered=True)
    summary["pga_bin"] = pd.Categorical(summary["pga_bin"], BIN_LABELS, ordered=True)
    summary = summary.sort_values(["model_label", "pga_bin"])
    return event_cov, summary


def write_table(summary: pd.DataFrame) -> None:
    summary.to_csv(OUT / "conditional_coverage_pga_bin_summary.csv", index=False)
    rows = []
    for _, r in summary.iterrows():
        rows.append(
            f"{r['model_label']} & {r['pga_bin']} & {100*r['median_coverage']:.1f} & "
            f"[{100*r['q05_coverage']:.1f}, {100*r['q95_coverage']:.1f}] & {int(r['median_event_count'])} \\\\" 
        )
    table = "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Descriptive conditional-coverage diagnostic by event-level PGA bin at $N=2000$. Coverage is first averaged within held-out earthquake events and then summarized over replicate event-disjoint runs; the diagnostic does not replace the marginal split-conformal guarantee.}",
        r"\label{tab:conditional_coverage_pga}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Model & Event PGA bin & Median coverage (\%) & 5--95\% over reps & Events/bin \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    (LATEX / "table9_conditional_coverage_pga.tex").write_text(table, encoding="utf-8")


def plot(summary: pd.DataFrame) -> None:
    mat = summary.pivot(index="model_label", columns="pga_bin", values="median_coverage").reindex(index=MODEL_ORDER, columns=BIN_LABELS)
    q05 = summary.pivot(index="model_label", columns="pga_bin", values="q05_coverage").reindex(index=MODEL_ORDER, columns=BIN_LABELS)
    q95 = summary.pivot(index="model_label", columns="pga_bin", values="q95_coverage").reindex(index=MODEL_ORDER, columns=BIN_LABELS)
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    im = ax.imshow(mat.values, vmin=0.70, vmax=1.00, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(BIN_LABELS)), BIN_LABELS)
    ax.set_yticks(range(len(MODEL_ORDER)), MODEL_ORDER)
    ax.set_title("Event-equal coverage by held-out event intensity", loc="left", fontsize=11, fontweight="bold")
    ax.text(-0.48, -0.87, "N=2000 event-disjoint test split; bins use event-median PGA", fontsize=8, color="#4B5563")
    for i, model in enumerate(MODEL_ORDER):
        for j, b in enumerate(BIN_LABELS):
            v = mat.loc[model, b]
            lo = q05.loc[model, b]
            hi = q95.loc[model, b]
            color = "white" if v < 0.83 else "#111827"
            ax.text(j, i, f"{100*v:.0f}\n[{100*lo:.0f},{100*hi:.0f}]", ha="center", va="center", color=color, fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Median event-equal coverage")
    ax.set_xlabel("Held-out event PGA bin")
    ax.set_ylabel("Surrogate")
    fig.text(0.12, 0.02, "Numbers show median coverage and 5--95% replicate interval; this is a diagnostic, not a conditional guarantee.", fontsize=7.5, color="#4B5563")
    base = FIG / "fig_r30_conditional_coverage_pga_bins"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=500, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    for ext in ["pdf", "svg", "png"]:
        src = base.with_suffix(f".{ext}")
        if src.exists():
            (LATEX / f"Figure_8.{ext}").write_bytes(src.read_bytes())


def write_report(summary: pd.DataFrame) -> None:
    worst = summary.loc[summary["median_coverage"].idxmin()]
    best = summary.loc[summary["median_coverage"].idxmax()]
    lines = [
        "# R30 conditional-coverage diagnostic by event-level PGA bin",
        "",
        "This diagnostic uses the R28 true event-disjoint N=2000 prediction export and the test split only.",
        "Coverage is aggregated event-equal within each event-level PGA bin to avoid row-level pseudo-replication.",
        "",
        f"- Worst median bin coverage: {worst['model_label']} / {worst['pga_bin']} = {100*worst['median_coverage']:.1f}%.",
        f"- Highest median bin coverage: {best['model_label']} / {best['pga_bin']} = {100*best['median_coverage']:.1f}%.",
        "- The result is a descriptive stress check, not a formal conditional conformal guarantee.",
        "",
        "## Summary table",
        "",
        summary.to_markdown(index=False),
    ]
    (OUT / "R30_CONDITIONAL_COVERAGE_PGA_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    pred = load_data()
    event_cov, summary = summarize(pred)
    event_cov.to_csv(OUT / "conditional_coverage_event_detail.csv", index=False)
    write_table(summary)
    plot(summary)
    write_report(summary)
    print(summary.to_string(index=False))
    print(f"[69] wrote conditional coverage outputs to {OUT}")


if __name__ == "__main__":
    main()
