"""R22 decision-risk sensitivity figure from full residual exports.

The figure evaluates an interval-based engineering screening rule:

safe if upper conformal bound <= drift threshold
unsafe/flagged otherwise

False-safe errors (actual exceedance but predicted safe) are weighted by a cost
ratio C. False-unsafe errors are weighted by 1. Metrics are event-equal to avoid
letting earthquake events with more components dominate the conclusion.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "outputs" / "high_target" / "r21_full_residual_trace" / "residual_trace_samples.csv"
EVENT = (
    ROOT
    / "outputs"
    / "high_target"
    / "event_disjoint_conformal_stress"
    / "event_disjoint_residual_samples.csv"
)
RECORDS = ROOT / "outputs" / "high_target" / "recorded_nsmp_full" / "nsmp_recorded_records.csv"
OUT = ROOT / "outputs" / "high_target" / "r22_decision_risk"
FIG_DIR = ROOT / "outputs" / "figures" / "high_target"
ROUND = ROOT / "rounds" / "R22_submission_integration_2026-06-01"
DOCS = ROOT / "docs"

N_TARGET = 500
THRESHOLDS = np.array([0.005, 0.0075, 0.010, 0.015, 0.020, 0.030, 0.040])
COST_RATIOS = np.array([1, 2, 5, 10, 25, 50, 100])

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

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 6.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def load() -> pd.DataFrame:
    frames = []
    for protocol, path in [
        ("main event-held-out", MAIN),
        ("event-disjoint target", EVENT),
    ]:
        df = pd.read_csv(path)
        df["protocol"] = protocol
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out[(out["N"].eq(N_TARGET)) & (out["split"].eq("test"))].copy()
    if RECORDS.exists():
        records = pd.read_csv(RECORDS)[["gm_id", "event_id", "event_title", "magnitude"]].drop_duplicates("gm_id")
    else:
        records = (
            out[["gm_id"]]
            .drop_duplicates("gm_id")
            .assign(event_id=lambda x: x["gm_id"], event_title=lambda x: "derived_event_" + x["gm_id"].astype(str), magnitude=np.nan)
        )
    out = out.merge(records, on="gm_id", how="left")
    out["upper_log"] = out["y_pred_log"].to_numpy(float) + out["q_value_log"].to_numpy(float)
    return out


def event_equal_rates(g: pd.DataFrame, threshold: float, cost_ratio: int) -> dict[str, float]:
    log_thr = np.log10(threshold)
    event_rows = []
    for event_id, e in g.groupby("event_id"):
        truth_unsafe = e["y_true_log"].to_numpy(float) > log_thr
        predicted_safe = e["upper_log"].to_numpy(float) <= log_thr
        false_safe = truth_unsafe & predicted_safe
        false_unsafe = (~truth_unsafe) & (~predicted_safe)
        event_rows.append(
            {
                "event_id": event_id,
                "true_exceed_rate": float(np.mean(truth_unsafe)),
                "false_safe_rate": float(np.mean(false_safe)),
                "false_unsafe_rate": float(np.mean(false_unsafe)),
                "expected_loss": float(cost_ratio * np.mean(false_safe) + np.mean(false_unsafe)),
            }
        )
    ev = pd.DataFrame(event_rows)
    return {
        "event_count": int(ev["event_id"].nunique()),
        "true_exceed_rate": float(ev["true_exceed_rate"].mean()),
        "false_safe_rate": float(ev["false_safe_rate"].mean()),
        "false_unsafe_rate": float(ev["false_unsafe_rate"].mean()),
        "expected_loss": float(ev["expected_loss"].mean()),
    }


def compute(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (protocol, model), g in samples.groupby(["protocol", "model"]):
        for thr in THRESHOLDS:
            for cost in COST_RATIOS:
                rates = event_equal_rates(g, float(thr), int(cost))
                rows.append(
                    {
                        "protocol": protocol,
                        "model": model,
                        "model_label": MODEL_LABELS.get(model, model),
                        "threshold_idr": float(thr),
                        "cost_ratio_false_safe": int(cost),
                        **rates,
                    }
                )
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "decision_risk_sensitivity_detail.csv", index=False)

    winners = (
        detail.sort_values(["protocol", "threshold_idr", "cost_ratio_false_safe", "expected_loss", "false_safe_rate"])
        .groupby(["protocol", "threshold_idr", "cost_ratio_false_safe"])
        .head(1)
        .reset_index(drop=True)
    )
    winners.to_csv(OUT / "decision_risk_sensitivity_winners.csv", index=False)
    return detail, winners


def draw(winners: pd.DataFrame, detail: pd.DataFrame) -> None:
    protocols = ["main event-held-out", "event-disjoint target"]
    fig = plt.figure(figsize=(7.2, 5.7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.78], hspace=0.48, wspace=0.34)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    line_ax = fig.add_subplot(gs[1, :])

    vmin = float(winners["expected_loss"].min())
    vmax = float(np.quantile(winners["expected_loss"], 0.92))
    for ax, protocol in zip(axes, protocols):
        sub = winners[winners["protocol"].eq(protocol)].copy()
        mat = sub.pivot(index="cost_ratio_false_safe", columns="threshold_idr", values="expected_loss").loc[COST_RATIOS, THRESHOLDS]
        im = ax.imshow(mat.to_numpy(float), cmap="YlOrRd", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(THRESHOLDS)))
        ax.set_xticklabels([f"{100*t:.2g}%" for t in THRESHOLDS], rotation=35, ha="right")
        ax.set_yticks(np.arange(len(COST_RATIOS)))
        ax.set_yticklabels([str(c) for c in COST_RATIOS])
        ax.set_xlabel("Drift threshold (IDR)")
        ax.set_ylabel("False-safe cost ratio")
        ax.set_title(protocol, loc="left", fontweight="bold")
        for i, cost in enumerate(COST_RATIOS):
            for j, thr in enumerate(THRESHOLDS):
                row = sub[(sub["cost_ratio_false_safe"].eq(cost)) & (sub["threshold_idr"].eq(thr))].iloc[0]
                color = "white" if row["expected_loss"] > (vmin + 0.62 * (vmax - vmin)) else "#1f1f1f"
                ax.text(
                    j,
                    i,
                    f"{MODEL_SHORT.get(row['model'], row['model'])}\n{row['expected_loss']:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.2,
                    color=color,
                )
    cbar = fig.colorbar(im, ax=axes, fraction=0.036, pad=0.03)
    cbar.set_label("Minimum event-equal expected loss")

    focus = detail[
        detail["model"].isin(["xgb_direct", "lgbm_direct", "pretrained_finetune", "scratch_mlp"])
        & detail["threshold_idr"].eq(0.01)
        & detail["protocol"].eq("event-disjoint target")
    ].copy()
    for model, g in focus.groupby("model"):
        g = g.sort_values("cost_ratio_false_safe")
        line_ax.plot(
            g["cost_ratio_false_safe"],
            g["expected_loss"],
            "-o",
            lw=1.4,
            ms=3.5,
            label=MODEL_SHORT.get(model, model),
        )
    line_ax.set_xscale("log")
    line_ax.set_xlabel("False-safe cost ratio at 1% IDR")
    line_ax.set_ylabel("Event-equal expected loss")
    line_ax.set_title("Cost sensitivity exposes model-dependent decision risk", loc="left")
    line_ax.grid(axis="y", color="#dddddd", lw=0.5)
    line_ax.legend(frameon=False, ncol=4, loc="upper left")

    fig.suptitle(
        "Decision-risk sensitivity: threshold x false-safe cost ratio",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=9.5,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.962,
        "Cells show the model with the lowest interval-based expected loss; false-safe errors are weighted by C, false-unsafe errors by 1.",
        ha="left",
        fontsize=6.7,
        color="#444444",
    )
    base = FIG_DIR / "fig_r22_decision_risk_sensitivity"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_report(winners: pd.DataFrame, detail: pd.DataFrame) -> None:
    lines = [
        "# R22 decision-risk sensitivity",
        "",
        "Decision rule: a case is predicted safe only when the upper conformal bound is below the drift threshold.",
        "False-safe errors are weighted by cost ratio C; false-unsafe errors have unit cost.",
        "Rates and expected losses are event-equal.",
        "",
        "## Winner changes across threshold and cost ratio",
        "",
    ]
    for protocol, g in winners.groupby("protocol"):
        counts = g["model_label"].value_counts()
        lines.append(f"### {protocol}")
        for model, n in counts.items():
            lines.append(f"- {model}: winner in {int(n)} of {len(g)} threshold-cost cells.")
        focus = g[(g["threshold_idr"].eq(0.01)) & (g["cost_ratio_false_safe"].isin([1, 10, 50, 100]))]
        lines.append("")
        lines.append("| threshold | cost ratio | winner | expected loss | false-safe | false-unsafe |")
        lines.append("| ---: | ---: | --- | ---: | ---: | ---: |")
        for _, r in focus.iterrows():
            lines.append(
                f"| {100*r['threshold_idr']:.2f}% | {int(r['cost_ratio_false_safe'])} | {r['model_label']} | "
                f"{r['expected_loss']:.3f} | {r['false_safe_rate']:.4f} | {r['false_unsafe_rate']:.4f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Manuscript-safe claim",
            "",
            "Decision-risk sensitivity is threshold- and cost-dependent. Therefore the manuscript should not report a single universally safest model. It should show a sensitivity surface and state that engineering risk preferences can reorder the model ranking.",
            "",
            "## Files",
            "",
            "- `decision_risk_sensitivity_detail.csv`",
            "- `decision_risk_sensitivity_winners.csv`",
            "- `fig_r22_decision_risk_sensitivity.*`",
        ]
    )
    text = "\n".join(lines) + "\n"
    (OUT / "R22_DECISION_RISK_SENSITIVITY_REPORT.md").write_text(text, encoding="utf-8")
    (ROUND / "R22_DECISION_RISK_SENSITIVITY_REPORT.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ROUND.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    samples = load()
    detail, winners = compute(samples)
    draw(winners, detail)
    write_report(winners, detail)
    print(f"[60] wrote {OUT / 'R22_DECISION_RISK_SENSITIVITY_REPORT.md'}", flush=True)
    print(f"[60] wrote {FIG_DIR / 'fig_r22_decision_risk_sensitivity.png'}", flush=True)


if __name__ == "__main__":
    main()


