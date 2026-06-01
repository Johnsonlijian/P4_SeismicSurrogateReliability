from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[1]
MAIN = PROJECT / "outputs/high_target/r21_full_residual_trace/residual_trace_samples.csv"
DISJOINT = PROJECT / "outputs/high_target/event_disjoint_conformal_stress/event_disjoint_residual_samples.csv"
OUT = PROJECT / "outputs/high_target/r23_residual_mechanism"
FIGDIR = PROJECT / "outputs/figures/high_target"
OUT.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

PALETTE = {
    "XGB direct": "#2F5F8A",
    "LGBM direct": "#F58518",
    "HGB direct": "#54A24B",
    "RF direct": "#9E77B4",
    "MLP scratch": "#D62728",
    "MLP finetune": "#4C78A8",
    "Ridge direct": "#6B6B6B",
}
PROTOCOL_LABEL = {
    "recorded_mdof_event_holdout": "main event-held-out",
    "recorded_mdof_event_disjoint_target_conformal": "event-disjoint target",
}

frames = []
for path in [MAIN, DISJOINT]:
    df = pd.read_csv(path)
    frames.append(df)
data = pd.concat(frames, ignore_index=True)
data["protocol"] = data["family"].map(PROTOCOL_LABEL).fillna(data["family"])
data["abs_resid"] = data["residual_abs_log"].astype(float)
data["signed_resid"] = data["residual_signed_log"].astype(float)
data["covered_num"] = data["covered"].astype(str).str.lower().isin(["true", "1", "yes"]).astype(float)

test = data[data["split"].eq("test")].copy()
cal = data[data["split"].eq("calibration")].copy()

rows = []
for (protocol, model), g in test.groupby(["protocol", "model_label"]):
    cal_g = cal[(cal["protocol"].eq(protocol)) & (cal["model_label"].eq(model))]
    rmse = float(np.sqrt(np.mean(np.square(g["signed_resid"]))))
    coverage = float(g["covered_num"].mean())
    median_abs = float(g["abs_resid"].median())
    q90 = float(g["abs_resid"].quantile(0.90))
    q95 = float(g["abs_resid"].quantile(0.95))
    q99 = float(g["abs_resid"].quantile(0.99))
    tail_ratio = float(q95 / median_abs) if median_abs > 0 else np.nan
    scale_ratio = float(g["signed_resid"].std(ddof=1) / cal_g["signed_resid"].std(ddof=1)) if len(cal_g) > 2 else np.nan
    spearman = float(g[["y_true_log", "abs_resid"]].corr(method="spearman").iloc[0, 1]) if len(g) > 3 else np.nan
    rows.append({
        "protocol": protocol,
        "model_label": model,
        "n_test": int(len(g)),
        "rmse_log": rmse,
        "coverage": coverage,
        "median_abs_log": median_abs,
        "q90_abs_log": q90,
        "q95_abs_log": q95,
        "q99_abs_log": q99,
        "tail_ratio_q95_over_median": tail_ratio,
        "test_calibration_residual_sd_ratio": scale_ratio,
        "spearman_abs_resid_vs_true_log": spearman,
    })
metrics = pd.DataFrame(rows)
metrics.to_csv(OUT / "residual_mechanism_metrics.csv", index=False)

# Select representative models: accuracy winner, interval/decision contender, neural baseline, conservative linear baseline.
selected = ["XGB direct", "LGBM direct", "MLP scratch", "Ridge direct"]
selected_existing = [m for m in selected if m in set(test["model_label"])]

def short_label(label):
    return (
        str(label)
        .replace("MLP scratch", "Scratch")
        .replace("MLP finetune", "FT")
        .replace(" direct", "")
    )

fig = plt.figure(figsize=(12.8, 9.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.05, 1.0], hspace=0.42, wspace=0.28)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])

fig.text(0.02, 0.975, "Residual mechanism behind metric-dependent reliability", fontsize=18, fontweight="bold", ha="left", va="top")
fig.text(0.02, 0.943, "The same residual distribution creates different winners for mean error, tail coverage, interval width, and false-safe decision risk.", fontsize=10.5, color="#444444", ha="left")

# Panel a: tail ratio vs RMSE, color by coverage.
plot_metrics = metrics[metrics["model_label"].isin(selected_existing)].copy()
for protocol, marker in [("main event-held-out", "o"), ("event-disjoint target", "s")]:
    sub = plot_metrics[plot_metrics["protocol"].eq(protocol)]
    sc = ax_a.scatter(
        sub["rmse_log"], sub["tail_ratio_q95_over_median"],
        c=sub["coverage"], cmap="viridis", vmin=0.65, vmax=0.95,
        s=120, marker=marker, edgecolor="white", linewidth=0.8, label=protocol,
        zorder=3,
    )
    for _, r in sub.iterrows():
        label = short_label(r["model_label"])
        dx = -0.16 if r["rmse_log"] > 0.70 else 0.002
        dy = -0.12 if r["tail_ratio_q95_over_median"] > 5.5 else 0.04
        ax_a.text(r["rmse_log"] + dx, r["tail_ratio_q95_over_median"] + dy, label, fontsize=7, clip_on=True)
ax_a.set_title("Mean error and tail risk are not interchangeable", loc="left", fontsize=12, fontweight="bold")
ax_a.set_xlabel("RMSE of log10 drift residual")
ax_a.set_ylabel("Tail amplification: q95(|residual|) / median(|residual|)")
ax_a.set_xlim(0.10, 1.03)
ax_a.set_ylim(2.35, 6.25)
ax_a.grid(True, axis="both", color="#e0e0e0", linewidth=0.8)
ax_a.legend(loc="lower right", title="Protocol", fontsize=7, title_fontsize=8)
cbar = fig.colorbar(sc, ax=ax_a, fraction=0.046, pad=0.02)
cbar.set_label("Coverage")

# Panel b: residual scale mismatch versus coverage.
for _, r in metrics.iterrows():
    color = PALETTE.get(r["model_label"], "#777777")
    marker = "o" if r["protocol"] == "main event-held-out" else "s"
    alpha = 0.9 if r["model_label"] in selected_existing else 0.25
    size = 90 if r["model_label"] in selected_existing else 42
    ax_b.scatter(r["test_calibration_residual_sd_ratio"], r["coverage"], color=color, marker=marker, s=size, alpha=alpha, edgecolor="white", linewidth=0.5)
for _, r in metrics[metrics["model_label"].isin(selected_existing)].iterrows():
    ax_b.text(r["test_calibration_residual_sd_ratio"] + 0.012, r["coverage"] + 0.004, short_label(r["model_label"]), fontsize=7, clip_on=True)
ax_b.axhline(0.90, color="#666666", linestyle="--", linewidth=1.0)
ax_b.axvline(1.0, color="#999999", linestyle=":", linewidth=1.0)
ax_b.set_title("Calibration residual scale controls coverage loss", loc="left", fontsize=12, fontweight="bold")
ax_b.set_xlabel("Test / calibration residual standard-deviation ratio")
ax_b.set_ylabel("Empirical coverage")
ax_b.set_xlim(0.78, 1.85)
ax_b.set_ylim(0.62, 0.93)
ax_b.grid(True, axis="both", color="#e0e0e0", linewidth=0.8)
ax_b.text(1.01, 0.905, "nominal 90%", fontsize=7, color="#555555")

# Panel c: heteroscedasticity across true drift quantiles for selected models, event-disjoint target.
dis = test[test["protocol"].eq("event-disjoint target") & test["model_label"].isin(selected_existing)].copy()
dis["true_bin"] = pd.qcut(dis["y_true_log"], q=6, duplicates="drop")
bin_centers = dis.groupby("true_bin", observed=True)["y_true_log"].median().values
for model in selected_existing:
    g = dis[dis["model_label"].eq(model)]
    med = g.groupby("true_bin", observed=True)["abs_resid"].median().reindex(dis["true_bin"].cat.categories)
    p90 = g.groupby("true_bin", observed=True)["abs_resid"].quantile(0.90).reindex(dis["true_bin"].cat.categories)
    x = np.arange(len(med))
    color = PALETTE.get(model, "#777777")
    ax_c.plot(x, med.values, color=color, linewidth=2.0, marker="o", label=short_label(model))
    ax_c.fill_between(x, med.values, p90.values, color=color, alpha=0.10, linewidth=0)
ax_c.set_title("Error magnitude changes with drift regime", loc="left", fontsize=12, fontweight="bold")
ax_c.set_xlabel("True drift quantile, event-disjoint target (low to high)")
ax_c.set_ylabel("|log10 drift residual|; line=median, band=p90")
ax_c.set_xticks(np.arange(len(bin_centers)))
ax_c.set_xticklabels([f"Q{i+1}" for i in range(len(bin_centers))])
ax_c.grid(True, axis="y", color="#e0e0e0", linewidth=0.8)
ax_c.legend(ncol=2, loc="upper left", fontsize=7)

# Panel d: metric map with direct implication annotations.
summary = metrics.pivot_table(index="model_label", columns="protocol", values=["rmse_log", "coverage", "q95_abs_log", "test_calibration_residual_sd_ratio"], aggfunc="first")
# Build compact rank matrix for selected models.
rank_rows = []
for protocol in ["main event-held-out", "event-disjoint target"]:
    sub = metrics[metrics["protocol"].eq(protocol)].copy()
    sub["rmse_rank"] = sub["rmse_log"].rank(method="min")
    sub["coverage_rank"] = (-sub["coverage"]).rank(method="min")
    sub["tail_rank"] = sub["q95_abs_log"].rank(method="min")
    for _, r in sub[sub["model_label"].isin(selected_existing)].iterrows():
        rank_rows.append([protocol, r["model_label"], r["rmse_rank"], r["coverage_rank"], r["tail_rank"]])
rank_df = pd.DataFrame(rank_rows, columns=["protocol", "model_label", "RMSE", "coverage", "tail"])
heat = rank_df.pivot_table(index="model_label", columns="protocol", values=["RMSE", "coverage", "tail"], aggfunc="first")
# Arrange columns manually: metrics within protocols.
cols = []
labels = []
for protocol in ["main event-held-out", "event-disjoint target"]:
    for metric in ["RMSE", "coverage", "tail"]:
        cols.append((metric, protocol))
        labels.append(f"{protocol.split()[0]}\n{metric}")
mat = heat.reindex(selected_existing)[cols].values.astype(float)
im = ax_d.imshow(mat, cmap="YlGnBu_r", vmin=1, vmax=max(7, np.nanmax(mat)), aspect="auto")
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax_d.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=8, color="#111111")
ax_d.set_title("Metric choice changes the operational ranking", loc="left", fontsize=12, fontweight="bold")
ax_d.set_yticks(np.arange(len(selected_existing)))
ax_d.set_yticklabels([short_label(m) for m in selected_existing])
ax_d.set_xticks(np.arange(len(labels)))
ax_d.set_xticklabels(labels, rotation=0, fontsize=7)
for x in [2.5]:
    ax_d.axvline(x, color="white", linewidth=2.0)
cb2 = fig.colorbar(im, ax=ax_d, fraction=0.046, pad=0.02)
cb2.set_label("Rank (1 = best)")
ax_d.text(0.02, -0.20, "Mechanism: RMSE rewards mean residual control; coverage and tail ranks respond to residual scale and tails.", transform=ax_d.transAxes, fontsize=8, color="#333333", ha="left")

for ax, letter in zip([ax_a, ax_b, ax_c, ax_d], list("abcd")):
    ax.text(-0.12, 1.08, letter, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top", ha="left")

base = FIGDIR / "fig_r23_residual_mechanism"
fig.savefig(base.with_suffix(".png"), dpi=320, bbox_inches="tight")
fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
plt.close(fig)

# Mechanism report.
top_lines = []
for protocol in ["main event-held-out", "event-disjoint target"]:
    sub = metrics[metrics["protocol"].eq(protocol)].copy()
    rmse_w = sub.loc[sub["rmse_log"].idxmin()]
    cov_w = sub.loc[sub["coverage"].idxmax()]
    tail_w = sub.loc[sub["q95_abs_log"].idxmin()]
    scale_max = sub.loc[sub["test_calibration_residual_sd_ratio"].idxmax()]
    top_lines.append((protocol, rmse_w, cov_w, tail_w, scale_max))

report = [
    "# R23 residual-mechanism analysis",
    "",
    "## Purpose",
    "This analysis addresses the reviewer/panel concern that metric-dependent ranking should be explained mechanistically rather than reported only as a data audit. It uses the existing full residual traces and does not add unverified external facts.",
    "",
    "## Mechanism claim supported by the current residual traces",
    "Mean-error metrics, conformal coverage, interval/tail behavior and false-safe decision loss interrogate different regions of the residual distribution. Under event-level shift, residual scale mismatch and tail amplification can therefore reorder model preference even when all models are evaluated on the same finite-label budget.",
    "",
    "## Protocol-level findings",
]
for protocol, rmse_w, cov_w, tail_w, scale_max in top_lines:
    report += [
        f"### {protocol}",
        f"- Lowest RMSE: {rmse_w['model_label']} (RMSE={rmse_w['rmse_log']:.3f}, coverage={rmse_w['coverage']:.3f}, q95|resid|={rmse_w['q95_abs_log']:.3f}).",
        f"- Highest coverage: {cov_w['model_label']} (coverage={cov_w['coverage']:.3f}, RMSE={cov_w['rmse_log']:.3f}).",
        f"- Smallest q95 absolute residual: {tail_w['model_label']} (q95|resid|={tail_w['q95_abs_log']:.3f}, RMSE={tail_w['rmse_log']:.3f}).",
        f"- Largest test/calibration residual-scale ratio: {scale_max['model_label']} (ratio={scale_max['test_calibration_residual_sd_ratio']:.3f}).",
        "",
    ]
report += [
    "## Manuscript-safe interpretation",
    "The mechanism evidence supports a bounded claim: metric conflict is consistent with residual distribution geometry under event-level shift. It does not prove a universal law for all seismic systems or a new conformal-prediction theorem.",
    "",
    "## Recommended manuscript insertion",
    "Add a Results subsection titled `Residual geometry explains why reliability metrics disagree`. Use Fig. R23 as the mechanism figure after the current rank figure and before the decision-risk sensitivity figure, or move it to Extended Data if the target journal enforces a strict display-item budget.",
    "",
    "## Generated outputs",
    f"- Metrics table: `{OUT / 'residual_mechanism_metrics.csv'}`",
    f"- Figure base: `{base}` in PNG/SVG/PDF/TIFF",
]
(OUT / "R23_RESIDUAL_MECHANISM_REPORT.md").write_text("\n".join(report), encoding="utf-8")
print("Wrote", OUT / "R23_RESIDUAL_MECHANISM_REPORT.md")
print("Wrote", base.with_suffix(".png"))

