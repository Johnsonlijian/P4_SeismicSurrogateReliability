"""Polish the R30 intensity-binned conditional-coverage figure.

This script reads the derived R30 conditional-coverage summary produced by
code/69_r30_conditional_coverage_intensity.py and redraws the manuscript Figure 8
with a coverage scale suited to the observed event-disjoint diagnostic range.
It does not recompute model predictions or alter numerical results.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs" / "high_target" / "r30_conditional_coverage_intensity" / "conditional_coverage_pga_bin_summary.csv"
OUT_DIR = ROOT / "outputs" / "figures" / "high_target"
LATEX_DIR = ROOT / "submission" / "structural_safety_2026-06-01" / "latex_source_flat"

MODELS = ["Ridge direct", "LGBM direct", "XGB direct", "MLP scratch"]
BINS = ["low PGA", "mid PGA", "high PGA"]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SUMMARY)
    val = df.pivot(index="model_label", columns="pga_bin", values="median_coverage").loc[MODELS, BINS]
    q05 = df.pivot(index="model_label", columns="pga_bin", values="q05_coverage").loc[MODELS, BINS]
    q95 = df.pivot(index="model_label", columns="pga_bin", values="q95_coverage").loc[MODELS, BINS]
    events = df.pivot(index="model_label", columns="pga_bin", values="median_event_count").loc[MODELS, BINS]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
    })
    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=False)
    im = ax.imshow(val.values, cmap="magma", norm=Normalize(vmin=0.35, vmax=0.75), aspect="auto")

    for i, model in enumerate(MODELS):
        for j, bin_name in enumerate(BINS):
            med = val.loc[model, bin_name]
            lo = q05.loc[model, bin_name]
            hi = q95.loc[model, bin_name]
            n_event = int(events.loc[model, bin_name])
            ax.text(
                j,
                i,
                f"{med:.2f}\n[{lo:.2f},{hi:.2f}]\nE={n_event}",
                ha="center",
                va="center",
                color="white",
                fontsize=8.2,
                linespacing=1.08,
            )

    ax.set_xticks(np.arange(len(BINS)), BINS)
    ax.set_yticks(np.arange(len(MODELS)), MODELS)
    ax.set_xlabel("Held-out event PGA bin")
    ax.set_ylabel("Surrogate")
    ax.set_title("Event-equal coverage by held-out event intensity", loc="left", fontweight="bold", pad=8)
    ax.text(
        0.0,
        1.10,
        "True event-disjoint N=2000 test split; bins use event-median PGA",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#475569",
    )
    ax.set_xticks(np.arange(-0.5, len(BINS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MODELS), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.25)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Median event-equal coverage")
    cbar.ax.tick_params(labelsize=8)
    fig.text(
        0.13,
        0.045,
        "Cell text: median coverage, event-bootstrap 5-95% interval, and median held-out events per bin. Diagnostic only, not a conditional guarantee.",
        fontsize=7.7,
        color="#475569",
    )
    fig.subplots_adjust(left=0.20, right=0.88, top=0.84, bottom=0.23)

    for ext in ["pdf", "svg", "png", "tiff"]:
        fig.savefig(OUT_DIR / f"fig_r30_conditional_coverage_pga_bins.{ext}", dpi=450 if ext in ["png", "tiff"] else None)
    plt.close(fig)

    if LATEX_DIR.exists():
        for ext in ["pdf", "svg", "png"]:
            src = OUT_DIR / f"fig_r30_conditional_coverage_pga_bins.{ext}"
            dst = LATEX_DIR / f"Figure_8.{ext}"
            dst.write_bytes(src.read_bytes())


if __name__ == "__main__":
    main()
