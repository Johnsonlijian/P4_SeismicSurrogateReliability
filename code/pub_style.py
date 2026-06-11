"""Shared publication figure style for the P4 reliability-filter manuscript.

Single source of truth for fonts, line weights, model palette and export
settings. Target: Nature/Science/top-engineering-journal main-text style.

Rules encoded here (R34 figure-style normalization):
- white background, no card panels, no rounded containers, no shadows,
  no gradients, no in-figure suptitles/subtitles/footnotes;
- Arial/Helvetica, 6.5-8 pt; thin axes (0.6 pt); top/right spines off;
- one model = one colour in every figure (colourblind-safe Tol palette);
- panel labels are bold lowercase letters placed by panel_label();
- export via save_figure(): vector PDF + SVG and 600 dpi PNG.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# One model = one colour, everywhere (Paul Tol bright scheme, colourblind safe).
MODEL_COLORS = {
    "xgb_direct": "#4477AA",
    "lgbm_direct": "#EE7733",
    "hgb_direct": "#228833",
    "rf_direct": "#AA3377",
    "ridge_direct": "#777777",
    "pretrained_finetune": "#66CCEE",
    "scratch_mlp": "#CC3311",
}

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

# Long-label aliases used by some derived CSVs.
LABEL_TO_KEY = {v: k for k, v in MODEL_LABELS.items()}

PROTOCOL_MARKERS = {"main event-held-out": "o", "event-disjoint target": "s"}

GRID_KW = dict(color="#d9d9d9", linewidth=0.5)


def color_for_label(model_label: str) -> str:
    return MODEL_COLORS.get(LABEL_TO_KEY.get(model_label, model_label), "#555555")


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7.0,
            "mathtext.fontset": "dejavusans",
            "axes.titlesize": 7.5,
            "axes.titleweight": "normal",
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.3,
            "legend.frameon": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def panel_label(ax, letter: str, dx: float = -0.14, dy: float = 1.04) -> None:
    """Bold lowercase panel letter, uniform position, Nature-style."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", ha="left", va="bottom", clip_on=False)


def log_pct_axis(ax, ticks=(0.5, 1.0, 2.0, 4.0)) -> None:
    """Log x-axis in percent with plain labels and no stray minor labels."""
    ax.set_xscale("log")
    ax.set_xticks(list(ticks))
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())


def text_color_for(value: float, vmin: float, vmax: float, cmap_name: str) -> str:
    """Black or white cell text depending on background luminance."""
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap((value - vmin) / max(vmax - vmin, 1e-12))
    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "#1a1a1a" if lum > 0.55 else "white"


def save_figure(fig, base: Path, png_dpi: int = 600) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix, kwargs in (
        (".pdf", {}),
        (".svg", {}),
        (".png", {"dpi": png_dpi}),
    ):
        path = base.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02, **kwargs)
        written.append(path)
    return written


def lighten(color: str, amount: float = 0.7) -> tuple:
    """Blend a colour towards white (for restrained fills)."""
    rgb = np.array(mpl.colors.to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * amount)
