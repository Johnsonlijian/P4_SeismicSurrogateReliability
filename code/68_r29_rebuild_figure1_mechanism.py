from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib import rcParams

PROJECT = Path(r"R:\NAS_DRIVE\IMUT\1-Research_Output\1-Papers\1_In_Preparation\2026-Claude\projects\P4_SeismicFoundationModel")
OUTDIR = PROJECT / "outputs" / "figures" / "high_target"
OUTDIR.mkdir(parents=True, exist_ok=True)

rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
})

fig = plt.figure(figsize=(12.0, 6.9), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis("off")

colors = {
    "blue": "#DDEFF8",
    "blue_edge": "#2B6C8A",
    "cream": "#FFF3CC",
    "cream_edge": "#967029",
    "green": "#E4F3DD",
    "green_edge": "#4E7D43",
    "rose": "#F7E0DD",
    "rose_edge": "#9A4C43",
    "gray": "#F2F3F5",
    "gray_edge": "#6B7280",
    "ink": "#1F2933",
}


def box(x, y, w, h, text, fc, ec, fs=10.5, lw=1.4, radius=0.08, weight="normal"):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.018,rounding_size={radius}",
                       linewidth=lw, facecolor=fc, edgecolor=ec)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color=colors["ink"], fontweight=weight, linespacing=1.08)
    return p


def arrow(x1, y1, x2, y2, color="#4B5563", lw=1.5, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=12, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=5, shrinkB=5))

# Title band
ax.text(0.55, 6.55, "Research object and mechanism of reliability-aware seismic surrogate screening",
        fontsize=16, fontweight="bold", color="#111827", ha="left", va="center")
ax.text(0.55, 6.18,
        "Recorded motions drive nonlinear numerical response labels; event-level uncertainty controls a necessary false-safe eligibility filter.",
        fontsize=10.2, color="#4B5563", ha="left", va="center")
ax.plot([0.55, 11.45], [5.93, 5.93], color="#CBD5E1", lw=1.0)

# Main pipeline
ys = 4.74
w, h = 1.48, 0.78
xs = [0.65, 2.35, 4.05, 5.75, 7.45, 9.15]
labels = [
    "Recorded\nNSMP motions",
    "Nonlinear\nMDOF response",
    "Finite target\nlabels",
    "Event-disjoint\ntrain/cal/test",
    "Split conformal\nprediction interval",
    "Drift-threshold\nscreening"
]
fcs = [colors["blue"], colors["blue"], colors["cream"], colors["gray"], colors["green"], colors["cream"]]
ecs = [colors["blue_edge"], colors["blue_edge"], colors["cream_edge"], colors["gray_edge"], colors["green_edge"], colors["cream_edge"]]
for x, lab, fc, ec in zip(xs, labels, fcs, ecs):
    box(x, ys, w, h, lab, fc, ec, fs=9.8, weight="bold" if "Finite" in lab or "Split" in lab else "normal")
for i in range(len(xs)-1):
    arrow(xs[i]+w, ys+h/2, xs[i+1], ys+h/2)

# False-safe metric branch
box(8.05, 3.36, 1.75, 0.70, "False-safe\nevent probability", colors["rose"], colors["rose_edge"], fs=9.4)
box(10.05, 3.36, 1.55, 0.70, r"$\beta_{\mathrm{FS,cons}}$\neligibility filter", colors["green"], colors["green_edge"], fs=9.4, weight="bold")
arrow(9.90, ys+0.08, 8.93, 4.05, rad=-0.18)
arrow(9.80, 3.71, 10.05, 3.71)
box(10.05, 2.35, 1.55, 0.62, "Action: accept,\nwiden, or reject", colors["gray"], colors["gray_edge"], fs=9.2)
arrow(10.82, 3.36, 10.82, 2.97)

# Guardrail architecture panel
ax.text(0.65, 3.82, "Event-level validity guardrails", fontsize=12.5, fontweight="bold", color="#111827", ha="left")
box(0.65, 2.96, 2.35, 0.66, "No record leakage\nby event identifier", colors["gray"], colors["gray_edge"], fs=9.1)
box(3.22, 2.96, 2.35, 0.66, "Equal event weight\nnot row weight", colors["gray"], colors["gray_edge"], fs=9.1)
box(5.79, 2.96, 2.35, 0.66, "Bootstrap events\nnot component rows", colors["gray"], colors["gray_edge"], fs=9.1)
arrow(3.00, 3.29, 3.22, 3.29, color="#64748B")
arrow(5.57, 3.29, 5.79, 3.29, color="#64748B")

# Evidence/meaning strip
ax.add_patch(Rectangle((0.65, 0.70), 10.95, 1.25, facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.0))
ax.text(0.85, 1.62, "Interpretation for Structural Safety", fontsize=11.8, fontweight="bold", color="#111827", ha="left")
ax.text(0.85, 1.22,
        "The filter targets the critical error mode: unsafe numerical response classified as safe. Passing the filter is necessary, not sufficient,",
        fontsize=9.3, color="#374151", ha="left")
ax.text(0.85, 0.91,
        "because excessive false-unsafe decisions can make a surrogate conservative but non-discriminatory for practical screening.",
        fontsize=9.3, color="#374151", ha="left")
ax.text(8.45, 1.39, "Response labels: numerical MDOF outputs", fontsize=9.0, color="#6B7280", ha="left")
ax.text(8.45, 1.05, "Ground motions: recorded public events", fontsize=9.0, color="#6B7280", ha="left")

# Panel labels for journal style
ax.text(0.40, 5.49, "a", fontsize=13, fontweight="bold", color="#111827")
ax.text(0.40, 3.93, "b", fontsize=13, fontweight="bold", color="#111827")
ax.text(0.40, 1.70, "c", fontsize=13, fontweight="bold", color="#111827")

fig.savefig(OUTDIR / "fig_r29_mechanism_gate_figure1.pdf", bbox_inches="tight", pad_inches=0.04)
fig.savefig(OUTDIR / "fig_r29_mechanism_gate_figure1.png", dpi=450, bbox_inches="tight", pad_inches=0.04)
fig.savefig(OUTDIR / "fig_r29_mechanism_gate_figure1.svg", bbox_inches="tight", pad_inches=0.04)
plt.close(fig)
print(OUTDIR / "fig_r29_mechanism_gate_figure1.pdf")
