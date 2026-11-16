"""Generate a compact, title-free MMASD system architecture diagram."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research_stage4" / "visualizations" / "report_ready"
OUT.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(8.2, 11.0))
ax.set_xlim(0, 8.2)
ax.set_ylim(0, 11.0)
ax.axis("off")

edge = "#263746"
text = "#17212B"


def box(x, y, w, h, heading, fontsize=10.2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.035,rounding_size=0.10",
        linewidth=1.2,
        edgecolor=edge,
        facecolor="white",
    ))
    ax.text(x + w / 2, y + h / 2, heading, ha="center", va="center",
            fontsize=fontsize, color=text, fontweight="medium", wrap=True)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=11, linewidth=1.25, color=edge))


# Main evaluation and data flow
box(1.2, 10.05, 5.8, 0.52, "Processed MMASD OpenPose Sequences")
box(1.2, 9.12, 5.8, 0.52, "Five-Fold Cohort Grouping")
box(1.2, 8.19, 5.8, 0.52, "Fold-Specific Data Preparation")
arrow(4.1, 10.03, 4.1, 9.66)
arrow(4.1, 9.10, 4.1, 8.73)

# Parallel input representations
box(0.45, 7.05, 3.35, 0.60, "Feature Projection", fontsize=9.7)
box(4.40, 7.05, 3.35, 0.60, "Joint Reliability Estimation", fontsize=9.5)
arrow(4.1, 8.17, 2.12, 7.68)
arrow(4.1, 8.17, 6.07, 7.68)

# Model and output
box(0.70, 5.92, 6.80, 0.60, "Spatial Graph Attention × 2")
arrow(2.12, 7.03, 3.05, 6.55)
arrow(6.07, 7.03, 5.15, 6.55)

box(0.70, 4.98, 6.80, 0.54, "Reliability-Weighted Joint Pooling")
arrow(4.1, 5.90, 4.1, 5.54)
box(0.70, 4.04, 6.80, 0.54, "Temporal Self-Attention × 2")
arrow(4.1, 4.96, 4.1, 4.60)
box(0.70, 3.10, 6.80, 0.54, "Temporal Mean Pooling")
arrow(4.1, 4.02, 4.1, 3.66)
box(0.70, 2.16, 6.80, 0.54, "Linear Classifier")
arrow(4.1, 3.08, 4.1, 2.72)
box(0.70, 1.22, 6.80, 0.54, "Ten Activity Classes")
arrow(4.1, 2.14, 4.1, 1.78)
box(0.70, 0.28, 6.80, 0.54, "Held-Out Fold Evaluation")
arrow(4.1, 1.20, 4.1, 0.84)

fig.subplots_adjust(left=0.025, right=0.975, top=0.99, bottom=0.02)
stem = OUT / "system_architecture_mmasd"
fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
plt.close(fig)
print(stem)
