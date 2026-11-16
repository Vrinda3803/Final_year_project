"""Create a two-model synthetic-corruption plot for A0 and A7."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research_stage4" / "visualizations" / "report_ready"
OUT.mkdir(parents=True, exist_ok=True)
DATA_PATH = ROOT / "research_stage4" / "robustness" / "stage4_robustness_benchmark_results.json"

with DATA_PATH.open("r", encoding="utf-8") as stream:
    data = json.load(stream)

conditions = ["clean", "random_20", "random_40", "wrist_occlusion", "ankle_occlusion", "consecutive_15"]
labels = ["Unaltered", "20% joint\ndropout", "40% joint\ndropout", "Both wrists\noccluded", "Both ankles\noccluded", "15-frame\nblackout"]
series = [
    ("A0", "a0_fixed_topo_repro", "#4C78A8"),
    # The source run used for A7 has the same implementation as the saved A7 result.
    ("A7", "a1_dynamic_no_reliability", "#E45756"),
]
x = np.arange(len(conditions))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

for report_id, source_id, color in series:
    entries = [data[source_id][condition] for condition in conditions]
    accuracy = [entry["mean_acc"] * 100 for entry in entries]
    macro_f1 = [entry["mean_macro_f1"] for entry in entries]
    axes[0].plot(x, accuracy, marker="o", markersize=5, linewidth=2, color=color, label=report_id)
    axes[1].plot(x, macro_f1, marker="o", markersize=5, linewidth=2, color=color, label=report_id)

axes[0].set_title("Accuracy")
axes[0].set_ylabel("Mean fold accuracy (%)")
axes[0].set_ylim(20, 75)
axes[1].set_title("Macro-F1")
axes[1].set_ylabel("Mean fold macro-F1")
axes[1].set_ylim(0.15, 0.68)
for axis in axes:
    axis.set_xticks(x, labels)
    axis.grid(axis="y", alpha=0.22)
    axis.set_axisbelow(True)
    axis.legend(title="Report configuration", frameon=False)

fig.suptitle("Performance under synthetic pose corruption", fontsize=15, fontweight="bold", y=1.03)
fig.text(0.5, -0.04, "A0: fixed bilateral connections. A7: dynamic bilateral connections with joint-reliability input.\nCorruptions were applied to held-out inputs without retraining; values are means across folds.", ha="center", fontsize=9, color="#4B5563")
fig.tight_layout()
stem = OUT / "synthetic_corruption_a0_a7_10class"
fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
plt.close(fig)
print(stem)
