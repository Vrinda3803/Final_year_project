"""Create row-normalized confusion matrices for report configurations A0 and A7."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research_stage4" / "visualizations" / "report_ready"
OUT.mkdir(parents=True, exist_ok=True)
DATASET = ROOT / "data" / "processed" / "mmasd_openpose" / "mmasd_openpose_64frames.npz"
PREDICTIONS = ROOT / "research_stage4" / "predictions"

with np.load(DATASET, allow_pickle=True) as dataset:
    class_codes = [str(code) for code in dataset["classes"].tolist()]

models = [
    ("A0", "a0_fixed_topo_repro", "#4C78A8"),
    # This saved prediction file is the same A7 implementation used in the report.
    ("A7", "a8_rcdb_input_rel_only", "#E45756"),
]
k = len(class_codes)
matrices = []
for report_id, source_id, color in models:
    path = PREDICTIONS / f"{source_id}_oof.csv"
    matrix = np.zeros((k, k), dtype=np.int64)
    with path.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            matrix[int(row["y_true"]), int(row["y_pred"])] += 1
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)
    matrices.append((report_id, color, normalized))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.1), constrained_layout=True)
image = None
for axis, (report_id, _, matrix) in zip(axes, matrices):
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    axis.set_title(report_id, fontweight="bold")
    axis.set_xticks(range(k), class_codes, rotation=45, ha="right")
    axis.set_yticks(range(k), class_codes)
    axis.set_xlabel("Predicted class code")
    axis.set_ylabel("True class code")
    for i in range(k):
        for j in range(k):
            if matrix[i, j] >= 0.08:
                axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                          color="white" if matrix[i, j] >= 0.55 else "#172033", fontsize=7)
colorbar = fig.colorbar(image, ax=axes, shrink=0.82, pad=0.02)
colorbar.set_label("Proportion within true class")
fig.suptitle("Out-of-fold confusion matrices: A0 and A7", fontsize=14, fontweight="bold")
stem = OUT / "confusion_matrices_a0_a7_10class"
fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
plt.close(fig)
print(stem)
