"""Generate report-ready plots from saved ten-class MMASD experiment outputs."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research_stage4" / "visualizations" / "report_ready"
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "mplconfig"))
sys.path.insert(0, str(ROOT / ".codex-deps"))
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DISPLAY_TO_SOURCE = {
    "A0": "a0_fixed_topo_repro",
    "A1": "a2_reliability_no_dynamic",
    "A2": "a3_rcdb_full_main",
    "A3": "a4_rcdb_no_temp_smooth",
    "A4": "a5_rcdb_with_phase",
    "A5": "a6_rcdb_wrist_only",
    "A6": "a7_rcdb_multi_pair",
    "A7": "a8_rcdb_input_rel_only",
    "A8": "a9_rcdb_with_margin",
}

DISPLAY_NAMES = {
    "A0": "Fixed bilateral connections",
    "A1": "Fixed connections with reliability input",
    "A2": "Full RCDB-STGT",
    "A3": "Without temporal smoothing",
    "A4": "With phase-based input",
    "A5": "Wrist connections only",
    "A6": "Five bilateral joint pairs",
    "A7": "Dynamic connections with reliability input",
    "A8": "With confusable-class margin loss",
}

COLORS = {"base": "#4C78A8", "best": "#E45756", "full": "#72B7B2", "muted": "#A9B4C2"}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def save_figure(fig, stem: str):
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_ablation_figure():
    master = load_json(ROOT / "research_stage4" / "metrics" / "stage4_ablation_master_summary.json")
    ids = list(DISPLAY_TO_SOURCE)
    rows = [master[DISPLAY_TO_SOURCE[display_id]] for display_id in ids]
    acc = np.array([row["mean_fold_acc"] * 100 for row in rows])
    acc_sd = np.array([row["std_fold_acc"] * 100 for row in rows])
    f1 = np.array([row["pooled_oof_macro_f1"] for row in rows])
    ece = np.array([row["mean_ece"] for row in rows])
    colors = [COLORS["best"] if display_id == "A7" else COLORS["base"] for display_id in ids]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    x = np.arange(len(ids))
    axes[0].bar(x, acc, color=colors, edgecolor="white", linewidth=0.6)
    axes[0].errorbar(x, acc, yerr=acc_sd, fmt="none", ecolor="#253247", capsize=3, linewidth=1)
    axes[0].set_title("Mean fold accuracy")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 85)
    axes[0].text(0.98, 0.96, "Error bars: fold SD", transform=axes[0].transAxes,
                 ha="right", va="top", fontsize=8, color="#4B5563")

    axes[1].bar(x, f1, color=colors, edgecolor="white", linewidth=0.6)
    axes[1].set_title("Pooled out-of-fold macro-F1")
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_ylim(0, 0.8)

    axes[2].bar(x, ece, color=colors, edgecolor="white", linewidth=0.6)
    axes[2].set_title("Mean expected calibration error")
    axes[2].set_ylabel("ECE (lower is better)")
    axes[2].set_ylim(0, 0.18)

    for axis in axes:
        axis.set_xticks(x, ids, rotation=0)
        axis.grid(axis="y", alpha=0.22)
        axis.set_axisbelow(True)
    fig.suptitle("Ten-class MMASD model comparison", fontsize=15, fontweight="bold", y=1.03)
    fig.text(
        0.5,
        -0.015,
        "Configurations follow the report labels A0–A8. The RBF baseline and duplicate run are excluded.",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    fig.tight_layout()
    save_figure(fig, "ablation_comparison_10class")


def make_robustness_figure():
    data = load_json(ROOT / "research_stage4" / "robustness" / "stage4_robustness_benchmark_results.json")
    # Source run a1 is the same implementation as report A7; a3 is report A2.
    series = [
        ("A0", "a0_fixed_topo_repro", COLORS["base"]),
        ("A2", "a3_rcdb_full_main", COLORS["full"]),
        ("A7", "a1_dynamic_no_reliability", COLORS["best"]),
    ]
    conditions = ["clean", "random_20", "random_40", "wrist_occlusion", "ankle_occlusion", "consecutive_15"]
    labels = ["Unaltered", "20% joint\ndropout", "40% joint\ndropout", "Both wrists\noccluded", "Both ankles\noccluded", "15-frame\nblackout"]
    x = np.arange(len(conditions))

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
    fig.text(
        0.5,
        -0.04,
        "Corruptions were applied to held-out inputs without retraining. Values are means across folds.",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    fig.tight_layout()
    save_figure(fig, "synthetic_corruption_comparison_10class")


def read_predictions(source_id: str):
    path = ROOT / "research_stage4" / "predictions" / f"{source_id}_oof.csv"
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def make_confusion_figure():
    dataset_path = ROOT / "data" / "processed" / "mmasd_openpose" / "mmasd_openpose_64frames.npz"
    with np.load(dataset_path, allow_pickle=True) as dataset:
        class_codes = [str(code) for code in dataset["classes"].tolist()]
    model_sources = [
        ("A0", "a0_fixed_topo_repro", COLORS["base"]),
        ("A2", "a3_rcdb_full_main", COLORS["full"]),
        ("A7", "a8_rcdb_input_rel_only", COLORS["best"]),
    ]
    k = len(class_codes)
    matrices = []
    for report_id, source_id, _ in model_sources:
        rows = read_predictions(source_id)
        matrix = np.zeros((k, k), dtype=np.int64)
        for row in rows:
            true = int(row["y_true"])
            pred = int(row["y_pred"])
            matrix[true, pred] += 1
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)
        matrices.append(normalized)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), constrained_layout=True)
    image = None
    for axis, ((report_id, _, _), matrix) in zip(axes, zip(model_sources, matrices)):
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        axis.set_title(report_id, fontweight="bold")
        axis.set_xticks(range(k), class_codes, rotation=45, ha="right")
        axis.set_yticks(range(k), class_codes)
        axis.set_xlabel("Predicted class code")
        axis.set_ylabel("True class code")
        for i in range(k):
            for j in range(k):
                if matrix[i, j] >= 0.08:
                    axis.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                              color="white" if matrix[i, j] >= 0.55 else "#172033", fontsize=7)
    colorbar = fig.colorbar(image, ax=axes, shrink=0.82, pad=0.02)
    colorbar.set_label("Proportion within true class")
    fig.suptitle("Out-of-fold confusion matrices", fontsize=15, fontweight="bold")
    save_figure(fig, "confusion_matrices_a0_a2_a7_10class")


def make_bilateral_traces():
    """Plot representative held-out examples using the saved fold-specific prior."""
    import torch
    from novel_contribution.data_processing.dataset_loader import (
        create_subject_independent_folds,
        load_raw_mmasd,
        partition_train_val_test,
    )
    from research_stage4.models.rcdb_stgt import RCDB_STGT

    data = load_raw_mmasd()
    class_codes = [str(code) for code in data["classes"].tolist()]
    folds = create_subject_independent_folds(data["groups"], n_folds=5, seed=42)
    fold = 2
    train_idx, test_idx = folds[fold]
    fit_data, _, _, norm_stats = partition_train_val_test(data, train_idx, test_idx, seed=42 + fold)
    del fit_data
    mu, sd = norm_stats["mu"], norm_stats["sd"]
    x_all = np.clip((data["X"] - mu) / sd, -8.0, 8.0).astype(np.float32)

    checkpoint_path = ROOT / "research_stage4" / "checkpoints" / "a3_rcdb_full_main" / f"fold_{fold}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = RCDB_STGT(in_channels=4, num_classes=10, d_model=64)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    label_to_idx = {code: idx for idx, code in enumerate(class_codes)}
    selected = []
    for code in ("dr", "tr"):
        candidates = [idx for idx in test_idx if int(data["y"][idx]) == label_to_idx[code]]
        if not candidates:
            raise RuntimeError(f"No held-out {code} sample in fold {fold}.")
        selected.append((code, candidates[0]))

    traces = []
    with torch.no_grad():
        for code, sample_idx in selected:
            x = torch.from_numpy(x_all[sample_idx:sample_idx + 1])
            q = torch.from_numpy(data["Q"][sample_idx:sample_idx + 1].astype(np.float32))
            r_joint, _ = model.reliability_estimator(x, q)
            h = model.input_proj(x)
            layer = model.spatial_layers[0]
            h_u, h_v = h[:, :, 4, :], h[:, :, 7, :]
            r_u, r_v = r_joint[:, :, 4:5], r_joint[:, :, 7:8]
            a_uv = layer.edge_learners[0](h_u, h_v, r_u, r_v)[0, :, 0].cpu().numpy()
            q_uv = torch.sqrt(torch.clamp(r_u * r_v, min=1e-6))[0, :, 0].cpu().numpy()
            g_uv = a_uv * q_uv
            traces.append((code, a_uv, q_uv, g_uv))

    titles = {"dr": "Drumming (dr)", "tr": "Tree Pose (tr)"}
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.8), sharex=True)
    frames = np.arange(64)
    for axis, (code, edge, reliability, effective) in zip(axes, traces):
        axis.plot(frames, edge, color="#4C78A8", linewidth=1.8, label="Learned bilateral strength")
        axis.plot(frames, reliability, color="#F2A541", linewidth=1.6, linestyle="--", label="Pair reliability")
        axis.plot(frames, effective, color="#54A24B", linewidth=1.8, label="Reliability-weighted connection")
        axis.set_ylim(0, 1.02)
        axis.set_ylabel("Value")
        axis.set_title(titles[code], loc="left", fontweight="bold")
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right", frameon=False, ncol=3, fontsize=8)
    axes[-1].set_xlabel("Frame")
    fig.suptitle("Example wrist-pair traces from held-out recordings", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        -0.015,
        "Fold 2 examples; the checkpoint and reliability prior were fitted without these held-out recordings.",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    fig.tight_layout()
    save_figure(fig, "bilateral_traces_heldout_fold2")


def write_latex_snippet():
    tex = r"""% Add after the model-configuration results table in Chapter 4.
Figure~\ref{fig:ablation-comparison} presents the same summary measures as Table~\ref{tab:final-model-results} in visual form. A7 has the highest mean accuracy and pooled macro-F1 among the distinct tested configurations, while its ECE is the lowest. The figure excludes the RBF baseline and the duplicate run.

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{research_stage4/visualizations/report_ready/ablation_comparison_10class.png}
\caption{Mean fold accuracy, pooled out-of-fold macro-F1, and mean expected calibration error for the distinct tested configurations. Error bars show fold standard deviation for accuracy. The duplicate run and RBF baseline are excluded.}
\label{fig:ablation-comparison}
\end{figure}

% Add after the class-wise F1 table in Chapter 4.
Figure~\ref{fig:confusion-matrices} compares out-of-fold predictions for the fixed-connection reference, the full RCDB-STGT model, and the highest-scoring configuration. Each row is normalized by the number of true examples in that class. Diagonal cells show correct classifications, while off-diagonal cells show which class codes were confused.

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{research_stage4/visualizations/report_ready/confusion_matrices_a0_a2_a7_10class.png}
\caption{Row-normalized out-of-fold confusion matrices for the fixed-connection reference (A0), full RCDB-STGT model (A2), and highest-scoring configuration (A7). Axis labels use the stored MMASD class codes.}
\label{fig:confusion-matrices}
\end{figure}

% Add after the synthetic-corruption results table in Chapter 4.
Figure~\ref{fig:synthetic-corruption} shows how the three selected configurations performed under the synthetic corruption conditions. A7 scores higher than A0 in the listed conditions, although performance declines for all models under some corruptions. At 40\% random joint dropout, A2 has a slightly higher mean fold macro-F1 than A7. These tests use artificial changes and do not represent naturally occurring pose-estimation failures.

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{research_stage4/visualizations/report_ready/synthetic_corruption_comparison_10class.png}
\caption{Accuracy and mean fold macro-F1 under synthetic pose-corruption conditions for A0, A2, and A7. Corruptions were applied to held-out inputs without retraining.}
\label{fig:synthetic-corruption}
\end{figure}

% Optional: add to an appendix or interpretability subsection, not as aggregate performance.
Figure~\ref{fig:bilateral-traces} shows the wrist-pair signals for two individual held-out recordings processed by the full RCDB-STGT model. The learned edge strength and reliability-weighted connection vary over time. These examples illustrate model behavior; they are not aggregate performance results or validated clinical explanations.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.9\textwidth]{research_stage4/visualizations/report_ready/bilateral_traces_heldout_fold2.png}
\caption{Illustrative wrist-pair traces for two held-out recordings. The fold-specific checkpoint and reliability prior were fitted without the illustrated recordings. These examples visualize model behavior and are not validated clinical explanations.}
\label{fig:bilateral-traces}
\end{figure}
"""
    (OUT / "report_figure_snippets.tex").write_text(tex, encoding="utf-8")


def main():
    make_ablation_figure()
    make_robustness_figure()
    make_confusion_figure()
    make_bilateral_traces()
    write_latex_snippet()
    print(f"Generated report-ready figures and LaTeX snippets in: {OUT}")


if __name__ == "__main__":
    main()
