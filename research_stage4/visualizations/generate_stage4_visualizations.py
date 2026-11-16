"""
Stage 4 High-Resolution Visualization Suite (300 DPI):
Reliability-Conditioned Dynamic Bilateral Graph Learning (RCDB-STGT).

Generates publication-quality figures:
  1. Fig 1: Complete Stage 4 Ablation Suite (A0-A9) & Baseline Benchmark
  2. Fig 2: Robustness Stress Degradation Curves (A0 vs A1 vs A3 under 6 corruption regimes)
  3. Fig 3: Dynamic Bilateral Attention Weight & Gate Heatmaps over Time
  4. Fig 4: Confusion Matrix Comparison (A0 vs A3 Confusable-Pair Disambiguation)
"""

from __future__ import annotations
import os, sys, json, csv
from pathlib import Path

ROOT = Path("d:/Documents/Autism_Project")
os.environ["MPLCONFIGDIR"] = str(ROOT / "research_stage4" / "scratch")
sys.path.insert(0, str(ROOT / ".codex-deps"))
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from novel_contribution.data_processing.dataset_loader import CLASSES

out_dir = ROOT / "research_stage4" / "visualizations"
out_dir.mkdir(parents=True, exist_ok=True)
scratch_dir = ROOT / "research_stage4" / "scratch"
scratch_dir.mkdir(parents=True, exist_ok=True)

CLASS_LABELS_SHORT = ["AS", "BS", "CL", "DR", "JM", "MFS", "MS", "SAC", "SQ", "TP"]


def plot_fig1_ablation_benchmark(master_summary_path: Path):
    if not master_summary_path.exists():
        print(f"Skipping Fig 1: {master_summary_path} not found yet.")
        return
        
    with open(master_summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    order = [
        "a0_fixed_topo_repro",
        "a1_dynamic_no_reliability",
        "a2_reliability_no_dynamic",
        "a3_rcdb_full_main",
        "a4_rcdb_no_temp_smooth",
        "a5_rcdb_with_phase",
        "a6_rcdb_wrist_only",
        "a7_rcdb_multi_pair",
        "a8_rcdb_input_rel_only",
        "a9_rcdb_with_margin"
    ]
    
    display_names = [
        "A0: Fixed Topo (E2 Repro)",
        "A1: Dynamic (No Rel Gate)",
        "A2: Rel Only (No Dynamic)",
        "A3: Full RCDB-STGT (Main)",
        "A4: No Temp Smoothing",
        "A5: With Phase Stream",
        "A6: Wrists Only [(4,7)]",
        "A7: Multi-Pair Bilateral",
        "A8: Input Rel Only (No Mult)",
        "A9: With Margin Loss"
    ]
    
    accs, stds, f1s = [], [], []
    valid_names = []
    
    for exp_id, d_name in zip(order, display_names):
        if exp_id in data:
            valid_names.append(d_name)
            accs.append(data[exp_id]["mean_fold_acc"] * 100)
            stds.append(data[exp_id]["std_fold_acc"] * 100)
            f1s.append(data[exp_id]["pooled_oof_macro_f1"])
            
    if not valid_names:
        return
        
    # Append RBF baseline for historical reference
    valid_names.append("RBF Kernel SVM (Ref)")
    accs.append(71.69)
    stds.append(8.14)
    f1s.append(0.6920)
    
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax1 = plt.subplots(figsize=(13, 6.5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(valid_names)))
    bars = ax1.bar(range(len(valid_names)), accs, yerr=stds, capsize=4, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8, width=0.55)
    
    ax1.set_ylabel("Mean Fold Accuracy (%)", fontsize=12, fontweight="bold", color="#2c3e50")
    ax1.set_ylim(30, 85)
    ax1.set_xticks(range(len(valid_names)))
    ax1.set_xticklabels(valid_names, rotation=35, ha="right", fontsize=9.5, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.6)
    
    ax2 = ax1.twinx()
    ax2.plot(range(len(valid_names)), f1s, color="#2c3e50", marker="s", linewidth=2.2, markersize=7, label="Pooled OOF Macro-F1")
    ax2.set_ylabel("Pooled OOF Macro-F1", fontsize=12, fontweight="bold", color="#2c3e50")
    ax2.set_ylim(0.35, 0.78)
    ax2.grid(False)
    
    for bar, acc in zip(bars, accs):
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f"{acc:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        
    plt.title("Stage 4 RCDB-STGT Ablation Benchmark: Dynamic Bilateral Graph & Reliability Gating\n(MMASD 1,374 recordings, 10 classes, 5-fold GroupKFold)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    fig.savefig(out_dir / "fig1_ablation_matrix_benchmark.png", dpi=300)
    fig.savefig(out_dir / "fig1_ablation_matrix_benchmark.svg")
    plt.close(fig)
    print("Saved Figure 1 (Ablation Benchmark).")


def plot_fig2_robustness(robustness_results_path: Path):
    if not robustness_results_path.exists():
        print(f"Skipping Fig 2: {robustness_results_path} not found yet.")
        return
        
    with open(robustness_results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    modes = ["clean", "random_20", "random_40", "wrist_occlusion", "ankle_occlusion", "consecutive_15"]
    mode_labels = ["Clean", "Random 20%", "Random 40%", "Wrist Occl.", "Ankle Occl.", "15-Frame Drop"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    styles = {
        "a0_fixed_topo_repro": {"color": "#e74c3c", "marker": "o", "label": "A0: Fixed Bilateral (E2)"},
        "a1_dynamic_no_reliability": {"color": "#3498db", "marker": "^", "label": "A1: Dynamic (No Rel Gate)"},
        "a3_rcdb_full_main": {"color": "#27ae60", "marker": "s", "label": "A3: Full RCDB-STGT (Main)"}
    }
    
    for exp_id, s in styles.items():
        if exp_id in data:
            accs = [data[exp_id][m]["mean_acc"] * 100 for m in modes]
            f1s = [data[exp_id][m]["mean_macro_f1"] for m in modes]
            ax1.plot(mode_labels, accs, marker=s["marker"], color=s["color"], linewidth=2.2, markersize=7, label=s["label"])
            ax2.plot(mode_labels, f1s, marker=s["marker"], color=s["color"], linewidth=2.2, markersize=7, label=s["label"])
            
    ax1.set_title("(A) Accuracy Under Pose Degradation", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Mean Accuracy (%)", fontsize=11, fontweight="bold")
    ax1.set_ylim(20, 70)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower left", fontsize=10)
    ax1.set_xticklabels(mode_labels, rotation=25, ha="right")
    
    ax2.set_title("(B) Macro-F1 Under Pose Degradation", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Macro-F1 Score", fontsize=11, fontweight="bold")
    ax2.set_ylim(0.15, 0.65)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="lower left", fontsize=10)
    ax2.set_xticklabels(mode_labels, rotation=25, ha="right")
    
    plt.suptitle("Robustness Stress Test: Fixed Bilateral vs Dynamic Gated Communication Under Sensor Degradation", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "fig2_robustness_stress_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig2_robustness_stress_comparison.svg", bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure 2 (Robustness Stress Comparison).")


def generate_fig3_for_sample(model, x_sample, q_sample, class_name: str):
    model.eval()
    with torch.no_grad():
        r_joint, _ = model.reliability_estimator(x_sample, q_sample)
        h = model.input_proj(x_sample)
        
        # In spatial layer 0:
        layer0 = model.spatial_layers[0]
        h_u = h[:, :, 4, :]      # RWrist
        h_v = h[:, :, 7, :]      # LWrist
        r_u = r_joint[:, :, 4:5]
        r_v = r_joint[:, :, 7:8]
        
        a_uv = layer0.edge_learners[0](h_u, h_v, r_u, r_v)
        q_uv = torch.sqrt(torch.clamp(r_u * r_v, min=1e-6))
        g_uv = q_uv * a_uv
        
    a_uv_t = a_uv[0, :, 0].cpu().numpy()
    g_uv_t = g_uv[0, :, 0].cpu().numpy()
    q_uv_t = q_uv[0, :, 0].cpu().numpy()
    
    t = np.arange(len(a_uv_t))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    ax1.plot(t, a_uv_t, color="#2980b9", linewidth=2.0, label="Dynamic Bilateral Strength $a_{uv}(t)$")
    ax1.plot(t, q_uv_t, color="#e67e22", linestyle="--", linewidth=1.8, label="Relational Reliability $q_{uv}(t)$")
    ax1.set_ylabel("Weight / Reliability", fontsize=11, fontweight="bold")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.set_title(f"Dynamic Bilateral Attention Profile: Wrist Pair (RWrist-LWrist) - Class: {class_name}", fontsize=12, fontweight="bold")
    
    ax2.fill_between(t, 0, g_uv_t, color="#27ae60", alpha=0.4, label="Reliability-Gated Connection $g_{uv}(t)$")
    ax2.plot(t, g_uv_t, color="#27ae60", linewidth=2.2)
    ax2.axhline(0.5, color="#c0392b", linestyle=":", linewidth=1.5, label="Fixed Baseline Topology Weight (0.50)")
    ax2.set_xlabel("Time Frame (t = 0..63)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Gated Edge Weight", fontsize=11, fontweight="bold")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    clean_name = class_name.lower().replace(" ", "_")
    fig.savefig(out_dir / f"fig3_dynamic_bilateral_heatmap_{clean_name}.png", dpi=300)
    fig.savefig(out_dir / f"fig3_dynamic_bilateral_heatmap_{clean_name}.svg")
    plt.close(fig)
    print(f"Saved Figure 3 (Dynamic Bilateral Heatmap: {class_name}).")


def load_oof_targets_and_preds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    y_true, y_pred = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y_true.append(int(row["y_true"]))
            y_pred.append(int(row["y_pred"]))
    return np.array(y_true), np.array(y_pred)


def plot_fig4_confusion_matrices(pred_a0_path: Path, pred_a3_path: Path):
    if not (pred_a0_path.exists() and pred_a3_path.exists()):
        print("Skipping Fig 4: OOF predictions not found.")
        return
        
    y_true_a0, y_pred_a0 = load_oof_targets_and_preds(pred_a0_path)
    y_true_a3, y_pred_a3 = load_oof_targets_and_preds(pred_a3_path)
    
    cm_a0 = confusion_matrix(y_true_a0, y_pred_a0, labels=list(range(len(CLASSES))))
    cm_a3 = confusion_matrix(y_true_a3, y_pred_a3, labels=list(range(len(CLASSES))))
    
    # Normalize by row (recall)
    cm_a0_norm = cm_a0.astype(float) / cm_a0.sum(axis=1, keepdims=True)
    cm_a3_norm = cm_a3.astype(float) / cm_a3.sum(axis=1, keepdims=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    im1 = ax1.imshow(cm_a0_norm, cmap="Blues", vmin=0, vmax=1.0)
    ax1.set_title("A0: Fixed Bilateral Topology (E2 Reproduction)", fontsize=12, fontweight="bold")
    ax1.set_xticks(range(10))
    ax1.set_yticks(range(10))
    ax1.set_xticklabels(CLASS_LABELS_SHORT, fontsize=9.5, fontweight="bold")
    ax1.set_yticklabels(CLASS_LABELS_SHORT, fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("True Activity", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Predicted Activity", fontsize=11, fontweight="bold")
    
    im2 = ax2.imshow(cm_a3_norm, cmap="Greens", vmin=0, vmax=1.0)
    ax2.set_title("A3: Proposed RCDB-STGT (Dynamic + Reliability-Gated)", fontsize=12, fontweight="bold")
    ax2.set_xticks(range(10))
    ax2.set_yticks(range(10))
    ax2.set_xticklabels(CLASS_LABELS_SHORT, fontsize=9.5, fontweight="bold")
    ax2.set_yticklabels(CLASS_LABELS_SHORT, fontsize=9.5, fontweight="bold")
    ax2.set_xlabel("Predicted Activity", fontsize=11, fontweight="bold")
    
    # Add text annotations for key diagonals
    for i in range(10):
        for j in range(10):
            val1 = cm_a0_norm[i, j]
            val2 = cm_a3_norm[i, j]
            if val1 > 0.05:
                ax1.text(j, i, f"{val1:.2f}", ha="center", va="center", color="white" if val1 > 0.5 else "black", fontsize=8)
            if val2 > 0.05:
                ax2.text(j, i, f"{val2:.2f}", ha="center", va="center", color="white" if val2 > 0.5 else "black", fontsize=8)
                
    plt.suptitle("Out-of-Fold Normalized Confusion Matrices: Confusable Exercise Pair Disambiguation\n(N = 1,374 recordings)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(out_dir / "fig4_confusion_matrices_comparison.png", dpi=300)
    fig.savefig(out_dir / "fig4_confusion_matrices_comparison.svg")
    plt.close(fig)
    print("Saved Figure 4 (Confusion Matrices Comparison).")


def main():
    stage4_dir = ROOT / "research_stage4"
    master_summary = stage4_dir / "metrics" / "stage4_ablation_master_summary.json"
    robustness_res = stage4_dir / "robustness" / "stage4_robustness_benchmark_results.json"
    a0_pred = stage4_dir / "predictions" / "a0_fixed_topo_repro_oof.csv"
    a3_pred = stage4_dir / "predictions" / "a3_rcdb_full_main_oof.csv"
    
    plot_fig1_ablation_benchmark(master_summary)
    plot_fig2_robustness(robustness_res)
    plot_fig4_confusion_matrices(a0_pred, a3_pred)
    
    # Generate Figure 3 from A3 checkpoint
    a3_ckpt_path = stage4_dir / "checkpoints" / "a3_rcdb_full_main" / "fold_2.pt"
    if a3_ckpt_path.exists():
        import torch
        from research_stage4.models.rcdb_stgt import RCDB_STGT
        from novel_contribution.data_processing.dataset_loader import load_raw_mmasd
        
        data = load_raw_mmasd()
        model = RCDB_STGT(in_channels=4, num_classes=10, d_model=64)
        ckpt = torch.load(a3_ckpt_path, map_location=torch.device("cpu"), weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.reliability_estimator.fit_velocity_priors(torch.tensor(data["X"]), torch.tensor(data["Q"]))
        
        # Pick a Drumming sample (label 3) and a Tree Pose sample (label 9)
        drum_idx = int(np.where(data["y"] == 3)[0][0])
        tree_idx = int(np.where(data["y"] == 9)[0][0])
        
        x_drum = torch.tensor(data["X"][drum_idx:drum_idx+1], dtype=torch.float32)
        q_drum = torch.tensor(data["Q"][drum_idx:drum_idx+1], dtype=torch.float32)
        generate_fig3_for_sample(model, x_drum, q_drum, "Drumming")
        
        x_tree = torch.tensor(data["X"][tree_idx:tree_idx+1], dtype=torch.float32)
        q_tree = torch.tensor(data["Q"][tree_idx:tree_idx+1], dtype=torch.float32)
        generate_fig3_for_sample(model, x_tree, q_tree, "Tree Pose")


if __name__ == "__main__":
    main()
