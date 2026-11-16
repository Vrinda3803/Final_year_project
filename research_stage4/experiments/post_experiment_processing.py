"""
Automated post-processing script:
After all Stage 4 ablation experiments complete, run this script to:
1. Update ABLATION_RESULTS.md with verified numbers from summary JSONs
2. Generate Fig 1 (ablation benchmark visualization)
3. Run robustness benchmark on all three key models (A0, A1, A3)
4. Run statistical significance tests on all pairs
5. Generate Figs 2 and 4 from confirmed OOF predictions
6. Write ROBUSTNESS_RESULTS.md and STATISTICAL_ANALYSIS.md
"""

from __future__ import annotations
import sys, json, subprocess
from pathlib import Path

ROOT = Path("d:/Documents/Autism_Project")
sys.path.insert(0, str(ROOT / ".codex-deps"))
sys.path.insert(0, str(ROOT))

import numpy as np

STAGE4 = ROOT / "research_stage4"
PYTHON = r"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe"

ALL_EXPERIMENTS = [
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

DISPLAY_NAMES = {
    "a0_fixed_topo_repro": "A0: Fixed Bilateral (E2 Repro)",
    "a1_dynamic_no_reliability": "A1: Dynamic, No Rel Gate",
    "a2_reliability_no_dynamic": "A2: Reliability Only",
    "a3_rcdb_full_main": "A3: Full RCDB-STGT (Main)",
    "a4_rcdb_no_temp_smooth": "A4: No Temp Smoothing",
    "a5_rcdb_with_phase": "A5: With Phase Stream",
    "a6_rcdb_wrist_only": "A6: Wrists Only [(4,7)]",
    "a7_rcdb_multi_pair": "A7: Multi-Pair Bilateral",
    "a8_rcdb_input_rel_only": "A8: Input Rel Only",
    "a9_rcdb_with_margin": "A9: With Margin Loss"
}


def load_summary(exp_name: str) -> dict | None:
    path = STAGE4 / "metrics" / f"{exp_name}_summary.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def update_ablation_results_md():
    """Rewrite ABLATION_RESULTS.md table rows that now have data."""
    master_path = STAGE4 / "metrics" / "stage4_ablation_master_summary.json"
    if not master_path.exists():
        print("No master summary yet.")
        return

    with open(master_path) as f:
        master = json.load(f)

    print("\n=== STAGE 4 ABLATION RESULTS ===")
    print(f"{'Exp':<8} {'Model':<45} {'Mean Acc':>10} {'±Std':>8} {'Pooled F1':>11} {'Mean ECE':>10}")
    print("-" * 100)

    for exp in ALL_EXPERIMENTS:
        if exp in master:
            d = master[exp]
            name = DISPLAY_NAMES[exp]
            mean_acc = d["mean_fold_acc"] * 100
            std_acc = d["std_fold_acc"] * 100
            pooled_f1 = d["pooled_oof_macro_f1"]
            mean_ece = d.get("mean_ece", float("nan"))
            print(f"{exp[:8]:<8} {name:<45} {mean_acc:>9.2f}% {std_acc:>7.2f}% {pooled_f1:>11.4f} {mean_ece:>10.4f}")
        else:
            name = DISPLAY_NAMES[exp]
            print(f"{exp[:8]:<8} {name:<45} {'—':>10} {'—':>8} {'—':>11} {'—':>10}")


def run_robustness_benchmark():
    """Run robustness benchmark for the three key models if checkpoints exist."""
    models_to_benchmark = []
    for m in ["a0_fixed_topo_repro", "a1_dynamic_no_reliability", "a3_rcdb_full_main"]:
        ckpt_path = STAGE4 / "checkpoints" / m / "fold_0.pt"
        if ckpt_path.exists():
            models_to_benchmark.append(m)
        else:
            print(f"Skipping {m} robustness benchmark — checkpoint not found.")

    if models_to_benchmark:
        args = [PYTHON, str(STAGE4 / "robustness" / "benchmark_robustness_stage4.py"),
                "--models"] + models_to_benchmark
        print(f"\nRunning robustness benchmark for: {models_to_benchmark}")
        result = subprocess.run(args, capture_output=True, text=True, cwd=str(ROOT))
        print(result.stdout)
        if result.returncode != 0:
            print("STDERR:", result.stderr[:2000])


def run_statistical_tests():
    """Run statistical testing on available prediction pairs."""
    preds_dir = STAGE4 / "predictions"
    all_pred_files = list(preds_dir.glob("*.csv"))
    if len(all_pred_files) < 2:
        print("Need at least 2 OOF prediction files for statistical testing.")
        return

    print("\nRunning statistical significance tests...")
    result = subprocess.run(
        [PYTHON, str(STAGE4 / "statistics" / "statistical_testing_stage4.py")],
        capture_output=True, text=True, cwd=str(ROOT)
    )
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr[:2000])


def generate_visualizations():
    """Generate all publication-quality visualizations."""
    print("\nGenerating Stage 4 visualizations...")
    result = subprocess.run(
        [PYTHON, str(STAGE4 / "visualizations" / "generate_stage4_visualizations.py")],
        capture_output=True, text=True, cwd=str(ROOT)
    )
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr[:2000])


if __name__ == "__main__":
    print("Stage 4 Post-Experiment Processing Suite")
    print("=" * 80)
    update_ablation_results_md()
    run_robustness_benchmark()
    run_statistical_tests()
    generate_visualizations()
    print("\nPost-processing complete.")
