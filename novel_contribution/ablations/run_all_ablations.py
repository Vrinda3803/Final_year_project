"""
Batch runner for all systematic single-factor ablation experiments under matched subject splits.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".codex-deps"))
sys.path.insert(0, str(ROOT))

from novel_contribution.evaluation.evaluate_folds import run_5fold_experiment


def run_ablation_suite(seed: int = 42):
    """
    Executes the complete controlled ablation suite:
    1. baseline_stgt: Standard STGT (confidence-blind)
    2. naive_concat: STGT with raw confidence concatenated
    3. spatial_only: SR-GAT only
    4. temporal_only: RATA only
    5. full: Proposed RA-STGT (SR-GAT + RATA)
    """
    modes = ["baseline_stgt", "naive_concat", "spatial_only", "temporal_only", "full"]
    ablation_results = {}
    
    print("\n" + "#"*70)
    print("STARTING CONTROLLED SINGLE-FACTOR ABLATION SUITE (5-FOLD SUBJECT-INDEPENDENT)")
    print("#"*70)
    
    for mode in modes:
        res = run_5fold_experiment(mode=mode, save_prefix="ablation", seed=seed)
        ablation_results[mode] = {
            "mean_accuracy": res["mean_fold_accuracy"],
            "std_accuracy": res["std_fold_accuracy"],
            "mean_macro_f1": res["mean_fold_macro_f1"],
            "std_macro_f1": res["std_fold_macro_f1"],
            "pooled_accuracy": res["pooled_oof_accuracy"],
            "pooled_macro_f1": res["pooled_oof_macro_f1"],
            "uncalibrated_ece": res["mean_uncalibrated_ece"],
            "calibrated_ece": res["mean_calibrated_ece"]
        }
        
    summary_path = ROOT / "novel_contribution/results/metrics/ablation_suite_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)
        
    print("\n" + "="*70)
    print("ALL ABLATIONS COMPLETED! Summary Table:")
    print("="*70)
    print(f"{'Variant Mode':<18} | {'Mean Fold Acc':<16} | {'Mean Fold F1':<16} | {'Pooled F1':<12} | {'Calibrated ECE'}")
    print("-" * 75)
    for m, r in ablation_results.items():
        print(f"{m:<18} | {r['mean_accuracy']*100:6.2f}% ± {r['std_accuracy']*100:4.2f}% | "
              f"{r['mean_macro_f1']:6.4f} ± {r['std_macro_f1']:4.4f} | {r['pooled_macro_f1']:6.4f}     | {r['calibrated_ece']:6.4f}")
    print("="*70)
    print(f"Summary saved to: {summary_path}\n")


if __name__ == "__main__":
    run_ablation_suite(seed=42)
