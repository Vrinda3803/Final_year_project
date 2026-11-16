"""
Robustness stress testing under controlled missingness (0% to 40%) and limb occlusion.
"""
from __future__ import annotations
import sys, json, copy
from pathlib import Path
from typing import Dict, Any, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".codex-deps"))
sys.path.insert(0, str(ROOT))

import torch
import numpy as np
from torch.utils.data import DataLoader

from novel_contribution.data_processing.dataset_loader import (
    load_raw_mmasd,
    create_subject_independent_folds,
    partition_train_val_test,
    SkeletonDataset,
    CLASSES
)
from novel_contribution.models.ra_stgt import RASTGT
from novel_contribution.training.trainer import compute_metrics, run_evaluation_pass


def apply_synthetic_missingness(
    dataset: SkeletonDataset,
    dropout_prob: float = 0.0,
    seed: int = 42
) -> SkeletonDataset:
    """
    Applies controlled keypoint dropout: with probability p, drops keypoints by zeroing coordinates and confidence.
    """
    if dropout_prob <= 0.0:
        return dataset
        
    rng = np.random.default_rng(seed)
    X_corrupt = dataset.X.clone()
    Q_corrupt = dataset.Q.clone()
    
    # Shape: (N, T, V)
    mask = rng.random(size=Q_corrupt.shape) < dropout_prob
    mask_tensor = torch.tensor(mask, dtype=torch.bool)
    
    # Zero out coordinates and confidence for dropped joints
    Q_corrupt[mask_tensor] = 0.0
    X_corrupt[mask_tensor.unsqueeze(-1).expand_as(X_corrupt)] = 0.0
    
    return SkeletonDataset(
        X_corrupt.numpy(),
        Q_corrupt.numpy(),
        dataset.y.numpy(),
        dataset.groups,
        dataset.clips
    )


def run_robustness_evaluation(seed: int = 42) -> Dict[str, Any]:
    """
    Compares baseline_stgt vs full RA-STGT across missingness levels (0% to 40%).
    """
    missing_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
    models_to_test = ["baseline_stgt", "full"]
    results = {m: [] for m in models_to_test}
    
    data = load_raw_mmasd()
    folds = create_subject_independent_folds(data["groups"], n_folds=5, seed=seed)
    device = torch.device("cpu")
    
    print("\n" + "="*70)
    print("STARTING ROBUSTNESS BENCHMARKING UNDER CONTROLLED OCCLUSION (0% - 40%)")
    print("="*70)
    
    for mode in models_to_test:
        print(f"\n--- Testing Robustness for: {mode} ---")
        level_accuracies = {lvl: [] for lvl in missing_levels}
        level_macro_f1s = {lvl: [] for lvl in missing_levels}
        
        for fold_num, (tr_idx, te_idx) in enumerate(folds, 1):
            _, _, test_d, _ = partition_train_val_test(data, tr_idx, te_idx, seed=seed + fold_num)
            
            # Load trained checkpoint
            ckpt_path = ROOT / f"novel_contribution/results/checkpoints/ra_stgt_{mode}_fold{fold_num}.pt"
            if not ckpt_path.exists():
                print(f"Warning: Checkpoint not found at {ckpt_path}. Skipping.")
                continue
                
            model = RASTGT(mode=mode).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()
            
            for lvl in missing_levels:
                corrupt_dataset = apply_synthetic_missingness(test_d["dataset"], dropout_prob=lvl, seed=seed + fold_num)
                loader = DataLoader(corrupt_dataset, batch_size=32, shuffle=False)
                
                _, targets, preds, _, _ = run_evaluation_pass(model, loader, device)
                acc, f1, _ = compute_metrics(targets, preds)
                level_accuracies[lvl].append(acc)
                level_macro_f1s[lvl].append(f1)
                
        # Aggregate across 5 folds
        for lvl in missing_levels:
            mean_acc = float(np.mean(level_accuracies[lvl]))
            std_acc = float(np.std(level_accuracies[lvl]))
            mean_f1 = float(np.mean(level_macro_f1s[lvl]))
            std_f1 = float(np.std(level_macro_f1s[lvl]))
            
            results[mode].append({
                "missingness_level": lvl,
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "mean_macro_f1": mean_f1,
                "std_macro_f1": std_f1
            })
            print(f"Mode: {mode:<14} | Missingness: {lvl*100:4.1f}% | Acc: {mean_acc*100:6.2f}% | Macro-F1: {mean_f1:6.4f}")
            
    # Calculate degradation slope delta F1 = F1_0% - F1_40%
    for mode in models_to_test:
        f1_clean = results[mode][0]["mean_macro_f1"]
        f1_40 = results[mode][-1]["mean_macro_f1"]
        delta = f1_clean - f1_40
        print(f"\n{mode} Degradation from 0% to 40% missingness: Clean F1 = {f1_clean:.4f} -> 40% F1 = {f1_40:.4f} (Drop: {delta:+.4f})")
        
    out_path = ROOT / "novel_contribution/results/metrics/robustness_stress_test_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nRobustness results saved to: {out_path}\n")
    return results


if __name__ == "__main__":
    run_robustness_evaluation()
