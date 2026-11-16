"""
5-Fold Subject-Independent evaluation runner for RA-STGT and ablation variants.
"""
from __future__ import annotations
import sys, os, json, csv
from pathlib import Path
from typing import Dict, Any, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".codex-deps"))
sys.path.insert(0, str(ROOT))

import torch
import numpy as np

from novel_contribution.data_processing.dataset_loader import (
    load_raw_mmasd, 
    create_subject_independent_folds, 
    partition_train_val_test,
    CLASSES
)
from novel_contribution.models.ra_stgt import RASTGT
from novel_contribution.training.trainer import train_fold, compute_metrics


def run_5fold_experiment(
    mode: str = "full",
    config_path: str | Path | None = None,
    save_prefix: str = "ra_stgt",
    seed: int = 42
) -> Dict[str, Any]:
    """
    Executes a complete 5-fold cross-validation experiment for a given model mode.
    """
    if config_path is None:
        config_path = ROOT / "novel_contribution/configs/default_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
        
    print(f"\n=======================================================")
    print(f"Starting 5-Fold Experiment | Mode: {mode} | Seed: {seed}")
    print(f"=======================================================")
    
    device = torch.device("cpu")
    data = load_raw_mmasd()
    folds = create_subject_independent_folds(data["groups"], n_folds=5, seed=seed)
    
    fold_results = []
    all_oof_records = []
    
    for fold_num, (tr_idx, te_idx) in enumerate(folds, 1):
        print(f"\n--- Running Fold {fold_num} / 5 ---")
        fit_d, val_d, test_d, _ = partition_train_val_test(data, tr_idx, te_idx, seed=seed + fold_num)
        
        # Instantiate model for this fold
        model = RASTGT(
            in_channels=config["dataset"]["in_channels"],
            num_classes=config["dataset"]["num_classes"],
            d_model=config["model"]["d_model"],
            num_spatial_heads=config["model"]["num_spatial_heads"],
            num_temporal_heads=config["model"]["num_temporal_heads"],
            num_spatial_layers=config["model"]["spatial_layers"],
            num_temporal_layers=config["model"]["temporal_layers"],
            dropout=config["model"]["dropout"],
            mode=mode
        )
        
        torch.manual_seed(seed + fold_num)
        np.random.seed(seed + fold_num)
        
        res = train_fold(
            model=model,
            fit_data=fit_d,
            val_data=val_d,
            test_data=test_d,
            fold=fold_num,
            config=config,
            device=device,
            save_checkpoint=True
        )
        
        print(f"Fold {fold_num} Results: Val Acc: {res['val_accuracy']*100:.2f}%, Val F1: {res['val_macro_f1']:.4f} | "
              f"Test Acc: {res['test_accuracy']*100:.2f}%, Test F1: {res['test_macro_f1']:.4f} | "
              f"Uncal ECE: {res['uncalibrated_ece']:.4f} -> Cal ECE: {res['calibrated_ece']:.4f}")
        
        fold_results.append(res)
        all_oof_records.extend(res["sample_records"])
        
    # Aggregate metrics across 5 folds
    test_accs = [r["test_accuracy"] for r in fold_results]
    test_f1s = [r["test_macro_f1"] for r in fold_results]
    uncal_eces = [r["uncalibrated_ece"] for r in fold_results]
    cal_eces = [r["calibrated_ece"] for r in fold_results]
    uncal_briers = [r["uncalibrated_brier"] for r in fold_results]
    cal_briers = [r["calibrated_brier"] for r in fold_results]
    
    # Compute pooled out-of-fold metrics across all 1,374 samples
    oof_y_true = np.array([r["true_label"] for r in all_oof_records])
    oof_y_pred = np.array([r["predicted_label"] for r in all_oof_records])
    pooled_acc, pooled_macro_f1, pooled_per_class = compute_metrics(oof_y_true, oof_y_pred)
    
    experiment_summary = {
        "experiment_name": f"{save_prefix}_{mode}",
        "mode": mode,
        "seed": seed,
        "n_folds": 5,
        "total_clips": len(all_oof_records),
        "mean_fold_accuracy": float(np.mean(test_accs)),
        "std_fold_accuracy": float(np.std(test_accs)),
        "mean_fold_macro_f1": float(np.mean(test_f1s)),
        "std_fold_macro_f1": float(np.std(test_f1s)),
        "pooled_oof_accuracy": float(pooled_acc),
        "pooled_oof_macro_f1": float(pooled_macro_f1),
        "pooled_per_class_f1": pooled_per_class,
        "mean_uncalibrated_ece": float(np.mean(uncal_eces)),
        "mean_calibrated_ece": float(np.mean(cal_eces)),
        "mean_uncalibrated_brier": float(np.mean(uncal_briers)),
        "mean_calibrated_brier": float(np.mean(cal_briers)),
        "fold_details": [
            {
                "fold": r["fold"],
                "best_epoch": r["best_epoch"],
                "optimal_temp": r["optimal_temperature"],
                "val_acc": r["val_accuracy"],
                "val_f1": r["val_macro_f1"],
                "test_acc": r["test_accuracy"],
                "test_f1": r["test_macro_f1"],
                "uncal_ece": r["uncalibrated_ece"],
                "cal_ece": r["calibrated_ece"]
            }
            for r in fold_results
        ]
    }
    
    # Save results to metrics directory
    metrics_dir = ROOT / "novel_contribution/results/metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_json = metrics_dir / f"{save_prefix}_{mode}_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(experiment_summary, f, indent=2)
        
    out_csv = metrics_dir / f"{save_prefix}_{mode}_oof.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_oof_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_oof_records)
        
    print(f"\n=======================================================")
    print(f"COMPLETED: {save_prefix}_{mode}")
    print(f"Mean Fold Accuracy: {experiment_summary['mean_fold_accuracy']*100:.2f}% ± {experiment_summary['std_fold_accuracy']*100:.2f}%")
    print(f"Mean Fold Macro-F1: {experiment_summary['mean_fold_macro_f1']:.4f} ± {experiment_summary['std_fold_macro_f1']:.4f}")
    print(f"Pooled OOF Accuracy: {pooled_acc*100:.2f}% | Pooled Macro-F1: {pooled_macro_f1:.4f}")
    print(f"ECE: {experiment_summary['mean_uncalibrated_ece']:.4f} -> {experiment_summary['mean_calibrated_ece']:.4f} (Calibrated)")
    print(f"Saved results to: {out_json}")
    print(f"=======================================================\n")
    
    return experiment_summary


if __name__ == "__main__":
    mode_arg = sys.argv[1] if len(sys.argv) > 1 else "full"
    run_5fold_experiment(mode=mode_arg)
