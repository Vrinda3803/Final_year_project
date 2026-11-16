"""
Stage 4 Robustness Benchmarking Suite:
Reliability-Conditioned Dynamic Bilateral Graph Learning (RCDB-STGT).

Evaluates models under 6 realistic sensor degradation conditions:
  1. Clean Input (0% noise)
  2. 20% Random Joint Missingness (Q=0, X=0)
  3. 40% Random Joint Missingness (Q=0, X=0)
  4. Bilateral Wrist Occlusion (dropping joints 4 & 7)
  5. Bilateral Ankle Occlusion (dropping joints 11 & 14)
  6. Consecutive-Frame Loss (15-frame temporal blackout, frames 20..35)

Compares:
  - A0: Fixed Bilateral Topology (E2 Reproduction)
  - A1: Dynamic Bilateral Graph without Reliability Gating
  - A3: Full RCDB-STGT (Main Proposed Model)
"""

from __future__ import annotations
import sys, os, json, time, argparse
from pathlib import Path

ROOT = Path("d:/Documents/Autism_Project")
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
from research_stage4.experiments.run_stage4_ablations import (
    build_model_for_experiment,
    compute_metrics
)


def corrupt_dataset(dataset: SkeletonDataset, mode: str, seed: int = 42) -> SkeletonDataset:
    rng = np.random.default_rng(seed)
    X = dataset.X.clone()
    Q = dataset.Q.clone()
    
    if mode == "clean":
        pass
    elif mode == "random_20":
        mask = rng.random(size=Q.shape) < 0.20
        Q[mask] = 0.0
        X[torch.tensor(mask).unsqueeze(-1).expand_as(X)] = 0.0
    elif mode == "random_40":
        mask = rng.random(size=Q.shape) < 0.40
        Q[mask] = 0.0
        X[torch.tensor(mask).unsqueeze(-1).expand_as(X)] = 0.0
    elif mode == "wrist_occlusion":
        # RWrist: 4, LWrist: 7
        Q[:, :, 4] = 0.0
        Q[:, :, 7] = 0.0
        X[:, :, 4, :] = 0.0
        X[:, :, 7, :] = 0.0
    elif mode == "ankle_occlusion":
        # RAnkle: 11, LAnkle: 14
        Q[:, :, 11] = 0.0
        Q[:, :, 14] = 0.0
        X[:, :, 11, :] = 0.0
        X[:, :, 14, :] = 0.0
    elif mode == "consecutive_15":
        # Drop 15 consecutive frames out of 64
        t_start, t_end = 20, 35
        Q[:, t_start:t_end, :] = 0.0
        X[:, t_start:t_end, :, :] = 0.0
    else:
        raise ValueError(f"Unknown corruption mode: {mode}")
        
    return SkeletonDataset(X.numpy(), Q.numpy(), dataset.y.numpy(), dataset.groups, dataset.clips)


def evaluate_model_on_corruptions(
    exp_name: str,
    data: dict,
    folds: list,
    device: torch.device
) -> dict:
    ckpt_dir = ROOT / "research_stage4" / "checkpoints" / exp_name
    modes = ["clean", "random_20", "random_40", "wrist_occlusion", "ankle_occlusion", "consecutive_15"]
    
    results_by_mode = {m: {"fold_accs": [], "fold_f1s": []} for m in modes}
    
    for fold, (train_idx, test_idx) in enumerate(folds):
        ckpt_path = ckpt_dir / f"fold_{fold}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {exp_name} fold {fold}: {ckpt_path}")
            
        fit_d, val_d, test_d, _ = partition_train_val_test(data, train_idx, test_idx, seed=42 + fold)
        
        model, _ = build_model_for_experiment(exp_name, device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        
        # Fit velocity priors on fit partition coordinates
        if hasattr(model, "reliability_estimator"):
            model.reliability_estimator.fit_velocity_priors(fit_d["dataset"].X, fit_d["dataset"].Q)
            
        temp = ckpt.get("temperature", 1.0)
        model.eval()
        
        for mode in modes:
            corrupted_test_ds = corrupt_dataset(test_d["dataset"], mode=mode, seed=42 + fold)
            loader = DataLoader(corrupted_test_ds, batch_size=32, shuffle=False)
            
            all_preds, all_targets = [], []
            with torch.no_grad():
                for x_b, q_b, y_b, _, _ in loader:
                    x_b, q_b = x_b.to(device), q_b.to(device)
                    logits = model(x_b, q_b)
                    probs = torch.softmax(logits / temp, dim=-1).cpu().numpy()
                    preds = np.argmax(probs, axis=-1)
                    all_preds.append(preds)
                    all_targets.append(y_b.numpy())
                    
            y_pred = np.concatenate(all_preds)
            y_true = np.concatenate(all_targets)
            
            acc, f1, _ = compute_metrics(y_true, y_pred)
            results_by_mode[mode]["fold_accs"].append(acc)
            results_by_mode[mode]["fold_f1s"].append(f1)
            
    summary = {}
    for mode in modes:
        accs = results_by_mode[mode]["fold_accs"]
        f1s = results_by_mode[mode]["fold_f1s"]
        summary[mode] = {
            "mean_acc": float(np.mean(accs)),
            "std_acc": float(np.std(accs)),
            "mean_macro_f1": float(np.mean(f1s)),
            "std_macro_f1": float(np.std(f1s)),
            "fold_accs": accs,
            "fold_f1s": f1s
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Stage 4 Robustness Benchmark")
    parser.add_argument("--models", nargs="+", default=["a0_fixed_topo_repro", "a1_dynamic_no_reliability", "a3_rcdb_full_main"])
    args = parser.parse_args()
    
    device = torch.device("cpu")
    print("Loading MMASD dataset for Stage 4 Robustness Stress Benchmark...")
    data = load_raw_mmasd()
    folds = create_subject_independent_folds(data["groups"], n_folds=5, seed=42)
    
    robustness_master = {}
    for exp_name in args.models:
        print(f"\n=======================================================")
        print(f"BENCHMARKING ROBUSTNESS: {exp_name.upper()}")
        print(f"=======================================================")
        res = evaluate_model_on_corruptions(exp_name, data, folds, device)
        robustness_master[exp_name] = res
        
        for mode, m in res.items():
            print(f"  {mode:<20} | Acc: {m['mean_acc']*100:.2f}% ± {m['std_acc']*100:.2f}% | F1: {m['mean_macro_f1']:.4f} ± {m['std_macro_f1']:.4f}")
            
    out_dir = ROOT / "research_stage4" / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "stage4_robustness_benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(robustness_master, f, indent=2)
    print(f"\nRobustness benchmark successfully saved to: {out_file}")


if __name__ == "__main__":
    main()
