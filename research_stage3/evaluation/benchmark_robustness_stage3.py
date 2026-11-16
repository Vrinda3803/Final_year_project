"""
Stage 3 Realistic Pose Uncertainty and Robustness Benchmarking.
Evaluates E0 (Vanilla STGT), E1 (RA-STGT v1), E2 (Topology Only), and E5 (Full PR-STGT)
under:
  1. Clean Input (0%)
  2. 20% Random Joint Missingness
  3. 40% Random Joint Missingness
  4. Bilateral Wrist Occlusion (dropping joints 4 & 7)
  5. Bilateral Ankle Occlusion (dropping joints 11 & 14)
  6. Consecutive-Frame Loss (15-frame temporal dropout)
"""
from __future__ import annotations
import sys, json
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
from novel_contribution.models.ra_stgt import RASTGT
from research_stage2.models.pr_stgt import PRSTGT


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    f1_list = []
    for c in range(len(CLASSES)):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = 2 * tp + fp + fn
        f1 = float(2 * tp / denom) if denom > 0 else 0.0
        f1_list.append(f1)
    return float(np.mean(f1_list))


def run_eval(model: torch.nn.Module, dataset: SkeletonDataset, device: torch.device):
    model.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, q, y, _, _ in loader:
            x, q = x.to(device), q.to(device)
            logits = model(x, q)
            all_preds.append(torch.argmax(logits, dim=-1).cpu().numpy())
            all_targets.append(y.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    acc = float(np.mean(y_true == y_pred))
    f1 = compute_macro_f1(y_true, y_pred)
    return acc, f1


def corrupt_dataset(dataset: SkeletonDataset, mode: str, seed: int = 42) -> SkeletonDataset:
    rng = np.random.default_rng(seed)
    X = dataset.X.clone()
    Q = dataset.Q.clone()
    
    if mode == "random_20":
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
        t_start = 20
        t_end = 35
        Q[:, t_start:t_end, :] = 0.0
        X[:, t_start:t_end, :, :] = 0.0
    elif mode == "clean":
        pass
    else:
        raise ValueError(f"Unknown mode: {mode}")
        
    return SkeletonDataset(X.numpy(), Q.numpy(), dataset.y.numpy(), dataset.groups, dataset.clips)


def evaluate_model_robustness(model_type: str, ckpt_dir: Path, data: dict, folds: list, device: torch.device):
    modes = ["clean", "random_20", "random_40", "wrist_occlusion", "ankle_occlusion", "consecutive_15"]
    mode_results = {m: {"accs": [], "f1s": []} for m in modes}
    
    for fold, (train_idx, test_idx) in enumerate(folds):
        fit_d, val_d, test_d, _ = partition_train_val_test(data, train_idx, test_idx, seed=42 + fold)
        
        # Load model
        if model_type == "e0_baseline":
            model = RASTGT(mode="baseline_stgt").to(device)
            ckpt_path = ROOT / "novel_contribution" / "results" / "checkpoints" / "baseline_stgt" / f"fold_{fold}.pt"
        elif model_type == "e1_ra_stgt":
            model = RASTGT(mode="full").to(device)
            ckpt_path = ROOT / "novel_contribution" / "results" / "checkpoints" / "ra_stgt_full" / f"fold_{fold}.pt"
        elif model_type == "e2_topology":
            model = PRSTGT(variant="e2_topology_only").to(device)
            ckpt_path = ROOT / "research_stage2" / "experiments" / "checkpoints" / "e2_topology_only" / f"fold_{fold}.pt"
        elif model_type == "e5_full":
            model = PRSTGT(variant="e5_full_pr_stgt").to(device)
            ckpt_path = ROOT / "research_stage2" / "experiments" / "checkpoints" / "e5_full_pr_stgt" / f"fold_{fold}.pt"
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
        if not ckpt_path.exists():
            print(f"Skipping {model_type} fold {fold}: checkpoint not found ({ckpt_path})")
            continue
            
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
        model.load_state_dict(state_dict)
        if hasattr(model, "reliability_estimator"):
            model.reliability_estimator.fit_velocity_priors(fit_d["dataset"].X, fit_d["dataset"].Q)
            
        for m in modes:
            corrupt_ds = corrupt_dataset(test_d["dataset"], mode=m, seed=42 + fold)
            acc, f1 = run_eval(model, corrupt_ds, device)
            mode_results[m]["accs"].append(acc)
            mode_results[m]["f1s"].append(f1)
            
    summary = {}
    for m in modes:
        if len(mode_results[m]["f1s"]) > 0:
            summary[m] = {
                "mean_acc": float(np.mean(mode_results[m]["accs"])),
                "std_acc": float(np.std(mode_results[m]["accs"])),
                "mean_f1": float(np.mean(mode_results[m]["f1s"])),
                "std_f1": float(np.std(mode_results[m]["f1s"]))
            }
    return summary


def main():
    device = torch.device("cpu")
    print("Loading MMASD data for Robustness Benchmarking...")
    data = load_raw_mmasd()
    folds = create_subject_independent_folds(data["groups"], n_folds=5, seed=42)
    
    models = [
        ("e0_baseline", ROOT / "novel_contribution" / "results" / "checkpoints" / "baseline_stgt"),
        ("e1_ra_stgt", ROOT / "novel_contribution" / "results" / "checkpoints" / "ra_stgt_full"),
        ("e2_topology", ROOT / "research_stage2" / "experiments" / "checkpoints" / "e2_topology_only"),
        ("e5_full", ROOT / "research_stage2" / "experiments" / "checkpoints" / "e5_full_pr_stgt")
    ]
    
    all_robustness = {}
    for m_type, p in models:
        print(f"\nBenchmarking Robustness for: {m_type}")
        res = evaluate_model_robustness(m_type, p, data, folds, device)
        all_robustness[m_type] = res
        for mode_name, metrics in res.items():
            print(f"  {mode_name:<18} -> Acc: {metrics['mean_acc']*100:.2f}%, F1: {metrics['mean_f1']:.4f}")
            
    out_path = ROOT / "research_stage3" / "experiments" / "robustness_stress_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_robustness, f, indent=2)
    print(f"\nSaved Stage 3 Robustness Stress Benchmark to: {out_path}")


if __name__ == "__main__":
    main()
