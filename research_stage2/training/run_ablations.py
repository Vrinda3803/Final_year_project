"""
Controlled Single-Factor Ablation Runner for PR-STGT (Stage 2).
Executes E2 (Topology), E3 (Phase), E4 (Margin), and E5 (Full PR-STGT) under identical 5-fold cross-validation.
"""
from __future__ import annotations
import sys, os, copy, json, time, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".codex-deps"))
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from novel_contribution.data_processing.dataset_loader import (
    load_raw_mmasd,
    create_subject_independent_folds,
    partition_train_val_test,
    CLASSES
)
from research_stage2.models.pr_stgt import PRSTGT
from research_stage2.models.losses import PRSTGTLoss
from novel_contribution.uncertainty_estimation.temperature_scaling import TemperatureScaler
from novel_contribution.uncertainty_estimation.calibration_metrics import compute_calibration_metrics
from novel_contribution.uncertainty_estimation.selective_prediction import evaluate_selective_prediction


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, dict[str, float]]:
    acc = float(np.mean(y_true == y_pred))
    per_class = {}
    f1_list = []
    for c, c_name in enumerate(CLASSES):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = 2 * tp + fp + fn
        f1 = float(2 * tp / denom) if denom > 0 else 0.0
        per_class[c_name] = f1
        f1_list.append(f1)
    macro_f1 = float(np.mean(f1_list))
    return acc, macro_f1, per_class


def run_evaluation_pass(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_logits, all_targets = [], []
    all_groups, all_clips = [], []
    with torch.no_grad():
        for x, q, y, grp, clip in loader:
            x, q = x.to(device), q.to(device)
            logits = model(x, q)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(y.numpy())
            all_groups.extend(grp)
            all_clips.extend(clip)
    logits_arr = np.concatenate(all_logits, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)
    preds_arr = np.argmax(logits_arr, axis=1)
    return logits_arr, targets_arr, preds_arr, all_groups, all_clips


def train_single_fold(
    variant: str,
    use_margin_loss: bool,
    fit_data: dict,
    val_data: dict,
    test_data: dict,
    fold: int,
    device: torch.device,
    epochs: int = 35,
    patience: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4
) -> dict:
    fit_loader = DataLoader(fit_data["dataset"], batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data["dataset"], batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data["dataset"], batch_size=32, shuffle=False)
    
    # Initialize model
    model = PRSTGT(
        in_channels=4,
        num_classes=10,
        d_model=64,
        num_spatial_heads=2,
        num_temporal_heads=2,
        dropout=0.25,
        variant=variant
    ).to(device)
    
    # Fit velocity priors
    model.reliability_estimator.fit_velocity_priors(fit_data["dataset"].X, fit_data["dataset"].Q)
    
    # Loss objective
    fit_labels = fit_data["dataset"].y.numpy()
    class_counts = np.bincount(fit_labels, minlength=len(CLASSES))
    criterion = PRSTGTLoss(
        class_counts=class_counts,
        gamma=1.2,
        use_margin_loss=use_margin_loss,
        margin=0.5,
        lambda_margin=0.25
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    best_val_loss = float("inf")
    best_val_f1 = 0.0
    best_model_state = None
    epochs_no_improve = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        for x_b, q_b, y_b, _, _ in fit_loader:
            x_b, q_b, y_b = x_b.to(device), q_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(x_b, q_b)
            loss, _ = criterion(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        scheduler.step()
        
        # Validation pass
        val_logits, val_targets, val_preds, _, _ = run_evaluation_pass(model, val_loader, device)
        _, val_f1, _ = compute_metrics(val_targets, val_preds)
        val_loss, _ = criterion(torch.tensor(val_logits, device=device), torch.tensor(val_targets, device=device))
        val_loss_val = float(val_loss.item())
        
        # Early stopping on validation macro-F1 and loss
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_loss = val_loss_val
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
                
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    # Calibration via Temperature Scaling
    scaler = TemperatureScaler()
    val_logits_raw, val_targets, _, _, _ = run_evaluation_pass(model, val_loader, device)
    scaler.fit(val_logits_raw, val_targets)
    model.temperature.data.fill_(float(scaler.temperature.item()))
    
    # Test evaluation
    test_logits, test_targets, test_preds, test_groups, test_clips = run_evaluation_pass(model, test_loader, device)
    test_probs = torch.softmax(torch.tensor(test_logits), dim=-1).numpy()
    test_acc, test_macro_f1, test_per_class = compute_metrics(test_targets, test_preds)
    
    cal_metrics = compute_calibration_metrics(test_probs, test_targets, n_bins=10)
    
    return {
        "model_state": best_model_state,
        "temperature": float(scaler.temperature.item()),
        "test_acc": test_acc,
        "test_macro_f1": test_macro_f1,
        "test_per_class": test_per_class,
        "ece": cal_metrics["ece"],
        "mce": cal_metrics["mce"],
        "brier_score": cal_metrics["brier_score"],
        "test_logits": test_logits,
        "test_probs": test_probs,
        "test_targets": test_targets,
        "test_preds": test_preds,
        "test_groups": test_groups,
        "test_clips": test_clips
    }


def run_experiment(exp_name: str, variant: str, use_margin: bool, data: dict, folds: list, device: torch.device):
    print(f"\n{'='*70}\nSTARTING EXPERIMENT: {exp_name.upper()} (Variant: {variant}, Margin: {use_margin})\n{'='*70}")
    t0 = time.time()
    
    fold_accs = []
    fold_f1s = []
    fold_eces = []
    
    oof_rows = []
    
    out_dir = ROOT / "research_stage2" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints" / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    for fold, (train_idx, test_idx) in enumerate(folds):
        f_t0 = time.time()
        fit_d, val_d, test_d, _ = partition_train_val_test(data, train_idx, test_idx, seed=42 + fold)
        
        res = train_single_fold(
            variant=variant,
            use_margin_loss=use_margin,
            fit_data=fit_d,
            val_data=val_d,
            test_data=test_d,
            fold=fold,
            device=device,
            epochs=35,
            patience=10
        )
        
        # Save fold checkpoint
        torch.save({
            "fold": fold,
            "variant": variant,
            "model_state": res["model_state"],
            "temperature": res["temperature"],
            "test_acc": res["test_acc"],
            "test_macro_f1": res["test_macro_f1"]
        }, ckpt_dir / f"fold_{fold}.pt")
        
        fold_accs.append(res["test_acc"])
        fold_f1s.append(res["test_macro_f1"])
        fold_eces.append(res["ece"])
        
        # Collect OOF predictions
        for i in range(len(res["test_targets"])):
            clip_name = res["test_clips"][i]
            grp = res["test_groups"][i]
            y_t = int(res["test_targets"][i])
            y_p = int(res["test_preds"][i])
            probs = res["test_probs"][i]
            conf = float(np.max(probs))
            
            row = [clip_name, grp, fold, y_t, y_p, conf] + [float(p) for p in probs]
            oof_rows.append(row)
            
        f_dur = time.time() - f_t0
        print(f"Fold {fold}: Acc = {res['test_acc']*100:.2f}%, Macro-F1 = {res['test_macro_f1']:.4f}, ECE = {res['ece']:.4f} ({f_dur:.1f}s)")

    # Save OOF CSV
    oof_csv_path = out_dir / f"{exp_name}_oof.csv"
    prob_cols = [f"prob_{c}" for c in CLASSES]
    header = "clip,group,fold,y_true,y_pred,confidence," + ",".join(prob_cols) + "\n"
    with open(oof_csv_path, "w", encoding="utf-8") as f:
        f.write(header)
        for r in oof_rows:
            line = f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]:.4f}," + ",".join(f"{p:.4f}" for p in r[6:]) + "\n"
            f.write(line)

    # Pooled metrics
    all_y_true = np.array([r[3] for r in oof_rows])
    all_y_pred = np.array([r[4] for r in oof_rows])
    pooled_acc, pooled_macro_f1, pooled_per_class = compute_metrics(all_y_true, all_y_pred)
    
    total_time = time.time() - t0
    mean_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs))
    mean_f1 = float(np.mean(fold_f1s))
    std_f1 = float(np.std(fold_f1s))
    mean_ece = float(np.mean(fold_eces))
    
    summary = {
        "experiment": exp_name,
        "variant": variant,
        "use_margin": use_margin,
        "mean_fold_acc": mean_acc,
        "std_fold_acc": std_acc,
        "mean_fold_macro_f1": mean_f1,
        "std_fold_macro_f1": std_f1,
        "pooled_oof_acc": pooled_acc,
        "pooled_oof_macro_f1": pooled_macro_f1,
        "mean_ece": mean_ece,
        "per_class_f1": pooled_per_class,
        "fold_accs": fold_accs,
        "fold_f1s": fold_f1s,
        "fold_eces": fold_eces,
        "total_runtime_seconds": total_time
    }
    
    summary_path = out_dir / f"{exp_name}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\nSUMMARY {exp_name.upper()}:")
    print(f"Mean Fold Accuracy: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"Mean Fold Macro-F1: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Pooled OOF Macro-F1: {pooled_macro_f1:.4f}")
    print(f"Total Time: {total_time:.1f}s")
    print(f"{'='*70}\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="all", choices=["all", "e2", "e3", "e4", "e5"])
    args = parser.parse_args()
    
    device = torch.device("cpu")
    print("Loading MMASD dataset...")
    data = load_raw_mmasd()
    print(f"Loaded {len(data['y'])} samples across {len(np.unique(data['groups']))} groups.")
    
    folds = create_subject_independent_folds(data["groups"], n_folds=5, seed=42)
    
    experiments = [
        ("e2_topology_only", "e2_topology_only", False),
        ("e3_phase_only", "e3_phase_only", False),
        ("e4_margin_only", "e4_margin_only", True),
        ("e5_full_pr_stgt", "e5_full_pr_stgt", True)
    ]
    
    if args.exp != "all":
        experiments = [e for e in experiments if e[0].startswith(args.exp)]
        
    all_summaries = {}
    for exp_name, variant, use_margin in experiments:
        summ = run_experiment(exp_name, variant, use_margin, data, folds, device)
        all_summaries[exp_name] = summ
        
    # Write overall ablation comparison table
    comp_path = ROOT / "research_stage2" / "experiments" / "ablation_comparison.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"All ablations completed! Saved comparison to {comp_path}")


if __name__ == "__main__":
    main()
