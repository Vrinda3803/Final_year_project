"""
Training loop and evaluation controller for a single fold.
"""
from __future__ import annotations
import sys, copy, json
from pathlib import Path
from typing import Dict, Any, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".codex-deps"))
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from novel_contribution.training.loss_functions import ClassBalancedFocalLoss
from novel_contribution.uncertainty_estimation.temperature_scaling import TemperatureScaler
from novel_contribution.uncertainty_estimation.calibration_metrics import compute_calibration_metrics
from novel_contribution.uncertainty_estimation.selective_prediction import evaluate_selective_prediction

CLASSES = ["as", "bs", "ce", "dr", "mfs", "ms", "sac", "sq", "tr", "tw"]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, Dict[str, float]]:
    """Computes overall accuracy, macro-F1, and per-class F1."""
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


def run_evaluation_pass(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list, list]:
    """Runs a forward pass over a dataloader to collect logits, targets, groups, and clips."""
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


def train_fold(
    model: nn.Module,
    fit_data: Dict[str, Any],
    val_data: Dict[str, Any],
    test_data: Dict[str, Any],
    fold: int,
    config: Dict[str, Any],
    device: torch.device,
    save_checkpoint: bool = True
) -> Dict[str, Any]:
    """
    Trains one fold model with early stopping, fits calibration, and evaluates on test set.
    """
    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 32)
    epochs = train_cfg.get("epochs", 30)
    patience = train_cfg.get("early_stopping_patience", 7)
    lr = train_cfg.get("learning_rate", 1e-3)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    
    # Initialize DataLoaders
    fit_loader = DataLoader(fit_data["dataset"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data["dataset"], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data["dataset"], batch_size=batch_size, shuffle=False)
    
    # Fit velocity priors on fit partition
    if hasattr(model, "reliability_estimator"):
        model.reliability_estimator.fit_velocity_priors(fit_data["dataset"].X, fit_data["dataset"].Q)
        
    model = model.to(device)
    
    # Class-balanced loss
    fit_labels = fit_data["dataset"].y.numpy()
    class_counts = np.bincount(fit_labels, minlength=len(CLASSES))
    criterion = ClassBalancedFocalLoss(class_counts, gamma=1.2).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    best_val_f1 = -1.0
    best_val_acc = -1.0
    best_model_state = None
    best_val_logits = None
    best_val_targets = None
    best_epoch = 0
    patience_counter = 0
    
    epoch_logs = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, q, y, _, _ in fit_loader:
            x, q, y = x.to(device), q.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x, q)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_loss += loss.item() * len(y)
            
        train_loss /= len(fit_data["dataset"])
        scheduler.step()
        
        # Validation pass
        val_logits, val_targets, val_preds, _, _ = run_evaluation_pass(model, val_loader, device)
        val_acc, val_f1, _ = compute_metrics(val_targets, val_preds)
        
        epoch_logs.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_accuracy": float(val_acc),
            "val_macro_f1": float(val_f1)
        })
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            best_val_logits = val_logits
            best_val_targets = val_targets
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
                
    # Restore best checkpoint
    model.load_state_dict(best_model_state)
    
    # Save checkpoint if requested
    checkpoint_path = ""
    if save_checkpoint:
        ckpt_dir = ROOT / "novel_contribution/results/checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = str(ckpt_dir / f"ra_stgt_{model.mode}_fold{fold}.pt")
        torch.save(best_model_state, checkpoint_path)
        
    # Fit temperature scaling calibration on best validation partition
    scaler = TemperatureScaler()
    opt_temp = scaler.fit(best_val_logits, best_val_targets)
    
    # Evaluate on held-out test partition
    test_logits, test_targets, test_preds_raw, test_groups, test_clips = run_evaluation_pass(model, test_loader, device)
    test_acc_raw, test_f1_raw, per_class_f1 = compute_metrics(test_targets, test_preds_raw)
    
    # Uncalibrated calibration metrics
    uncal_probs = torch.softmax(torch.tensor(test_logits), dim=-1).numpy()
    cal_metrics_uncal = compute_calibration_metrics(uncal_probs, test_targets)
    
    # Calibrated probabilities
    cal_probs = scaler.predict_proba(test_logits)
    cal_metrics_cal = compute_calibration_metrics(cal_probs, test_targets)
    
    # Selective prediction analysis on calibrated probabilities
    selective_res = evaluate_selective_prediction(cal_probs, test_targets)
    
    # Build out-of-fold sample records
    sample_records = []
    for i in range(len(test_targets)):
        sample_records.append({
            "fold": fold,
            "clip_id": test_clips[i],
            "subject_id": test_groups[i],
            "true_label": int(test_targets[i]),
            "predicted_label": int(test_preds_raw[i]),
            "confidence_uncal": float(np.max(uncal_probs[i])),
            "confidence_cal": float(np.max(cal_probs[i])),
            "is_correct": bool(test_preds_raw[i] == test_targets[i])
        })
        
    fold_result = {
        "fold": fold,
        "mode": model.mode,
        "best_epoch": best_epoch,
        "optimal_temperature": opt_temp,
        "val_accuracy": float(best_val_acc),
        "val_macro_f1": float(best_val_f1),
        "test_accuracy": float(test_acc_raw),
        "test_macro_f1": float(test_f1_raw),
        "per_class_f1": per_class_f1,
        "uncalibrated_ece": cal_metrics_uncal["ece"],
        "calibrated_ece": cal_metrics_cal["ece"],
        "uncalibrated_brier": cal_metrics_uncal["brier_score"],
        "calibrated_brier": cal_metrics_cal["brier_score"],
        "selective_prediction": selective_res,
        "sample_records": sample_records,
        "epoch_logs": epoch_logs,
        "checkpoint_path": checkpoint_path
    }
    return fold_result
