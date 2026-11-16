"""
Stage 4 Experiment & Ablation Runner:
Reliability-Conditioned Dynamic Bilateral Graph Learning (RCDB-STGT).

Executes the full Stage 4 ablation matrix (A0 through A9) under strict 5-fold GroupKFold cross-validation.
"""

from __future__ import annotations
import sys, os, copy, json, time, argparse
from pathlib import Path

ROOT = Path("d:/Documents/Autism_Project")
sys.path.insert(0, str(ROOT / ".codex-deps"))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from novel_contribution.data_processing.dataset_loader import (
    load_raw_mmasd,
    create_subject_independent_folds,
    partition_train_val_test,
    CLASSES
)
from novel_contribution.uncertainty_estimation.temperature_scaling import TemperatureScaler
from novel_contribution.uncertainty_estimation.calibration_metrics import compute_calibration_metrics
from research_stage2.models.losses import PRSTGTLoss
from research_stage3.models.pr_stgt_stage3 import PRSTGTStage3
from research_stage4.models.rcdb_stgt import RCDB_STGT, count_parameters


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


def build_model_for_experiment(exp_name: str, device: torch.device) -> tuple[nn.Module, bool]:
    """
    Instantiates the model and loss flag corresponding to the ablation ID.
    Returns: (model, use_margin_loss)
    """
    use_margin = False
    
    if exp_name == "a0_fixed_topo_repro":
        # A0: Exactly reproduces E2 (Fixed topology with relational reliability)
        model = PRSTGTStage3(
            in_channels=4,
            num_classes=10,
            d_model=64,
            num_spatial_heads=2,
            num_temporal_heads=2,
            dropout=0.25,
            use_relational_edges=True,
            use_relational_reliability=True,
            use_phase_stream=False
        )
    elif exp_name == "a1_dynamic_no_reliability":
        # A1: Dynamic bilateral graph without reliability conditioning (q_uv = 1)
        model = RCDB_STGT(
            in_channels=4,
            num_classes=10,
            d_model=64,
            use_dynamic_edges=True,
            use_reliability_gate=False,
            use_temporal_smoothing=True,
            use_phase_stream=False
        )
    elif exp_name == "a2_reliability_no_dynamic":
        # A2: Reliability-conditioned bilateral graph without dynamic movement context
        model = PRSTGTStage3(
            in_channels=4,
            num_classes=10,
            d_model=64,
            dropout=0.25,
            use_relational_edges=True,
            use_relational_reliability=True,
            use_phase_stream=False
        )
    elif exp_name == "a3_rcdb_full_main":
        # A3: Main Proposed RCDB-STGT (Dynamic + Reliability-Conditioned + Temporal Smoothing)
        model = RCDB_STGT(
            in_channels=4,
            num_classes=10,
            d_model=64,
            use_dynamic_edges=True,
            use_reliability_gate=True,
            use_temporal_smoothing=True,
            use_phase_stream=False
        )
    elif exp_name == "a4_rcdb_no_temp_smooth":
        # A4: A3 without temporal edge smoothing (purely instantaneous per-frame a_uv)
        model = RCDB_STGT(
            in_channels=4,
            num_classes=10,
            d_model=64,
            use_dynamic_edges=True,
            use_reliability_gate=True,
            use_temporal_smoothing=False,
            use_phase_stream=False
        )
    elif exp_name == "a5_rcdb_with_phase":
        # A5: A3 with harmonic phase velocity stream
        model = RCDB_STGT(
            in_channels=4,
            num_classes=10,
            d_model=64,
            use_dynamic_edges=True,
            use_reliability_gate=True,
            use_temporal_smoothing=True,
            use_phase_stream=True
        )
    elif exp_name == "a6_rcdb_wrist_only":
        # A6: A3 with wrist-only bilateral edges [(4, 7)]
        model = RCDB_STGT(
            in_channels=4,
            num_classes=10,
            d_model=64,
            bilateral_pairs=[(4, 7)],
            use_dynamic_edges=True,
            use_reliability_gate=True,
            use_temporal_smoothing=True,
            use_phase_stream=False
        )
    elif exp_name == "a7_rcdb_multi_pair":
        # A7: A3 with multiple bilateral joint pairs (wrists, ankles, elbows, knees, hips)
        multi_pairs = [(4, 7), (11, 14), (3, 6), (10, 13), (9, 12)]
        model = RCDB_STGT(
            in_channels=4,
            num_classes=10,
            d_model=64,
            bilateral_pairs=multi_pairs,
            use_dynamic_edges=True,
            use_reliability_gate=True,
            use_temporal_smoothing=True,
            use_phase_stream=False
        )
    elif exp_name == "a8_rcdb_input_rel_only":
        # A8: A3 without explicit multiplicative gate (reliability fed to MLP only)
        model = RCDB_STGT(
            in_channels=4,
            num_classes=10,
            d_model=64,
            use_dynamic_edges=True,
            use_reliability_gate=False,
            use_temporal_smoothing=True,
            use_phase_stream=False
        )
    elif exp_name == "a9_rcdb_with_margin":
        # A9: A3 with confusable-pair margin loss
        model = RCDB_STGT(
            in_channels=4,
            num_classes=10,
            d_model=64,
            use_dynamic_edges=True,
            use_reliability_gate=True,
            use_temporal_smoothing=True,
            use_phase_stream=False
        )
        use_margin = True
    else:
        raise ValueError(f"Unknown experiment name: {exp_name}")
        
    return model.to(device), use_margin


def train_single_fold(
    exp_name: str,
    fit_data: dict,
    val_data: dict,
    test_data: dict,
    fold: int,
    device: torch.device,
    seed: int = 42,
    epochs: int = 35,
    patience: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4
) -> dict:
    torch.manual_seed(seed + fold)
    np.random.seed(seed + fold)
    
    fit_loader = DataLoader(fit_data["dataset"], batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data["dataset"], batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data["dataset"], batch_size=32, shuffle=False)
    
    model, use_margin = build_model_for_experiment(exp_name, device)
    
    # Fit velocity priors on fit partition coordinates
    model.reliability_estimator.fit_velocity_priors(fit_data["dataset"].X, fit_data["dataset"].Q)
    
    # Loss objective
    fit_labels = fit_data["dataset"].y.numpy()
    class_counts = np.bincount(fit_labels, minlength=len(CLASSES))
    criterion = PRSTGTLoss(
        class_counts=class_counts,
        gamma=1.2,
        use_margin_loss=use_margin,
        margin=0.5,
        lambda_margin=0.25
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
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
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Fold {fold} Early stopped at epoch {epoch} (Best Val F1: {best_val_f1:.4f})", flush=True)
                break
                
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Fold {fold} Epoch {epoch}/{epochs} | Val F1: {val_f1:.4f} (Best: {best_val_f1:.4f})", flush=True)
                
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    # Temperature scaling calibration
    scaler = TemperatureScaler()
    val_logits_raw, val_targets, _, _, _ = run_evaluation_pass(model, val_loader, device)
    scaler.fit(val_logits_raw, val_targets)
    temp_val = float(scaler.temperature.item())
    model.temperature.data.fill_(temp_val)
    
    # Final evaluation on outer held-out test fold
    test_logits, test_targets, test_preds, test_groups, test_clips = run_evaluation_pass(model, test_loader, device)
    cal_test_logits = test_logits / temp_val
    test_probs = torch.softmax(torch.tensor(cal_test_logits), dim=-1).numpy()
    test_preds_cal = np.argmax(test_probs, axis=1)
    
    test_acc, test_macro_f1, test_per_class = compute_metrics(test_targets, test_preds_cal)
    cal_metrics = compute_calibration_metrics(test_probs, test_targets, n_bins=10)
    
    return {
        "model_state": best_model_state,
        "temperature": temp_val,
        "test_acc": test_acc,
        "test_macro_f1": test_macro_f1,
        "test_per_class": test_per_class,
        "ece": cal_metrics["ece"],
        "brier_score": cal_metrics["brier_score"],
        "test_probs": test_probs,
        "test_targets": test_targets,
        "test_preds": test_preds_cal,
        "test_groups": test_groups,
        "test_clips": test_clips
    }


def run_experiment(exp_name: str, data: dict, folds: list, device: torch.device, seed: int = 42) -> dict:
    print(f"\n{'='*75}\nSTARTING STAGE 4 EXPERIMENT: {exp_name.upper()} (Seed: {seed})\n{'='*75}", flush=True)
    t0 = time.time()
    
    out_dir = ROOT / "research_stage4"
    ckpt_dir = out_dir / "checkpoints" / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    fold_accs = []
    fold_f1s = []
    fold_eces = []
    oof_rows = []
    
    for fold, (train_idx, test_idx) in enumerate(folds):
        f_t0 = time.time()
        fit_d, val_d, test_d, _ = partition_train_val_test(data, train_idx, test_idx, seed=seed + fold)
        
        res = train_single_fold(
            exp_name=exp_name,
            fit_data=fit_d,
            val_data=val_d,
            test_data=test_d,
            fold=fold,
            device=device,
            seed=seed,
            epochs=35,
            patience=10
        )
        
        # Save fold checkpoint
        torch.save({
            "fold": fold,
            "seed": seed,
            "experiment": exp_name,
            "model_state": res["model_state"],
            "temperature": res["temperature"],
            "test_acc": res["test_acc"],
            "test_macro_f1": res["test_macro_f1"]
        }, ckpt_dir / f"fold_{fold}.pt")
        
        fold_accs.append(res["test_acc"])
        fold_f1s.append(res["test_macro_f1"])
        fold_eces.append(res["ece"])
        
        for i in range(len(res["test_targets"])):
            clip_name = res["test_clips"][i]
            grp = res["test_groups"][i]
            y_t = int(res["test_targets"][i])
            y_p = int(res["test_preds"][i])
            probs = res["test_probs"][i]
            conf = float(np.max(probs))
            row = [clip_name, grp, fold, y_t, y_p, conf] + [float(p) for p in probs]
            oof_rows.append(row)
            
        dur = time.time() - f_t0
        print(f"Fold {fold}: Acc = {res['test_acc']*100:.2f}%, Macro-F1 = {res['test_macro_f1']:.4f}, ECE = {res['ece']:.4f} ({dur:.1f}s)", flush=True)
        
    # Save OOF CSV
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    oof_csv_path = pred_dir / f"{exp_name}_oof.csv"
    
    prob_cols = [f"prob_{c}" for c in CLASSES]
    header = "clip,group,fold,y_true,y_pred,confidence," + ",".join(prob_cols) + "\n"
    with open(oof_csv_path, "w", encoding="utf-8") as f:
        f.write(header)
        for r in oof_rows:
            line = f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]:.4f}," + ",".join(f"{p:.4f}" for p in r[6:]) + "\n"
            f.write(line)
            
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
        "seed": seed,
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
    
    metric_dir = out_dir / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)
    with open(metric_dir / f"{exp_name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\nSUMMARY {exp_name.upper()}:")
    print(f"Mean Fold Accuracy: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"Mean Fold Macro-F1: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Pooled OOF Macro-F1: {pooled_macro_f1:.4f}")
    print(f"Total Time: {total_time:.1f}s")
    print(f"{'='*75}\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Stage 4 Ablation Runner")
    parser.add_argument("--exp", type=str, default="a3", help="Experiment name or 'all'")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    args = parser.parse_args()
    
    device = torch.device("cpu")
    print("Loading MMASD dataset for Stage 4 Research Experiments...")
    data = load_raw_mmasd()
    print(f"Loaded {len(data['y'])} samples across {len(np.unique(data['groups']))} cohorts.")
    
    folds = create_subject_independent_folds(data["groups"], n_folds=5, seed=42)
    
    all_experiments = [
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
    
    if args.exp == "all":
        exps_to_run = all_experiments
    else:
        # Match prefix or exact name
        matched = [e for e in all_experiments if e.startswith(args.exp)]
        if not matched:
            raise ValueError(f"No experiment matches '{args.exp}'. Choose from: {all_experiments}")
        exps_to_run = matched
        
    master_summaries = {}
    for exp_name in exps_to_run:
        summary = run_experiment(exp_name, data, folds, device, seed=args.seed)
        master_summaries[exp_name] = summary
        
    master_path = ROOT / "research_stage4" / "metrics" / "stage4_ablation_master_summary.json"
    existing = {}
    if master_path.exists():
        try:
            with open(master_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    existing.update(master_summaries)
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    print(f"Updated master summary at: {master_path}")


if __name__ == "__main__":
    main()
