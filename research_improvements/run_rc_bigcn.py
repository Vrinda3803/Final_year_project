"""
5-Fold Cross-Validation Evaluation for RC-BiGCN
Preserving strict grouped cohort splitting, inner validation for early stopping, and out-of-fold metrics.
"""

from __future__ import annotations
import sys, os, time, json
from pathlib import Path

ROOT = Path("d:/Documents/Autism_Project")
sys.path.insert(0, str(ROOT / ".codex-deps"))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from experiments.mmasd_openpose_temporal_features import grouped_folds, metric, CLASSES
from research_improvements.rc_bigcn import RC_BiGCN

DEVICE = torch.device("cpu")
BATCH_SIZE = 32
EPOCHS = 65
PATIENCE = 15
LR = 1e-3
WEIGHT_DECAY = 1e-4

def train_eval_fold(fold_idx, X, Q, y, groups, tr, te, seed=42):
    torch.manual_seed(seed + fold_idx)
    rng = np.random.default_rng(seed + fold_idx)
    
    # Inner validation split (disjoint groups)
    ug = np.unique(groups[tr])
    rng.shuffle(ug)
    n_val = max(1, round(0.18 * len(ug)))
    val_groups = ug[:n_val]
    va_idx = tr[np.isin(groups[tr], val_groups)]
    fit_idx = tr[~np.isin(groups[tr], val_groups)]
    
    # Class weights for imbalanced cross-entropy
    counts = np.bincount(y[fit_idx], minlength=len(CLASSES)).astype(np.float32)
    weights = np.zeros_like(counts)
    valid_classes = counts > 0
    weights[valid_classes] = counts[valid_classes].sum() / (valid_classes.sum() * counts[valid_classes])
    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    
    # Convert datasets
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X[fit_idx], dtype=torch.float32),
                      torch.tensor(Q[fit_idx], dtype=torch.float32),
                      torch.tensor(y[fit_idx], dtype=torch.long)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X[va_idx], dtype=torch.float32),
                      torch.tensor(Q[va_idx], dtype=torch.float32),
                      torch.tensor(y[va_idx], dtype=torch.long)),
        batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X[te], dtype=torch.float32),
                      torch.tensor(Q[te], dtype=torch.float32),
                      torch.tensor(y[te], dtype=torch.long)),
        batch_size=BATCH_SIZE, shuffle=False
    )
    
    model = RC_BiGCN(in_channels=5, num_classes=len(CLASSES), base_channels=32, dropout=0.3).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    
    best_val_f1 = -1.0
    best_state = None
    wait = 0
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for bx, bq, by in train_loader:
            bx, bq, by = bx.to(DEVICE), bq.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(bx, bq)
            loss = F.cross_entropy(logits, by, weight=weight_tensor)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        scheduler.step()
        
        # Validation pass
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for bx, bq, by in val_loader:
                bx, bq = bx.to(DEVICE), bq.to(DEVICE)
                preds = model(bx, bq).argmax(dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(by.numpy())
                
        val_acc, val_f1 = metric(np.array(val_targets), np.array(val_preds))
        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
                
    # Evaluate best checkpoint on held-out test fold
    model.load_state_dict(best_state)
    model.eval()
    test_preds, test_targets, test_probs = [], [], []
    with torch.no_grad():
        for bx, bq, by in test_loader:
            bx, bq = bx.to(DEVICE), bq.to(DEVICE)
            logits = model(bx, bq)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)
            test_probs.append(probs)
            test_preds.extend(preds)
            test_targets.extend(by.numpy())
            
    test_probs = np.concatenate(test_probs, axis=0)
    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)
    test_acc, test_f1 = metric(test_targets, test_preds)
    
    print(f"Fold {fold_idx}: Val F1 = {best_val_f1:.4f} | Test Acc = {test_acc*100:.2f}%, Test Macro-F1 = {test_f1:.4f}")
    return test_acc, test_f1, test_preds, test_probs

def main():
    data = np.load(ROOT / "data/processed/mmasd_openpose/mmasd_openpose_64frames.npz")
    X, Q, y, groups = data['X'], data['Q'], data['y'], data['groups']
    
    print(f"Starting 5-Fold Evaluation for RC-BiGCN on {len(y)} clips across {len(np.unique(groups))} groups...")
    fold_accs, fold_f1s = [], []
    oof_preds = np.zeros_like(y)
    oof_probs = np.zeros((len(y), len(CLASSES)), dtype=np.float32)
    
    start_time = time.time()
    for fold_idx, (tr, te) in enumerate(grouped_folds(groups), 1):
        acc, f1, preds, probs = train_eval_fold(fold_idx, X, Q, y, groups, tr, te)
        fold_accs.append(acc)
        fold_f1s.append(f1)
        oof_preds[te] = preds
        oof_probs[te] = probs
        
    total_time = time.time() - start_time
    pooled_acc, pooled_f1 = metric(y, oof_preds)
    
    print("\n" + "="*60)
    print("FINAL 5-FOLD EVALUATION RESULTS FOR RC-BiGCN:")
    print(f"Mean Fold Accuracy: {np.mean(fold_accs)*100:.2f}% ± {np.std(fold_accs)*100:.2f}%")
    print(f"Mean Fold Macro-F1: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
    print(f"Pooled OOF Accuracy: {pooled_acc*100:.2f}%")
    print(f"Pooled OOF Macro-F1: {pooled_f1:.4f}")
    print(f"Total Runtime: {total_time:.1f}s")
    print("="*60)
    
    # Save results
    results = {
        "model": "RC-BiGCN",
        "mean_fold_acc": float(np.mean(fold_accs)),
        "std_fold_acc": float(np.std(fold_accs)),
        "mean_fold_macro_f1": float(np.mean(fold_f1s)),
        "std_fold_macro_f1": float(np.std(fold_f1s)),
        "pooled_oof_acc": float(pooled_acc),
        "pooled_oof_macro_f1": float(pooled_f1),
        "fold_accs": [float(a) for a in fold_accs],
        "fold_f1s": [float(f) for f in fold_f1s],
        "total_runtime": total_time
    }
    with open(ROOT / "research_improvements" / "rc_bigcn_results.json", "w") as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(ROOT / "research_improvements" / "rc_bigcn_oof.npz",
                        y_true=y, y_pred=oof_preds, probs=oof_probs)
    print("Saved results and out-of-fold predictions.")

if __name__ == "__main__":
    main()
