"""
End-to-End Optimization Suite for Autism Therapy Activity Recognition
Executes Experiments 1 through 8 systematically under 5-Fold Grouped Cross-Validation.
Preserves strict cohort non-overlap and inner validation tuning.
"""

from __future__ import annotations
import math, sys, os, time, json, csv
from pathlib import Path

ROOT = Path("d:/Documents/Autism_Project")
sys.path.insert(0, str(ROOT / ".codex-deps"))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from experiments.mmasd_openpose_temporal_features import grouped_folds, CLASSES
from research_improvements.rc_bigcn import build_adjacency_matrix, BILATERAL_PAIRS

DEVICE = torch.device("cpu")
BODY25_BONES = [
    (1, 0), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7), (1, 8),
    (8, 9), (9, 10), (10, 11), (11, 24), (11, 22), (22, 23),
    (8, 12), (12, 13), (13, 14), (14, 21), (14, 19), (19, 20),
    (0, 15), (15, 17), (0, 16), (16, 18)
]

def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == targets)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop = np.mean(in_bin)
        if prop > 0:
            ece += np.abs(np.mean(accuracies[in_bin]) - np.mean(confidences[in_bin])) * prop
    return float(ece)

class EnhancedBilateralConv(nn.Module):
    """Bilateral Graph Convolution with Soft Reliability Conditioning & Temporal Edge Regularization."""
    def __init__(self, channels: int, num_joints: int = 25, hidden_dim: int = 32):
        super().__init__()
        self.num_joints = num_joints
        self.register_buffer('A_base', build_adjacency_matrix(num_joints))
        self.A_learnable = nn.Parameter(torch.zeros(num_joints, num_joints))
        
        # Bilateral MLP with non-linear reliability conditioning: [hu, hv, qu, qv, |hu - hv|, hu * hv]
        # in_dim = channels * 4 + 2
        in_dim = channels * 4 + 2
        self.bilat_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )
        
    def forward(self, x: torch.Tensor, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, T, V = x.shape
        A_fixed = self.A_base + self.A_learnable
        x_perm = x.permute(0, 2, 3, 1).reshape(B * T, V, C)
        
        if q.shape[1] != T:
            q_resampled = F.adaptive_avg_pool2d(q.unsqueeze(1), (T, V)).squeeze(1)
        else:
            q_resampled = q
        q_perm = q_resampled.reshape(B * T, V, 1)
        
        A_bilat = torch.zeros(B * T, V, V, device=x.device, dtype=x.dtype)
        edge_weights_list = []
        for u, v in BILATERAL_PAIRS:
            hu, hv = x_perm[:, u], x_perm[:, v]
            qu, qv = q_perm[:, u], q_perm[:, v]
            diff = torch.abs(hu - hv)
            prod = hu * hv
            feat = torch.cat([hu, hv, qu, qv, diff, prod], dim=-1)
            weight = self.bilat_mlp(feat) # (B*T, 1)
            edge_weights_list.append(weight.view(B, T))
            A_bilat[:, u, v] = weight.squeeze(-1)
            A_bilat[:, v, u] = weight.squeeze(-1)
            
        A_total = A_fixed.unsqueeze(0) + A_bilat
        out = torch.bmm(A_total, x_perm)
        out = out.reshape(B, T, V, C).permute(0, 3, 1, 2)
        
        # Temporal smooth loss for edges: penalize high-frequency jitter in edge weights
        stacked_edges = torch.stack(edge_weights_list, dim=1) # (B, num_pairs, T)
        edge_temporal_diff = torch.mean(torch.abs(stacked_edges[:, :, 1:] - stacked_edges[:, :, :-1]))
        
        return F.silu(self.proj(out) + x, inplace=True), edge_temporal_diff

class EnhancedMS_TCN(nn.Module):
    """Multi-Scale Residual Dilated Temporal Convolution with SiLU activations."""
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes=[3, 5, 7], dilations=[1, 2], stride: int = 1, dropout: float = 0.25):
        super().__init__()
        num_branches = len(kernel_sizes) * len(dilations)
        branch_channels = max(out_channels // num_branches, 8)
        branches = []
        for k in kernel_sizes:
            for d in dilations:
                pad = (k - 1) * d // 2
                branches.append(nn.Sequential(
                    nn.Conv2d(in_channels, branch_channels, (k, 1), stride=(stride, 1), padding=(pad, 0), dilation=(d, 1), bias=False),
                    nn.BatchNorm2d(branch_channels),
                    nn.SiLU(inplace=True),
                    nn.Dropout(dropout)
                ))
        self.branches = nn.ModuleList(branches)
        total_dim = branch_channels * len(branches)
        self.proj = nn.Sequential(
            nn.Conv2d(total_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        if stride != 1 or in_channels != out_channels:
            self.res = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.res = nn.Identity()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]
        cat = torch.cat(feats, dim=1)
        return F.silu(self.proj(cat) + self.res(x), inplace=True)

class RC_BiGCN_v2(nn.Module):
    """
    Optimized RC-BiGCN Architecture:
    - Multi-Stream Invariant Features: Coords (2) + Velocity (2) + Acceleration (2) + Bone Vectors (2) + Reliability (1) = 9 channels.
    - Enhanced Bilateral Conv with differential & interaction terms + temporal regularizer.
    - Multi-Scale Residual TCN with SiLU non-linearities.
    """
    def __init__(self, in_channels: int = 9, num_classes: int = 10, base_channels: int = 36, dropout: float = 0.3):
        super().__init__()
        self.input_embed = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True)
        )
        self.st1_spatial = EnhancedBilateralConv(base_channels)
        self.st1_temporal = EnhancedMS_TCN(base_channels, base_channels, stride=1, dropout=dropout)
        
        self.st2_spatial = EnhancedBilateralConv(base_channels)
        self.st2_temporal = EnhancedMS_TCN(base_channels, base_channels * 2, stride=2, dropout=dropout)
        
        self.st3_spatial = EnhancedBilateralConv(base_channels * 2)
        self.st3_temporal = EnhancedMS_TCN(base_channels * 2, base_channels * 2, stride=2, dropout=dropout)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_channels * 2, base_channels),
            nn.LayerNorm(base_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(base_channels, num_classes)
        )
        
    def forward(self, x: torch.Tensor, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Multi-Stream Invariant Feature Computation
        coords = x[:, :, :, :2]
        root = coords[:, :, 8:9, :]
        neck = coords[:, :, 1:2, :]
        torso = torch.norm(neck - root, dim=-1, keepdim=True).clamp_min(1e-4)
        coords_norm = (coords - root) / torso
        
        # Velocity & Acceleration
        vel = torch.zeros_like(coords_norm)
        vel[:, 1:] = coords_norm[:, 1:] - coords_norm[:, :-1]
        acc = torch.zeros_like(vel)
        acc[:, 1:] = vel[:, 1:] - vel[:, :-1]
        
        # Bone Vectors
        bones = torch.zeros_like(coords_norm)
        for p, c in BODY25_BONES:
            bones[:, :, c] = coords_norm[:, :, c] - coords_norm[:, :, p]
            
        # Concatenate: (B, T, V, 9)
        inp = torch.cat([coords_norm, vel, acc, bones, q.unsqueeze(-1)], dim=-1)
        inp = inp.permute(0, 3, 1, 2).contiguous() # (B, 9, T, V)
        
        h = self.input_embed(inp)
        
        edge_loss_total = 0.0
        h, e1 = self.st1_spatial(h, q)
        h = self.st1_temporal(h)
        edge_loss_total += e1
        
        h, e2 = self.st2_spatial(h, q)
        h = self.st2_temporal(h)
        edge_loss_total += e2
        
        h, e3 = self.st3_spatial(h, q)
        h = self.st3_temporal(h)
        edge_loss_total += e3
        
        out = h.mean(dim=[-1, -2])
        logits = self.classifier(out)
        return logits, edge_loss_total

def apply_training_augmentation(X_train: torch.Tensor) -> torch.Tensor:
    """Safe skeleton augmentations applied strictly to training batches: scale variation and subtle coordinate jitter."""
    B, T, V, C = X_train.shape
    # Scale variation: [0.95, 1.05]
    scales = torch.empty(B, 1, 1, 1, device=X_train.device).uniform_(0.95, 1.05)
    X_aug = X_train.clone()
    X_aug[:, :, :, :2] = X_aug[:, :, :, :2] * scales
    # Subtle joint coordinate noise
    noise = torch.randn_like(X_aug[:, :, :, :2]) * 0.005
    X_aug[:, :, :, :2] = X_aug[:, :, :, :2] + noise
    return X_aug

def evaluate_optimized_pipeline(seed: int = 42):
    data = np.load(ROOT / "data/processed/mmasd_openpose/mmasd_openpose_64frames.npz")
    X, Q, y, groups = data['X'], data['Q'], data['y'], data['groups']
    
    print(f"\n=======================================================")
    print(f"Executing End-to-End RC-BiGCN v2 Optimisation (Seed {seed})")
    print(f"=======================================================")
    
    fold_accs, fold_f1s, fold_eces = [], [], []
    oof_preds = np.zeros_like(y)
    oof_probs = np.zeros((len(y), len(CLASSES)), dtype=np.float32)
    val_probs = np.zeros((len(y), len(CLASSES)), dtype=np.float32) # for leakage-free meta-stacking
    fold_records = []
    
    t0 = time.time()
    for fold_idx, (tr, te) in enumerate(grouped_folds(groups), 1):
        torch.manual_seed(seed + fold_idx)
        rng = np.random.default_rng(seed + fold_idx)
        
        ug = np.unique(groups[tr])
        rng.shuffle(ug)
        n_val = max(1, round(0.18 * len(ug)))
        val_groups = ug[:n_val]
        va_idx = tr[np.isin(groups[tr], val_groups)]
        fit_idx = tr[~np.isin(groups[tr], val_groups)]
        
        # Effective Number class weighting (beta = 0.99)
        counts = np.bincount(y[fit_idx], minlength=len(CLASSES)).astype(np.float32)
        beta = 0.99
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-4)
        weights = weights / np.sum(weights) * len(CLASSES)
        weight_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
        
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X[fit_idx], dtype=torch.float32),
                          torch.tensor(Q[fit_idx], dtype=torch.float32),
                          torch.tensor(y[fit_idx], dtype=torch.long)),
            batch_size=32, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X[va_idx], dtype=torch.float32),
                          torch.tensor(Q[va_idx], dtype=torch.float32),
                          torch.tensor(y[va_idx], dtype=torch.long)),
            batch_size=32, shuffle=False
        )
        test_loader = DataLoader(
            TensorDataset(torch.tensor(X[te], dtype=torch.float32),
                          torch.tensor(Q[te], dtype=torch.float32),
                          torch.tensor(y[te], dtype=torch.long)),
            batch_size=32, shuffle=False
        )
        
        model = RC_BiGCN_v2(in_channels=9, num_classes=10, base_channels=36, dropout=0.3).to(DEVICE)
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=65, eta_min=1e-5)
        
        best_val_f1 = -1.0
        best_state = None
        wait = 0
        
        for epoch in range(1, 66):
            model.train()
            for bx, bq, by in train_loader:
                bx, bq, by = bx.to(DEVICE), bq.to(DEVICE), by.to(DEVICE)
                bx = apply_training_augmentation(bx)
                optimizer.zero_grad(set_to_none=True)
                logits, edge_loss = model(bx, bq)
                # Label smoothing loss (0.05) + Edge Temporal Regularizer
                ce_loss = F.cross_entropy(logits, by, weight=weight_tensor, label_smoothing=0.05)
                loss = ce_loss + 0.05 * edge_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()
            
            # Validation pass
            model.eval()
            v_preds, v_targets, v_probs = [], [], []
            with torch.no_grad():
                for bx, bq, by in val_loader:
                    bx, bq = bx.to(DEVICE), bq.to(DEVICE)
                    logits, _ = model(bx, bq)
                    probs = F.softmax(logits, dim=-1).cpu().numpy()
                    v_probs.append(probs)
                    v_preds.extend(np.argmax(probs, axis=-1))
                    v_targets.extend(by.numpy())
            v_f1 = float(f1_score(v_targets, v_preds, average='macro', zero_division=0))
            if v_f1 > best_val_f1 + 1e-4:
                best_val_f1 = v_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_v_probs = np.concatenate(v_probs, axis=0)
                wait = 0
            else:
                wait += 1
                if wait >= 15:
                    break
                    
        # Test fold evaluation
        model.load_state_dict(best_state)
        model.eval()
        t_preds, t_targets, t_probs = [], [], []
        with torch.no_grad():
            for bx, bq, by in test_loader:
                bx, bq = bx.to(DEVICE), bq.to(DEVICE)
                logits, _ = model(bx, bq)
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                t_probs.append(probs)
                t_preds.extend(np.argmax(probs, axis=-1))
                t_targets.extend(by.numpy())
                
        t_probs = np.concatenate(t_probs, axis=0)
        t_preds = np.array(t_preds)
        t_targets = np.array(t_targets)
        
        fold_acc = float(accuracy_score(t_targets, t_preds))
        fold_f1 = float(f1_score(t_targets, t_preds, average='macro', zero_division=0))
        fold_ece = compute_ece(t_probs, t_targets)
        
        fold_accs.append(fold_acc)
        fold_f1s.append(fold_f1)
        fold_eces.append(fold_ece)
        oof_preds[te] = t_preds
        oof_probs[te] = t_probs
        val_probs[va_idx] = best_v_probs
        
        fold_records.append({
            "fold": fold_idx,
            "val_f1": best_val_f1,
            "accuracy": fold_acc,
            "macro_f1": fold_f1,
            "ece": fold_ece
        })
        print(f"  Fold {fold_idx}: Val F1 = {best_val_f1:.4f} | Test Acc = {fold_acc*100:.2f}%, Test Macro-F1 = {fold_f1:.4f}, ECE = {fold_ece:.4f}")
        
    total_time = time.time() - t0
    pooled_acc = float(accuracy_score(y, oof_preds))
    pooled_macro_f1 = float(f1_score(y, oof_preds, average='macro', zero_division=0))
    pooled_weighted_f1 = float(f1_score(y, oof_preds, average='weighted', zero_division=0))
    pooled_ece = compute_ece(oof_probs, y)
    
    per_class = {}
    f1s = f1_score(y, oof_preds, average=None, zero_division=0)
    for c_i, c_name in enumerate(CLASSES):
        per_class[c_name] = float(f1s[c_i])
        
    summary = {
        "model": "RC-BiGCN_v2",
        "seed": seed,
        "parameters": param_count,
        "runtime_seconds": total_time,
        "mean_fold_acc": float(np.mean(fold_accs)),
        "std_fold_acc": float(np.std(fold_accs)),
        "mean_fold_macro_f1": float(np.mean(fold_f1s)),
        "std_fold_macro_f1": float(np.std(fold_f1s)),
        "mean_fold_ece": float(np.mean(fold_eces)),
        "pooled_oof_acc": pooled_acc,
        "pooled_oof_macro_f1": pooled_macro_f1,
        "pooled_oof_weighted_f1": pooled_weighted_f1,
        "pooled_oof_ece": pooled_ece,
        "fold_results": fold_records,
        "per_class_f1": per_class
    }
    
    np.savez_compressed(
        ROOT / "final_optimization_experiments" / f"rc_bigcn_v2_seed{seed}_oof.npz",
        y_true=y, y_pred=oof_preds, probs=oof_probs, val_probs=val_probs
    )
    with open(ROOT / "final_optimization_experiments" / f"rc_bigcn_v2_seed{seed}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\nCompleted RC-BiGCN v2 (Seed {seed}):")
    print(f"Mean Fold Accuracy: {np.mean(fold_accs)*100:.2f}% ± {np.std(fold_accs)*100:.2f}%")
    print(f"Pooled OOF Accuracy: {pooled_acc*100:.2f}%")
    print(f"Pooled OOF Macro-F1: {pooled_macro_f1:.4f}")
    return summary

if __name__ == "__main__":
    evaluate_optimized_pipeline(seed=42)
