"""
Controlled Ablation Experiment Suite for RC-BiGCN:
Isolating:
C0: Fixed Graph + MS-TCN (No bilateral edges, fixed anatomical adjacency only)
C1: Dynamic Bilateral Graph + MS-TCN (Bilateral MLP uses [hu, hv] only, no reliability conditioning)
C2: Dynamic Bilateral Graph + Reliability Input (Reliability q in input channels, bilateral MLP uses [hu, hv] only)
C3: Reliability-Conditioned Dynamic Bilateral Graph + MS-TCN (Full model, bilateral MLP uses [hu, hv, qu, qv])

Maintains strict 5-fold grouped cross-validation, identical splits, identical training schedules and seeds.
"""

from __future__ import annotations
import math, sys, os, time, json
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
from research_improvements.rc_bigcn import build_adjacency_matrix, BILATERAL_PAIRS, MultiScaleTemporalConv

DEVICE = torch.device("cpu")
BATCH_SIZE = 32
EPOCHS = 60
PATIENCE = 15
LR = 1e-3
WEIGHT_DECAY = 1e-4

def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == targets)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            avg_conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(acc_in_bin - avg_conf_in_bin) * prop_in_bin
    return float(ece)

class AblationBilateralConv(nn.Module):
    """
    Parametrized Graph Convolutional block supporting:
    - mode 'fixed': Fixed anatomical adjacency + learnable global matrix only (no bilateral edges).
    - mode 'dynamic_no_rel': Dynamic bilateral edges conditioned on [hu, hv] only.
    - mode 'dynamic_rel_cond': Dynamic bilateral edges conditioned on [hu, hv, qu, qv].
    """
    def __init__(self, channels: int, num_joints: int = 25, mode: str = 'dynamic_rel_cond'):
        super().__init__()
        self.num_joints = num_joints
        self.mode = mode
        self.register_buffer('A_base', build_adjacency_matrix(num_joints))
        self.A_learnable = nn.Parameter(torch.zeros(num_joints, num_joints))
        
        if mode == 'dynamic_rel_cond':
            mlp_in = channels * 2 + 2
        elif mode == 'dynamic_no_rel':
            mlp_in = channels * 2
        else:
            mlp_in = 0
            
        if mlp_in > 0:
            self.bilat_mlp = nn.Sequential(
                nn.Linear(mlp_in, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        else:
            self.bilat_mlp = None
            
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )
        
    def forward(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        B, C, T, V = x.shape
        A_fixed = self.A_base + self.A_learnable
        x_perm = x.permute(0, 2, 3, 1).reshape(B * T, V, C)
        
        if self.bilat_mlp is not None:
            if q.shape[1] != T:
                q_resampled = F.adaptive_avg_pool2d(q.unsqueeze(1), (T, V)).squeeze(1)
            else:
                q_resampled = q
            q_perm = q_resampled.reshape(B * T, V, 1)
            
            A_bilat = torch.zeros(B * T, V, V, device=x.device, dtype=x.dtype)
            for u, v in BILATERAL_PAIRS:
                hu, hv = x_perm[:, u], x_perm[:, v]
                if self.mode == 'dynamic_rel_cond':
                    qu, qv = q_perm[:, u], q_perm[:, v]
                    feat = torch.cat([hu, hv, qu, qv], dim=-1)
                else: # 'dynamic_no_rel'
                    feat = torch.cat([hu, hv], dim=-1)
                    
                weight = self.bilat_mlp(feat)
                A_bilat[:, u, v] = weight.squeeze(-1)
                A_bilat[:, v, u] = weight.squeeze(-1)
            A_total = A_fixed.unsqueeze(0) + A_bilat
        else:
            A_total = A_fixed.unsqueeze(0).expand(B * T, -1, -1)
            
        out = torch.bmm(A_total, x_perm)
        out = out.reshape(B, T, V, C).permute(0, 3, 1, 2)
        return F.relu(self.proj(out) + x, inplace=True)


class RC_BiGCN_AblationModel(nn.Module):
    """Configurable ablation model representing C0, C1, C2, or C3."""
    def __init__(self, in_channels: int, mode: str, base_channels: int = 32, num_joints: int = 25, dropout: float = 0.3):
        super().__init__()
        self.mode = mode
        self.input_embed = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.st1_spatial = AblationBilateralConv(base_channels, num_joints, mode=mode)
        self.st1_temporal = MultiScaleTemporalConv(base_channels, base_channels, stride=1, dropout=dropout)
        self.st2_spatial = AblationBilateralConv(base_channels, num_joints, mode=mode)
        self.st2_temporal = MultiScaleTemporalConv(base_channels, base_channels * 2, stride=2, dropout=dropout)
        self.st3_spatial = AblationBilateralConv(base_channels * 2, num_joints, mode=mode)
        self.st3_temporal = MultiScaleTemporalConv(base_channels * 2, base_channels * 2, stride=2, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_channels * 2, base_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(base_channels, 10)
        )
        
    def forward(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        # Preprocessing: root-centric + torso-scale normalized
        coords = x[:, :, :, :2]
        root = coords[:, :, 8:9, :]
        neck = coords[:, :, 1:2, :]
        torso = torch.norm(neck - root, dim=-1, keepdim=True).clamp_min(1e-4)
        coords_norm = (coords - root) / torso
        vel_norm = torch.zeros_like(coords_norm)
        vel_norm[:, 1:] = coords_norm[:, 1:] - coords_norm[:, :-1]
        
        # Build input
        if self.input_embed[0].in_channels == 5:
            inp = torch.cat([coords_norm, vel_norm, q.unsqueeze(-1)], dim=-1)
        else: # 4 channels (x, y, vx, vy)
            inp = torch.cat([coords_norm, vel_norm], dim=-1)
        inp = inp.permute(0, 3, 1, 2).contiguous()
        
        h = self.input_embed(inp)
        h = self.st1_spatial(h, q)
        h = self.st1_temporal(h)
        h = self.st2_spatial(h, q)
        h = self.st2_temporal(h)
        h = self.st3_spatial(h, q)
        h = self.st3_temporal(h)
        out = h.mean(dim=[-1, -2])
        return self.classifier(out)

def build_experiment_model(exp_id: str) -> RC_BiGCN_AblationModel:
    if exp_id == "C0":
        # Fixed Graph + MS-TCN (No bilateral edges, q in input)
        return RC_BiGCN_AblationModel(in_channels=5, mode='fixed', base_channels=32)
    elif exp_id == "C1":
        # Dynamic Bilateral Graph + MS-TCN (No reliability anywhere, in_channels=4, mode='dynamic_no_rel')
        return RC_BiGCN_AblationModel(in_channels=4, mode='dynamic_no_rel', base_channels=32)
    elif exp_id == "C2":
        # Dynamic Bilateral Graph + Reliability Input (q in input, but bilateral MLP does not use q)
        return RC_BiGCN_AblationModel(in_channels=5, mode='dynamic_no_rel', base_channels=32)
    elif exp_id == "C3":
        # Reliability-Conditioned Dynamic Bilateral Graph + MS-TCN (q in input, bilateral MLP uses q)
        return RC_BiGCN_AblationModel(in_channels=5, mode='dynamic_rel_cond', base_channels=32)
    else:
        raise ValueError(f"Unknown experiment ID: {exp_id}")

def run_experiment(exp_id: str, exp_desc: str, X, Q, y, groups, seed: int = 42) -> dict:
    print(f"\n{'='*75}\nExecuting Ablation Experiment {exp_id}: {exp_desc}\n{'='*75}")
    fold_accs, fold_f1s, fold_eces = [], [], []
    oof_preds = np.zeros_like(y)
    oof_probs = np.zeros((len(y), len(CLASSES)), dtype=np.float32)
    fold_results = []
    
    t0 = time.time()
    for fold_idx, (tr, te) in enumerate(grouped_folds(groups), 1):
        torch.manual_seed(seed + fold_idx)
        rng = np.random.default_rng(seed + fold_idx)
        
        # Inner validation split (disjoint recording groups)
        ug = np.unique(groups[tr])
        rng.shuffle(ug)
        n_val = max(1, round(0.18 * len(ug)))
        val_groups = ug[:n_val]
        va_idx = tr[np.isin(groups[tr], val_groups)]
        fit_idx = tr[~np.isin(groups[tr], val_groups)]
        
        counts = np.bincount(y[fit_idx], minlength=len(CLASSES)).astype(np.float32)
        weights = np.zeros_like(counts)
        valid_classes = counts > 0
        weights[valid_classes] = counts[valid_classes].sum() / (valid_classes.sum() * counts[valid_classes])
        weight_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
        
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
        
        model = build_experiment_model(exp_id).to(DEVICE)
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
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
            
            # Validation
            model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for bx, bq, by in val_loader:
                    bx, bq = bx.to(DEVICE), bq.to(DEVICE)
                    preds = model(bx, bq).argmax(dim=-1).cpu().numpy()
                    val_preds.extend(preds)
                    val_targets.extend(by.numpy())
            val_f1 = float(f1_score(val_targets, val_preds, average='macro', zero_division=0))
            if val_f1 > best_val_f1 + 1e-4:
                best_val_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= PATIENCE:
                    break
                    
        # Test fold evaluation
        model.load_state_dict(best_state)
        model.eval()
        t_preds, t_targets, t_probs = [], [], []
        with torch.no_grad():
            for bx, bq, by in test_loader:
                bx, bq = bx.to(DEVICE), bq.to(DEVICE)
                logits = model(bx, bq)
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
        
        fold_results.append({
            "fold": fold_idx,
            "accuracy": fold_acc,
            "macro_f1": fold_f1,
            "ece": fold_ece
        })
        print(f"  Fold {fold_idx}: Acc = {fold_acc*100:.2f}%, Macro-F1 = {fold_f1:.4f}, ECE = {fold_ece:.4f}")
        
    total_time = time.time() - t0
    pooled_acc = float(accuracy_score(y, oof_preds))
    pooled_macro_f1 = float(f1_score(y, oof_preds, average='macro', zero_division=0))
    pooled_weighted_f1 = float(f1_score(y, oof_preds, average='weighted', zero_division=0))
    pooled_ece = compute_ece(oof_probs, y)
    
    per_class_f1 = {}
    f1_all = f1_score(y, oof_preds, average=None, zero_division=0)
    for c_idx, c_name in enumerate(CLASSES):
        per_class_f1[c_name] = float(f1_all[c_idx])
        
    summary = {
        "exp_id": exp_id,
        "description": exp_desc,
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
        "fold_results": fold_results,
        "per_class_f1": per_class_f1
    }
    
    # Save OOF predictions
    np.savez_compressed(
        ROOT / "research_improvements" / f"ablation_{exp_id.lower()}_oof.npz",
        y_true=y, y_pred=oof_preds, probs=oof_probs
    )
    return summary

def main():
    data = np.load(ROOT / "data/processed/mmasd_openpose/mmasd_openpose_64frames.npz")
    X, Q, y, groups = data['X'], data['Q'], data['y'], data['groups']
    
    experiments = [
        ("C0", "Fixed Graph + MS-TCN (No Bilateral Edges)"),
        ("C1", "Dynamic Bilateral Graph + MS-TCN (No Reliability Anywhere)"),
        ("C2", "Dynamic Bilateral Graph + Reliability Input (q in input only)"),
        ("C3", "Reliability-Conditioned Dynamic Bilateral Graph + MS-TCN (Full Model)")
    ]
    
    master_summaries = {}
    for exp_id, exp_desc in experiments:
        summary = run_experiment(exp_id, exp_desc, X, Q, y, groups, seed=42)
        master_summaries[exp_id] = summary
        
    out_json = ROOT / "research_improvements" / "rc_bigcn_controlled_ablations_summary.json"
    with open(out_json, "w") as f:
        json.dump(master_summaries, f, indent=2)
    print(f"\nAll ablation experiments completed. Saved to {out_json}")

if __name__ == "__main__":
    main()
