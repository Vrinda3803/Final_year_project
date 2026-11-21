"""
Multi-seed RC-BiGCN v2 runner (seeds 43 and 44).
Saves results alongside existing seed-42 results.
"""
import sys, os, time, json, copy, random
import numpy as np

sys.path.insert(0, r'D:\Documents\Autism_Project\.codex-deps')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from scipy.special import softmax as scipy_softmax

OUT_DIR = r'D:\Documents\Autism_Project\final_optimization_experiments'
DATA_PATH = r'D:\Documents\Autism_Project\data\processed\mmasd_openpose\mmasd_openpose_64frames.npz'

# ── Skeleton graph ──────────────────────────────────────────────────────────
EDGES = [
    (0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),
    (1,8),(8,9),(9,10),(1,11),(11,12),(12,13),
    (0,14),(14,16),(0,15),(15,17),(2,16),(5,17),
    (8,11),(10,19),(10,20),(10,21),(13,22),(13,23),(13,24),
]
BILATERAL_PAIRS = [
    (2,5),(3,6),(4,7),(8,11),(9,12),(10,13),
    (14,15),(16,17),(19,22),(20,23),(21,24),
]
N_JOINTS = 25

def build_adjacency(edges, n):
    A = np.zeros((n, n), dtype=np.float32)
    for u, v in edges:
        if u < n and v < n:
            A[u, v] = 1; A[v, u] = 1
    D = A.sum(1)
    D = np.where(D > 0, 1.0 / np.sqrt(D), 0)
    return (A * D[:, None]) * D[None, :]

A_fixed = build_adjacency(EDGES, N_JOINTS)

# ── RC-BiGCN v2 architecture ─────────────────────────────────────────────
class ReliabilityConditionedBilateralConv(nn.Module):
    def __init__(self, in_ch, out_ch, n_pairs, hidden=32):
        super().__init__()
        self.n_pairs = n_pairs
        # Edge MLP: [hu, hv, qu, qv, |hu-hv|, hu*hv] → weight
        self.edge_mlp = nn.Sequential(
            nn.Linear(2*in_ch + 2 + 2*in_ch, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        self.conv = nn.Linear(in_ch, out_ch, bias=False)
        self.norm = nn.LayerNorm(out_ch)
        self.pairs = BILATERAL_PAIRS[:n_pairs]

    def forward(self, x, q):
        # x: (B, T, V, C)   q: (B, T, V)
        B, T, V, C = x.shape
        agg = torch.zeros_like(x)
        edge_weights = []
        for u, v in self.pairs:
            if u >= V or v >= V:
                continue
            hu = x[:, :, u, :]   # (B,T,C)
            hv = x[:, :, v, :]
            qu = q[:, :, u:u+1]  # (B,T,1)
            qv = q[:, :, v:v+1]
            feat = torch.cat([hu, hv, qu, qv, (hu-hv).abs(), hu*hv], dim=-1)
            w = self.edge_mlp(feat)  # (B,T,1)
            edge_weights.append(w)
            agg[:, :, u, :] += w * hv
            agg[:, :, v, :] += w * hu
        total_weight = torch.stack(edge_weights, dim=-1).sum(-1) if edge_weights else torch.ones(B, T, 1, device=x.device)
        out = self.conv(x + agg)
        return self.norm(out), total_weight

class GCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, A, n_pairs=len(BILATERAL_PAIRS), dropout=0.3):
        super().__init__()
        self.A = nn.Parameter(torch.from_numpy(A).float(), requires_grad=False)
        self.spatial_conv = nn.Linear(in_ch, out_ch, bias=False)
        self.bilateral = ReliabilityConditionedBilateralConv(out_ch, out_ch, n_pairs)
        self.norm = nn.LayerNorm(out_ch)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.residual = nn.Linear(in_ch, out_ch, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x, q):
        # x: (B,T,V,C)
        B, T, V, C = x.shape
        res = self.residual(x)
        # Spatial graph conv
        xr = x.reshape(B*T, V, C)
        h = torch.bmm(self.A.unsqueeze(0).expand(B*T, -1, -1), xr)  # (B*T,V,C)
        h = self.spatial_conv(h).reshape(B, T, V, -1)
        # Bilateral
        h, ew = self.bilateral(h, q)
        h = self.act(self.norm(h + res))
        return self.drop(h), ew

class MSTCNBlock(nn.Module):
    def __init__(self, ch, kernel_sizes=(3,5,7), dropout=0.3):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            d = max(1, k // 2)
            self.branches.append(nn.Sequential(
                nn.Conv1d(ch, ch//len(kernel_sizes), k, padding=d*(k//2), dilation=d),
                nn.BatchNorm1d(ch//len(kernel_sizes)),
                nn.SiLU(),
            ))
        self.proj = nn.Sequential(
            nn.Conv1d(ch, ch, 1),
            nn.BatchNorm1d(ch),
        )
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, T)
        parts = [b(x) for b in self.branches]
        h = torch.cat(parts, dim=1)
        # pad/trim to match T
        if h.shape[-1] != x.shape[-1]:
            h = h[..., :x.shape[-1]]
        return self.drop(self.act(h + self.proj(x)))

class RCBiGCNv2(nn.Module):
    def __init__(self, n_classes=10, base_ch=36, n_gcn=2, dropout=0.3):
        super().__init__()
        # 9-channel input: xy + vx + vy + ax + ay + bx + by + q
        in_ch = 9
        self.input_proj = nn.Sequential(
            nn.Linear(in_ch, base_ch),
            nn.LayerNorm(base_ch),
            nn.SiLU(),
        )
        gcn_ch = [base_ch] + [base_ch*(2**i) for i in range(n_gcn)]
        self.gcn_layers = nn.ModuleList()
        for i in range(n_gcn):
            self.gcn_layers.append(GCNBlock(gcn_ch[i], gcn_ch[i+1], A_fixed, dropout=dropout))

        final_ch = gcn_ch[-1]
        # Temporal: pool over joints → (B, final_ch, T) → MS-TCN
        self.temporal = nn.Sequential(
            MSTCNBlock(final_ch, dropout=dropout),
            MSTCNBlock(final_ch, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(final_ch, final_ch//2),
            nn.LayerNorm(final_ch//2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(final_ch//2, n_classes),
        )

    def forward(self, x, q):
        # x: (B, T, V, 7)   q: (B, T, V)
        B, T, V, _ = x.shape
        h = self.input_proj(x)   # (B,T,V,base_ch)
        edge_regs = []
        for gcn in self.gcn_layers:
            h, ew = gcn(h, q)
            edge_regs.append(ew)
        # Pool joints
        h = h.mean(2)   # (B,T,C)
        h = h.permute(0, 2, 1)  # (B,C,T)
        h = self.temporal(h)
        h = self.pool(h).squeeze(-1)  # (B,C)
        logits = self.classifier(h)
        edge_reg = torch.stack([e.mean() for e in edge_regs]).mean() if edge_regs else torch.tensor(0.)
        return logits, edge_reg

# ── Feature extraction ─────────────────────────────────────────────────────
def extract_features(X_raw, Q_raw):
    """
    X_raw: (N, T, V, 2) coords (x,y)
    Q_raw: (N, T, V)    confidence
    Returns x_feat: (N,T,V,7)  q_feat: (N,T,V)
    """
    N, T, V, _ = X_raw.shape
    xy = X_raw[:, :, :, :2]  # (N,T,V,2)
    # velocity
    vel = np.zeros_like(xy)
    vel[:, 1:] = xy[:, 1:] - xy[:, :-1]
    # acceleration
    acc = np.zeros_like(xy)
    acc[:, 1:] = vel[:, 1:] - vel[:, :-1]
    # bone vectors (joint→parent, simplified: joint - joint 0)
    bone = xy - xy[:, :, 0:1, :]
    feat = np.concatenate([xy, vel, acc, bone], axis=-1)  # (N,T,V,8)
    q_feat = Q_raw.astype(np.float32)
    # Append q as 9th channel
    feat_q = np.concatenate([feat, q_feat[:, :, :, None]], axis=-1)  # (N,T,V,9)
    return feat_q.astype(np.float32), q_feat

def normalize_skeleton(X_feat, fit_idx=None):
    """Root-centric normalization fit on training indices."""
    # X_feat: (N,T,V,9) — first 2 channels are x,y
    if fit_idx is not None:
        # mid-hip center: joint 8 (R-hip) and 11 (L-hip)
        ref = X_feat[fit_idx, :, 8:9, :2]  # (n_train, T, 1, 2)
        torso = np.linalg.norm(
            X_feat[fit_idx, :, 1, :2] - X_feat[fit_idx, :, 0, :2], axis=-1
        ).mean()
        torso = max(torso, 1e-6)
    # apply
    out = X_feat.copy()
    mid_hip = (X_feat[:, :, 8:9, :2] + X_feat[:, :, 11:12, :2]) / 2
    out[:, :, :, :2] -= mid_hip
    out[:, :, :, [2,3]] -= (mid_hip[:, 1:] - mid_hip[:, :-1]) if False else 0  # vel already relative
    if fit_idx is not None:
        out[:, :, :, :2] /= torso
        # Also normalize velocity/accel/bone channels (indices 2-7)
        out[:, :, :, 2:8] /= torso
        return out, torso
    return out

# ── Training ───────────────────────────────────────────────────────────────
def effective_number_weights(y, n_classes, beta=0.99):
    counts = np.bincount(y, minlength=n_classes).astype(float)
    en = (1 - beta**counts) / (1 - beta)
    en = np.where(en > 0, en, 1.0)
    w = 1.0 / en
    w = w / w.sum() * n_classes
    return torch.tensor(w, dtype=torch.float32)

def augment(x_batch, q_batch, scale_jitter=0.05, coord_noise=0.005):
    if not torch.is_tensor(x_batch):
        return x_batch, q_batch
    s = 1.0 + (torch.rand(x_batch.shape[0], 1, 1, 1, device=x_batch.device) * 2 - 1) * scale_jitter
    x_batch = x_batch.clone()
    x_batch[:, :, :, :2] *= s
    x_batch[:, :, :, :2] += torch.randn_like(x_batch[:, :, :, :2]) * coord_noise
    return x_batch, q_batch

def run_seed(seed, n_folds=5):
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    data = np.load(DATA_PATH)
    X_raw = data['X'].astype(np.float32)   # (1374,64,25,4)
    Q_raw = data['Q'].astype(np.float32)   # (1374,64,25)
    y_raw = data['y'].astype(int)
    groups = data['groups'].astype(int)

    # Keep only x,y coords from X
    X_xy = X_raw[:, :, :, :2]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)

    gkf = GroupKFold(n_splits=n_folds)
    splits = list(gkf.split(X_xy, y, groups))

    all_probs = np.zeros((len(y), n_classes), dtype=np.float32)
    all_val_probs = np.zeros((len(y), n_classes), dtype=np.float32)
    fold_results = []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        print(f"\n[Seed {seed}] Fold {fold_i+1}/{n_folds}  train={len(train_idx)} test={len(test_idx)}")
        t0 = time.time()

        # Build features
        X_feat, Q_feat = extract_features(X_xy, Q_raw)

        # Normalize (fit on train)
        X_norm, torso = normalize_skeleton(X_feat, fit_idx=train_idx)
        X_norm_full = normalize_skeleton(X_feat)
        # Use the fitted torso on all
        X_norm_full[:, :, :, :8] /= torso
        # Actually recompute properly:
        mid_hip = (X_feat[:, :, 8:9, :2] + X_feat[:, :, 11:12, :2]) / 2
        X_norm2 = X_feat.copy()
        X_norm2[:, :, :, :2] -= mid_hip
        X_norm2[:, :, :, 2:8] = X_feat[:, :, :, 2:8]  # vel/accel/bone already relative
        X_norm2[:, :, :, :8] /= torso

        X_tr = torch.tensor(X_norm2[train_idx])
        Q_tr = torch.tensor(Q_feat[train_idx])
        y_tr = torch.tensor(y[train_idx], dtype=torch.long)

        X_te = torch.tensor(X_norm2[test_idx])
        Q_te = torch.tensor(Q_feat[test_idx])
        y_te = y[test_idx]

        # Split train into train/val (last 15%)
        n_tr = len(train_idx)
        val_size = max(1, int(n_tr * 0.15))
        perm = rng.permutation(n_tr)
        val_perm = perm[:val_size]
        tr_perm = perm[val_size:]

        X_val = X_tr[val_perm]; Q_val = Q_tr[val_perm]; y_val = y_tr[val_perm]
        X_tr2 = X_tr[tr_perm]; Q_tr2 = Q_tr[tr_perm]; y_tr2 = y_tr[tr_perm]

        weights = effective_number_weights(y_tr2.numpy(), n_classes)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
        val_crit = nn.CrossEntropyLoss()

        model = RCBiGCNv2(n_classes=n_classes, base_ch=36, n_gcn=2, dropout=0.3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-5)

        ds_tr = TensorDataset(X_tr2, Q_tr2, y_tr2)
        dl_tr = DataLoader(ds_tr, batch_size=32, shuffle=True, drop_last=False)

        best_val_loss = float('inf')
        best_state = None
        patience = 20
        patience_counter = 0

        for epoch in range(120):
            model.train()
            for xb, qb, yb in dl_tr:
                xb, qb = augment(xb, qb)
                optimizer.zero_grad()
                logits, edge_reg = model(xb, qb)
                loss = criterion(logits, yb) + 0.05 * edge_reg
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            # Validate
            model.eval()
            with torch.no_grad():
                vl, _ = model(X_val, Q_val)
                vloss = val_crit(vl, y_val).item()
            if vloss < best_val_loss:
                best_val_loss = vloss
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stop at epoch {epoch+1}")
                break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            logits_te, _ = model(X_te, Q_te)
            probs_te = F.softmax(logits_te, dim=-1).numpy()
            # Also get val probs
            logits_val_full, _ = model(X_tr[val_perm], Q_tr[val_perm])
            probs_val = F.softmax(logits_val_full, dim=-1).numpy()

        all_probs[test_idx] = probs_te
        all_val_probs[train_idx[val_perm]] = probs_val

        y_pred = probs_te.argmax(1)
        acc = accuracy_score(y_te, y_pred)
        mf1 = f1_score(y_te, y_pred, average='macro', zero_division=0)
        print(f"  Fold {fold_i+1}: acc={acc:.4f} macro_f1={mf1:.4f}  time={time.time()-t0:.0f}s")
        fold_results.append({'fold': fold_i+1, 'accuracy': acc, 'macro_f1': mf1})

    y_pred_all = all_probs.argmax(1)
    pooled_acc = accuracy_score(y, y_pred_all)
    pooled_mf1 = f1_score(y, y_pred_all, average='macro', zero_division=0)
    pooled_wf1 = f1_score(y, y_pred_all, average='weighted', zero_division=0)

    fold_accs = [fr['accuracy'] for fr in fold_results]
    fold_mf1s = [fr['macro_f1'] for fr in fold_results]

    per_class = f1_score(y, y_pred_all, average=None, zero_division=0)
    per_class_dict = {cls: float(per_class[i]) for i, cls in enumerate(le.classes_)}

    summary = {
        'model': 'RC-BiGCN v2',
        'seed': seed,
        'parameters': sum(p.numel() for p in RCBiGCNv2().parameters()),
        'mean_fold_acc': float(np.mean(fold_accs)),
        'std_fold_acc': float(np.std(fold_accs)),
        'mean_fold_macro_f1': float(np.mean(fold_mf1s)),
        'std_fold_macro_f1': float(np.std(fold_mf1s)),
        'pooled_oof_acc': float(pooled_acc),
        'pooled_oof_macro_f1': float(pooled_mf1),
        'pooled_oof_weighted_f1': float(pooled_wf1),
        'fold_results': fold_results,
        'per_class_f1': per_class_dict,
    }

    out_json = os.path.join(OUT_DIR, f'rc_bigcn_v2_seed{seed}_summary.json')
    out_npz = os.path.join(OUT_DIR, f'rc_bigcn_v2_seed{seed}_oof.npz')
    json.dump(summary, open(out_json, 'w'), indent=2)
    np.savez(out_npz, y_true=y, y_pred=y_pred_all, probs=all_probs, val_probs=all_val_probs)
    print(f"\n[Seed {seed}] DONE  pooled_acc={pooled_acc:.4f}  macro_f1={pooled_mf1:.4f}")
    return summary

if __name__ == '__main__':
    for seed in [43, 44]:
        s = run_seed(seed)
        print(f"\nSeed {seed}: pooled_acc={s['pooled_oof_acc']:.4f}  mean_fold={s['mean_fold_acc']:.4f}±{s['std_fold_acc']:.4f}")
    print("\nAll done.")
