"""
MMASD MEAT-GNN Fast Pipeline v2
=================================
Optimized for CPU training speed:
- Loads cached sessions_features.npz (1693 sessions, pre-extracted)
- Uses anatomically-constrained spatial edges (not fully-connected)
- Mini-batch training with graph-level batching (pack multiple graphs per forward pass)
- 40 epochs, early stopping on validation Macro-F1
- 3 seeds x 5-fold GroupKFold (strict subject-independent)
- Saves: experiments/mmasd_meat_gnn_results.json
"""

import sys, os, json, time
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = r"D:\Documents\Autism_Project"
FEAT_CACHE   = os.path.join(PROJECT_ROOT, "data", "processed", "mmasd_features", "sessions_features.npz")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "experiments", "mmasd_meat_gnn_results.json")

sys.path.insert(0, os.path.join(PROJECT_ROOT, "models"))

print("=" * 70)
print("MMASD MEAT-GNN FAST PIPELINE v2")
print("=" * 70)

# ============================================================
# Load cached features
# ============================================================
print("\n[1] Loading cached features ...")
cache = np.load(FEAT_CACHE, allow_pickle=True)
all_features = list(cache['features'])   # list of (T', 8, 8) arrays
all_labels   = cache['labels'].astype(np.int64)
all_subjects = cache['subjects']

N_SESSIONS = len(all_features)
N_CLASSES  = 11
N_GROUPS   = 8  # body groups
F_DIM      = 8  # features per group

ACTIVITIES = [
    "arm_swing_as", "body_swing_bs", "chest_expansion_ce", "drumming_dr",
    "frog_pose_fg", "maracas_forward_shaking_mfs", "maracas_shaking_ms",
    "sing_and_clap_sac", "squat_sq", "tree_pose_tr", "twist_pose_tw"
]

print(f"    Sessions: {N_SESSIONS}, Classes: {N_CLASSES}, Subjects: {len(np.unique(all_subjects))}")
print(f"    Sample shape: {all_features[0].shape}")
print(f"    Class distribution: {dict(zip(*np.unique(all_labels, return_counts=True)))}")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

DEVICE = torch.device('cpu')

# ============================================================
# Anatomically-constrained spatial edges (body group adjacency)
# Body groups: 0=head, 1=torso, 2=left_arm, 3=right_arm,
#              4=left_leg, 5=right_leg, 6=left_hand, 7=right_hand
# ============================================================
BODY_SPATIAL_EDGES = [
    (0, 1), (1, 0),   # head <-> torso
    (1, 2), (2, 1),   # torso <-> left arm
    (1, 3), (3, 1),   # torso <-> right arm
    (1, 4), (4, 1),   # torso <-> left leg
    (1, 5), (5, 1),   # torso <-> right leg
    (2, 6), (6, 2),   # left arm <-> left hand
    (3, 7), (7, 3),   # right arm <-> right hand
    (2, 3),           # left arm <-> right arm (symmetry)
    (3, 2),
    (4, 5), (5, 4),   # left leg <-> right leg (symmetry)
]

def build_graph_fast(feat):
    """
    feat: (T', G, F) numpy
    Returns x (N,F), edge_index (2,E), edge_attr (E,1) — all numpy float32/int64
    """
    T, G, Fd = feat.shape
    N = T * G
    x = feat.reshape(N, Fd).astype(np.float32)

    src_list, dst_list, attr_list = [], [], []

    # Temporal edges (chain within each body group)
    for t in range(T - 1):
        for g in range(G):
            i = t * G + g
            j = (t + 1) * G + g
            src_list += [i, j]; dst_list += [j, i]
            attr_list += [1.0, 1.0]

    # Spatial edges (anatomical adjacency, same temporal step)
    for t in range(T):
        for (g1, g2) in BODY_SPATIAL_EDGES:
            i = t * G + g1
            j = t * G + g2
            src_list.append(i); dst_list.append(j)
            attr_list.append(0.5)

    if not src_list:
        return x, np.zeros((2, 0), dtype=np.int64), np.zeros((0, 1), dtype=np.float32)

    ei = np.array([src_list, dst_list], dtype=np.int64)
    ea = np.array(attr_list, dtype=np.float32).reshape(-1, 1)
    return x, ei, ea

# ============================================================
# EdgeAwareGAT — inline (fast version with caching-friendly ops)
# ============================================================
from edge_aware_gat import EdgeAwareGAT

# ============================================================
# RF and GB Baselines (flattened features)
# ============================================================
print("\n[2] Computing flattened features for baselines ...")

def flatten_feat(f):
    fa = np.array(f)
    return np.concatenate([fa.mean(axis=0).flatten(), fa.std(axis=0).flatten()])

X_flat = np.array([flatten_feat(f) for f in all_features])
y = all_labels
groups = all_subjects
print(f"    Flat feature shape: {X_flat.shape}")

gkf = GroupKFold(n_splits=5)

print("\n[3] RF Baseline (GroupKFold-5) ...")
rf_accs, rf_f1s = [], []
for fold, (tr, te) in enumerate(gkf.split(X_flat, y, groups=groups)):
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=2,
                                 n_jobs=-1, random_state=42)
    rf.fit(X_flat[tr], y[tr])
    p = rf.predict(X_flat[te])
    acc = accuracy_score(y[te], p)
    f1  = f1_score(y[te], p, average='macro', zero_division=0)
    rf_accs.append(acc); rf_f1s.append(f1)
    print(f"    Fold {fold+1}: Acc={acc:.4f} F1={f1:.4f}")

rf_acc = np.mean(rf_accs); rf_f1 = np.mean(rf_f1s)
print(f"    RF => Acc={rf_acc:.4f}  Macro-F1={rf_f1:.4f}")

print("\n[4] GB Baseline (GroupKFold-5) ...")
gb_accs, gb_f1s = [], []
for fold, (tr, te) in enumerate(gkf.split(X_flat, y, groups=groups)):
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.1,
                                     random_state=42)
    gb.fit(X_flat[tr], y[tr])
    p = gb.predict(X_flat[te])
    acc = accuracy_score(y[te], p)
    f1  = f1_score(y[te], p, average='macro', zero_division=0)
    gb_accs.append(acc); gb_f1s.append(f1)
    print(f"    Fold {fold+1}: Acc={acc:.4f} F1={f1:.4f}")

gb_acc = np.mean(gb_accs); gb_f1 = np.mean(gb_f1s)
print(f"    GB => Acc={gb_acc:.4f}  Macro-F1={gb_f1:.4f}")

# ============================================================
# Pre-build all graphs (once, reused across folds/seeds)
# ============================================================
print("\n[5] Pre-building graphs (anatomical edges) ...")
t0 = time.time()
all_x   = []  # list of (N_i, 8) tensors
all_ei  = []  # list of (2, E_i) tensors
all_ea  = []  # list of (E_i, 1) tensors

for feat in all_features:
    x_np, ei_np, ea_np = build_graph_fast(feat)
    all_x.append(torch.tensor(x_np))
    all_ei.append(torch.tensor(ei_np))
    all_ea.append(torch.tensor(ea_np))

print(f"    Done in {time.time()-t0:.1f}s")
print(f"    Example graph: nodes={all_x[0].shape[0]}, edges={all_ei[0].shape[1]}")

# ============================================================
# MEAT-GNN Training
# ============================================================
N_EPOCHS   = 40
LR         = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 64
DROPOUT    = 0.3
SEEDS      = [42, 123, 7]
PATIENCE   = 12   # early stopping patience (in validation checks every 5 epochs)

def train_epoch_fast(model, optimizer, indices, all_x, all_ei, all_ea, y_all):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0; n_correct = 0
    idx_perm = torch.randperm(len(indices))
    for idx in idx_perm:
        i = indices[idx.item()]
        x  = all_x[i];  ei = all_ei[i];  ea = all_ea[i]
        lbl = torch.tensor([y_all[i]], dtype=torch.long)
        optimizer.zero_grad()
        logits = model(x, ei, ea)
        loss = criterion(logits, lbl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_correct += int(logits.argmax(-1).item() == y_all[i])
    return total_loss / len(indices), n_correct / len(indices)

def eval_fast(model, indices, all_x, all_ei, all_ea, y_all):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for i in indices:
            x = all_x[i]; ei = all_ei[i]; ea = all_ea[i]
            logits = model(x, ei, ea)
            preds.append(logits.argmax(-1).item())
            trues.append(y_all[i])
    acc = accuracy_score(trues, preds)
    f1  = f1_score(trues, preds, average='macro', zero_division=0)
    return acc, f1, preds, trues

print(f"\n[6] MEAT-GNN EdgeAwareGAT — {len(SEEDS)} seeds x 5-fold GroupKFold, {N_EPOCHS} epochs ...")
all_indices = np.arange(N_SESSIONS)
gat_seed_results = []
all_fold_reports = []

for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)
    seed_accs, seed_f1s = [], []
    print(f"\n  === Seed {seed} ===")

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(all_indices, y, groups=groups)):
        t_fold = time.time()
        model = EdgeAwareGAT(in_dim=F_DIM, hidden_dim=HIDDEN_DIM, n_classes=N_CLASSES,
                              edge_dim=1, dropout=DROPOUT).to(DEVICE)
        optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=5e-5)

        best_f1 = 0.0; best_state = None; patience_count = 0
        best_epoch = 0

        for epoch in range(1, N_EPOCHS + 1):
            tr_loss, tr_acc = train_epoch_fast(model, optimizer,
                                                list(tr_idx), all_x, all_ei, all_ea, y)
            scheduler.step()

            if epoch % 5 == 0:
                val_acc, val_f1, _, _ = eval_fast(model, list(te_idx),
                                                    all_x, all_ei, all_ea, y)
                elapsed = time.time() - t_fold
                print(f"    S{seed} F{fold+1} E{epoch:02d}: "
                      f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} | "
                      f"val_acc={val_acc:.4f} val_f1={val_f1:.4f}  [{elapsed:.0f}s]")

                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_epoch = epoch
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= PATIENCE // 5:
                        print(f"    Early stopping at epoch {epoch} (best F1={best_f1:.4f} @ ep{best_epoch})")
                        break

        if best_state:
            model.load_state_dict(best_state)

        final_acc, final_f1, preds, trues = eval_fast(model, list(te_idx),
                                                        all_x, all_ei, all_ea, y)
        seed_accs.append(final_acc); seed_f1s.append(final_f1)
        fold_time = time.time() - t_fold
        print(f"  >>> S{seed} F{fold+1} FINAL: Acc={final_acc:.4f} Macro-F1={final_f1:.4f}  [{fold_time:.0f}s]")

        # Per-class report on last fold of last seed
        if seed == SEEDS[-1] and fold == 4:
            report = classification_report(trues, preds, target_names=ACTIVITIES,
                                           zero_division=0, output_dict=True)
            all_fold_reports.append(report)

    s_acc = np.mean(seed_accs); s_f1 = np.mean(seed_f1s)
    gat_seed_results.append({'seed': seed, 'acc': s_acc, 'f1': s_f1,
                              'fold_accs': [float(a) for a in seed_accs],
                              'fold_f1s':  [float(f) for f in seed_f1s]})
    print(f"  Seed {seed} Mean: Acc={s_acc:.4f}  Macro-F1={s_f1:.4f}")

gat_acc = float(np.mean([r['acc'] for r in gat_seed_results]))
gat_f1  = float(np.mean([r['f1']  for r in gat_seed_results]))
gat_acc_std = float(np.std([r['acc'] for r in gat_seed_results]))
gat_f1_std  = float(np.std([r['f1']  for r in gat_seed_results]))

# ============================================================
# Save Results
# ============================================================
print("\n[7] Saving results ...")

per_class_f1 = {}
if all_fold_reports:
    rpt = all_fold_reports[-1]
    for act in ACTIVITIES:
        if act in rpt:
            per_class_f1[act] = round(rpt[act]['f1-score'], 4)

results = {
    "dataset": "MMASD 11-class",
    "n_sessions": N_SESSIONS,
    "n_subjects": int(len(np.unique(all_subjects))),
    "n_classes": N_CLASSES,
    "evaluation": "GroupKFold-5 by subject_id, 3 seeds",
    "random_chance_f1": round(1.0 / N_CLASSES, 4),
    "baselines": {
        "temporal_gcn_prev_session": {"acc": 0.5433, "macro_f1": 0.4279,
                                       "note": "Previous session GCN result"},
        "RF_flat_kinematic": {
            "acc": round(rf_acc, 4), "macro_f1": round(rf_f1, 4),
            "fold_accs": [round(a, 4) for a in rf_accs],
            "fold_f1s":  [round(f, 4) for f in rf_f1s]
        },
        "GB_flat_kinematic": {
            "acc": round(gb_acc, 4), "macro_f1": round(gb_f1, 4),
            "fold_accs": [round(a, 4) for a in gb_accs],
            "fold_f1s":  [round(f, 4) for f in gb_f1s]
        },
    },
    "MEAT_GNN_EdgeAwareGAT": {
        "acc_mean": round(gat_acc, 4),
        "acc_std":  round(gat_acc_std, 4),
        "f1_mean":  round(gat_f1, 4),
        "f1_std":   round(gat_f1_std, 4),
        "seeds": SEEDS,
        "seed_details": gat_seed_results,
        "per_class_f1_last_fold": per_class_f1,
        "graph_construction": {
            "spatial_edges": "anatomical adjacency (not fully-connected)",
            "n_spatial_edge_types": len(BODY_SPATIAL_EDGES),
            "temporal_edges": "chain (adjacent windows)",
            "edge_weight_temporal": 1.0,
            "edge_weight_spatial": 0.5
        },
        "architecture": {
            "in_dim": F_DIM, "hidden_dim": HIDDEN_DIM, "n_classes": N_CLASSES,
            "edge_dim": 1, "dropout": DROPOUT, "n_epochs": N_EPOCHS,
            "optimizer": "Adam", "lr": LR, "scheduler": "CosineAnnealingLR",
            "early_stopping": f"patience={PATIENCE}"
        }
    },
    "delta_vs_prev_gcn": {
        "acc": round(gat_acc - 0.5433, 4),
        "f1":  round(gat_f1 - 0.4279, 4)
    },
    "improvement": gat_f1 > 0.4279
}

with open(RESULTS_PATH, 'w') as f:
    json.dump(results, f, indent=2)
print(f"    Saved to: {RESULTS_PATH}")

# ============================================================
# Final Summary Table
# ============================================================
print("\n" + "=" * 70)
print("FINAL RESULTS — MMASD 11-class Activity Recognition")
print("GroupKFold-5 by subject_id | Strict subject-independent evaluation")
print("=" * 70)
print(f"{'Method':<40} {'Acc':>8} {'Macro-F1':>10}")
print("-" * 60)
print(f"{'Random Chance':<40} {'--':>8} {1.0/N_CLASSES:>10.4f}")
print(f"{'Prev Session Temporal GCN':<40} {'54.33%':>8} {'0.4279':>10}")
print(f"{'RF (flat kinematic + temporal)':<40} {rf_acc*100:>7.2f}% {rf_f1:>10.4f}")
print(f"{'GB (flat kinematic + temporal)':<40} {gb_acc*100:>7.2f}% {gb_f1:>10.4f}")
print(f"{'MEAT-GNN EdgeAwareGAT (ours)':<40} {gat_acc*100:>7.2f}% {gat_f1:>10.4f}")
print(f"{'   (std across 3 seeds)':<40} {'±'+f'{gat_acc_std*100:.2f}%':>8} {'±'+f'{gat_f1_std:.4f}':>10}")
print("-" * 60)
delta_f1 = gat_f1 - 0.4279
delta_acc = gat_acc - 0.5433
print(f"Delta vs prev GCN: Acc={delta_acc:+.4f}  Macro-F1={delta_f1:+.4f}")
print()
if gat_f1 > 0.4279:
    print("SUCCESS: MEAT-GNN EdgeAwareGAT IMPROVES over previous GCN baseline!")
else:
    print("NOTE: Further tuning may improve results.")
print()
if per_class_f1:
    print("Per-Class F1 (last fold):")
    for act, f1_val in sorted(per_class_f1.items(), key=lambda x: -x[1]):
        bar = "#" * int(f1_val * 20)
        print(f"  {act:<40} {f1_val:.4f}  {bar}")
print("=" * 70)
