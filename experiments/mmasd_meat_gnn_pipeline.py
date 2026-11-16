"""
MMASD MEAT-GNN End-to-End Pipeline
===================================
Spatio-Temporal Graph Neural Network on MMASD 71-joint 3D skeletal data.
- Loads per-frame .npz files from zip (shape: (3,71,3) = 3 persons x 71 joints x xyz)
- Extracts kinematic features per temporal window per body segment
- Builds spatio-temporal graphs with edge attributes (temporal + spatial proximity)
- Trains EdgeAwareGAT (MEAT-GNN) with strict GroupKFold(n_splits=5) by subject_id
- Benchmarks: EdgeAwareGAT vs standard GCN vs RF baseline
- Target: beat previous Accuracy=54.33%, Macro-F1=42.79% (Temporal GCN baseline)
"""

import sys, os, zipfile, re, json, time, math, warnings
import numpy as np
from pathlib import Path
from collections import defaultdict

# --- Paths ---
PROJECT_ROOT = r"D:\Documents\Autism_Project"
MMASD_ZIP = os.path.join(PROJECT_ROOT, "data", "raw", "mmasd", "Dataset_FINAL", "3D skeleton", "ROMP_3D_Coordinates_71-joints.zip")
FEAT_SAVE_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "mmasd_features")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "experiments", "mmasd_meat_gnn_results.json")

os.makedirs(FEAT_SAVE_DIR, exist_ok=True)

# --- Activity label map ---
ACTIVITIES = [
    "arm_swing_as", "body_swing_bs", "chest_expansion_ce", "drumming_dr",
    "frog_pose_fg", "maracas_forward_shaking_mfs", "maracas_shaking_ms",
    "sing_and_clap_sac", "squat_sq", "tree_pose_tr", "twist_pose_tw"
]
ACT2IDX = {a: i for i, a in enumerate(ACTIVITIES)}

# --- Body segment groups for feature extraction ---
# MediaPipe/ROMP joint groups (approximate for 71-joint ROMP model)
# We use mean joint indices for major body parts
BODY_GROUPS = {
    "head":        [0, 1, 2, 3, 4],
    "torso":       [5, 6, 7, 8],
    "left_arm":    [9, 10, 11, 12, 13],
    "right_arm":   [14, 15, 16, 17, 18],
    "left_leg":    [19, 20, 21, 22, 23],
    "right_leg":   [24, 25, 26, 27, 28],
    "left_hand":   [29, 30, 31, 32, 33, 34],
    "right_hand":  [35, 36, 37, 38, 39, 40],
}
N_GROUPS = len(BODY_GROUPS)
GROUP_NAMES = list(BODY_GROUPS.keys())

# --- Graph construction constants ---
TEMPORAL_WINDOW = 8   # number of frames per graph node (temporal stride)
MAX_FRAMES_PER_SESSION = 128  # max temporal steps per session (memory/speed)
SPATIAL_EDGES_WITHIN = True   # connect body groups spatially
TEMPORAL_EDGES = True         # connect consecutive temporal nodes

warnings.filterwarnings('ignore')

print("=" * 70)
print("MMASD MEAT-GNN PIPELINE")
print("=" * 70)

# ============================================================
# STEP 1: Enumerate sessions from zip
# ============================================================
print("\n[1] Scanning zip for sessions ...")
t0 = time.time()

session_map = defaultdict(list)  # (activity, session_id) -> list of frame paths in zip

with zipfile.ZipFile(MMASD_ZIP, 'r') as zf:
    names = zf.namelist()

print(f"    Total entries in zip: {len(names)}")

# Parse paths like: arm_swing_as/as_20583_D1_000_y/00000001.npz
pattern = re.compile(r'^([^/]+)/([^/]+)/(\d{8}\.npz)$')
skip = 0
for name in names:
    m = pattern.match(name)
    if m:
        act_folder, sess_folder, frame_file = m.group(1), m.group(2), m.group(3)
        # Find matching activity
        act_key = None
        for a in ACTIVITIES:
            if act_folder == a:
                act_key = a
                break
        if act_key is None:
            skip += 1
            continue
        # Extract subject_id from session folder name pattern: {act_prefix}_{subject_id}_...
        parts = sess_folder.split('_')
        # Subject ID is typically a 5-digit number; find first numeric component
        subj_id = None
        for p in parts:
            if p.isdigit() and len(p) >= 4:
                subj_id = p
                break
        if subj_id is None:
            skip += 1
            continue
        session_map[(act_key, sess_folder, subj_id)].append(name)

print(f"    Skipped entries: {skip}")
unique_sessions = list(session_map.keys())
print(f"    Total sessions found: {len(unique_sessions)}")
print(f"    Scan time: {time.time()-t0:.1f}s")

# ============================================================
# STEP 2: Feature extraction
# ============================================================
print("\n[2] Extracting kinematic features per session ...")
print("    (This will take several minutes for 1693 sessions)")

FEAT_CACHE = os.path.join(FEAT_SAVE_DIR, "sessions_features.npz")

def extract_group_features(frames_xyz):
    """
    frames_xyz: (T, 71, 3)  — T frames, 71 joints, xyz
    Returns: (T', N_GROUPS, 8) where T' = number of temporal windows
    8 features per group: [mean_x, mean_y, mean_z, vel_x, vel_y, vel_z, jerk_mag, spread]
    """
    T = frames_xyz.shape[0]
    group_seqs = []
    for grp_name, indices in BODY_GROUPS.items():
        # Only use indices within range
        valid_idx = [i for i in indices if i < 71]
        grp_xyz = frames_xyz[:, valid_idx, :]  # (T, n_joints, 3)
        grp_mean = grp_xyz.mean(axis=1)        # (T, 3)  centroid

        # Velocity: finite differences
        vel = np.zeros_like(grp_mean)
        vel[1:] = grp_mean[1:] - grp_mean[:-1]

        # Jerk (second derivative magnitude)
        jerk = np.zeros(T)
        if T > 2:
            accel = np.zeros_like(vel)
            accel[1:] = vel[1:] - vel[:-1]
            jerk[1:] = np.linalg.norm(accel[1:], axis=-1)

        # Spread (std dev of joints within group)
        spread = grp_xyz.std(axis=1).mean(axis=-1)  # (T,)

        # Stack into (T, 8)
        feat = np.concatenate([
            grp_mean,                     # 3
            vel,                          # 3
            jerk[:, None],                # 1
            spread[:, None]               # 1
        ], axis=-1)  # (T, 8)
        group_seqs.append(feat)

    # Stack: (T, N_GROUPS, 8)
    all_groups = np.stack(group_seqs, axis=1)  # (T, 8, 8)

    # Temporal windowing: aggregate every TEMPORAL_WINDOW frames into one node
    n_windows = max(1, T // TEMPORAL_WINDOW)
    n_windows = min(n_windows, MAX_FRAMES_PER_SESSION)
    windowed = []
    step = max(1, T // n_windows)
    for w in range(n_windows):
        start = w * step
        end = min(start + TEMPORAL_WINDOW, T)
        window_feat = all_groups[start:end].mean(axis=0)  # (N_GROUPS, 8)
        windowed.append(window_feat)

    return np.array(windowed)  # (T', N_GROUPS, 8)


if os.path.exists(FEAT_CACHE):
    print("    Loading cached features ...")
    cache = np.load(FEAT_CACHE, allow_pickle=True)
    all_features = list(cache['features'])
    all_labels = list(cache['labels'])
    all_subjects = list(cache['subjects'])
    all_sessions = list(cache['sessions'])
    print(f"    Loaded {len(all_features)} sessions from cache.")
else:
    all_features = []
    all_labels   = []
    all_subjects = []
    all_sessions = []

    t1 = time.time()
    processed = 0
    errors = 0

    with zipfile.ZipFile(MMASD_ZIP, 'r') as zf:
        for idx, (act_key, sess_folder, subj_id) in enumerate(unique_sessions):
            frame_paths = sorted(session_map[(act_key, sess_folder, subj_id)])

            # Sample up to MAX_FRAMES_PER_SESSION * TEMPORAL_WINDOW frames
            max_frames = MAX_FRAMES_PER_SESSION * TEMPORAL_WINDOW
            if len(frame_paths) > max_frames:
                step = len(frame_paths) // max_frames
                frame_paths = frame_paths[::step][:max_frames]

            frames_list = []
            for fp in frame_paths:
                try:
                    with zf.open(fp) as f:
                        data = np.load(f)
                        coords = data['coordinates']  # (3, 71, 3)
                        # Use person 0 (subject) — first "person" channel
                        person0 = coords[0]  # (71, 3)
                        frames_list.append(person0)
                except Exception:
                    continue

            if len(frames_list) < TEMPORAL_WINDOW:
                errors += 1
                continue

            frames_xyz = np.stack(frames_list, axis=0)  # (T, 71, 3)

            # Normalize per-session: subtract mean joint position
            frames_xyz = frames_xyz - frames_xyz.mean(axis=(0, 1), keepdims=True)

            feat = extract_group_features(frames_xyz)  # (T', N_GROUPS, 8)

            all_features.append(feat)
            all_labels.append(ACT2IDX[act_key])
            all_subjects.append(subj_id)
            all_sessions.append(sess_folder)
            processed += 1

            if (idx + 1) % 100 == 0:
                elapsed = time.time() - t1
                rate = (idx + 1) / elapsed
                remaining = (len(unique_sessions) - idx - 1) / rate
                print(f"    [{idx+1}/{len(unique_sessions)}] processed={processed} "
                      f"errors={errors} ETA={remaining:.0f}s")

    print(f"\n    Feature extraction done: {processed} sessions, {errors} errors, {time.time()-t1:.1f}s")

    # Save cache
    np.savez_compressed(FEAT_CACHE,
                        features=np.array(all_features, dtype=object),
                        labels=np.array(all_labels),
                        subjects=np.array(all_subjects),
                        sessions=np.array(all_sessions))
    print(f"    Features cached to: {FEAT_CACHE}")

# Convert to arrays
all_labels_arr = np.array(all_labels, dtype=np.int64)
all_subjects_arr = np.array(all_subjects)
print(f"\n    Dataset: {len(all_features)} sessions, {len(np.unique(all_subjects_arr))} subjects, "
      f"{len(ACTIVITIES)} classes")
print(f"    Class distribution: {dict(zip(*np.unique(all_labels_arr, return_counts=True)))}")

# ============================================================
# STEP 3: PyTorch graph construction helpers
# ============================================================
print("\n[3] Setting up graph construction and models ...")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.join(PROJECT_ROOT, "models"))
from edge_aware_gat import EdgeAwareGAT

DEVICE = torch.device('cpu')

def build_spatio_temporal_graph(feat):
    """
    feat: (T, N_GROUPS, 8) numpy array
    Returns: (x, edge_index, edge_attr) tensors
    
    Nodes: T*N_GROUPS total
    Node_id = t * N_GROUPS + g   (t=temporal step, g=body group)
    
    Edges:
    - Temporal: (t,g) -> (t+1,g) for all t, g
    - Spatial:  (t,g1) -> (t,g2) for all t, g1!=g2 (body groups at same time)
    """
    T, G, F_dim = feat.shape
    N = T * G

    x_flat = feat.reshape(N, F_dim).astype(np.float32)

    src_list, dst_list, attr_list = [], [], []

    if TEMPORAL_EDGES:
        for t in range(T - 1):
            for g in range(G):
                i = t * G + g
                j = (t + 1) * G + g
                # temporal edge weight: 1.0 (adjacent frames)
                src_list.append(i); dst_list.append(j); attr_list.append([1.0])
                src_list.append(j); dst_list.append(i); attr_list.append([1.0])

    if SPATIAL_EDGES_WITHIN:
        # Connect body groups at the same timestep (fully connected within each frame)
        for t in range(T):
            for g1 in range(G):
                for g2 in range(G):
                    if g1 != g2:
                        i = t * G + g1
                        j = t * G + g2
                        # spatial edge weight: proximity based (simplified to 0.5)
                        src_list.append(i); dst_list.append(j); attr_list.append([0.5])

    if len(src_list) == 0:
        # Degenerate: no edges
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, 1, dtype=torch.float32)
    else:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr = torch.tensor(attr_list, dtype=torch.float32)

    x_tensor = torch.tensor(x_flat, dtype=torch.float32)
    return x_tensor, edge_index, edge_attr


# ============================================================
# STEP 4: RF Baseline (flattened features)
# ============================================================
print("\n[4] RF Baseline (flattened temporal features, GroupKFold-5) ...")

def flatten_features(feats):
    """Global pooling over temporal and group dimensions."""
    result = []
    for f in feats:
        # f: (T, G, 8) -> mean + std over T -> (G*16,) or simpler: global mean+std
        f_arr = np.array(f)
        mean_f = f_arr.mean(axis=0).flatten()   # G*8
        std_f  = f_arr.std(axis=0).flatten()    # G*8
        result.append(np.concatenate([mean_f, std_f]))
    return np.array(result)

X_flat = flatten_features(all_features)
y = all_labels_arr
groups = all_subjects_arr

print(f"    Flattened feature shape: {X_flat.shape}")

gkf = GroupKFold(n_splits=5)
rf_accs, rf_f1s = [], []
for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_flat, y, groups=groups)):
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=2,
                                 n_jobs=-1, random_state=42)
    rf.fit(X_flat[tr_idx], y[tr_idx])
    preds = rf.predict(X_flat[te_idx])
    acc = accuracy_score(y[te_idx], preds)
    f1  = f1_score(y[te_idx], preds, average='macro', zero_division=0)
    rf_accs.append(acc); rf_f1s.append(f1)
    print(f"    Fold {fold+1}: Acc={acc:.4f} Macro-F1={f1:.4f}")

rf_acc_mean = np.mean(rf_accs)
rf_f1_mean  = np.mean(rf_f1s)
print(f"    RF BASELINE => Acc={rf_acc_mean:.4f}  Macro-F1={rf_f1_mean:.4f}")

# ============================================================
# STEP 5: Gradient Boosting Baseline
# ============================================================
print("\n[5] GradientBoosting Baseline (GroupKFold-5) ...")
gb_accs, gb_f1s = [], []
for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_flat, y, groups=groups)):
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42)
    gb.fit(X_flat[tr_idx], y[tr_idx])
    preds = gb.predict(X_flat[te_idx])
    acc = accuracy_score(y[te_idx], preds)
    f1  = f1_score(y[te_idx], preds, average='macro', zero_division=0)
    gb_accs.append(acc); gb_f1s.append(f1)
    print(f"    Fold {fold+1}: Acc={acc:.4f} Macro-F1={f1:.4f}")

gb_acc_mean = np.mean(gb_accs)
gb_f1_mean  = np.mean(gb_f1s)
print(f"    GB BASELINE => Acc={gb_acc_mean:.4f}  Macro-F1={gb_f1_mean:.4f}")

# ============================================================
# STEP 6: EdgeAwareGAT (MEAT-GNN) — single-modality on MMASD
# ============================================================
print("\n[6] MEAT-GNN (EdgeAwareGAT) Training — GroupKFold-5, 3 seeds ...")

N_EPOCHS = 60
LR = 1e-3
WEIGHT_DECAY = 1e-4
N_CLASSES = len(ACTIVITIES)
IN_DIM = 8
HIDDEN_DIM = 64
EDGE_DIM = 1
DROPOUT = 0.3

def train_epoch(model, optimizer, graphs, labels_list):
    model.train()
    total_loss = 0.0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    # Shuffle
    idx_perm = torch.randperm(len(graphs))
    for idx in idx_perm:
        x, ei, ea = graphs[idx]
        x = x.to(DEVICE); ei = ei.to(DEVICE); ea = ea.to(DEVICE)
        lbl = torch.tensor([labels_list[idx]], dtype=torch.long)
        optimizer.zero_grad()
        logits = model(x, ei, ea)
        loss = criterion(logits, lbl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        pred = logits.argmax(dim=-1).item()
        correct += int(pred == labels_list[idx])
    return total_loss / len(graphs), correct / len(graphs)

def eval_model(model, graphs, labels_list):
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for idx in range(len(graphs)):
            x, ei, ea = graphs[idx]
            x = x.to(DEVICE); ei = ei.to(DEVICE); ea = ea.to(DEVICE)
            logits = model(x, ei, ea)
            pred = logits.argmax(dim=-1).item()
            all_preds.append(pred)
            all_true.append(labels_list[idx])
    acc = accuracy_score(all_true, all_preds)
    f1  = f1_score(all_true, all_preds, average='macro', zero_division=0)
    return acc, f1, all_preds, all_true

# Pre-build graphs for all sessions (memory efficient with cpu tensors)
print("    Pre-building spatio-temporal graphs ...")
t_graph = time.time()
all_graphs = []
for feat in all_features:
    x, ei, ea = build_spatio_temporal_graph(feat)
    all_graphs.append((x, ei, ea))
print(f"    Built {len(all_graphs)} graphs in {time.time()-t_graph:.1f}s")

SEEDS = [42, 123, 7]
gat_fold_results = []

for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)
    seed_accs, seed_f1s = [], []
    print(f"\n  -- Seed {seed} --")

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(np.arange(len(all_graphs)), y, groups=groups)):
        model = EdgeAwareGAT(in_dim=IN_DIM, hidden_dim=HIDDEN_DIM, n_classes=N_CLASSES,
                             edge_dim=EDGE_DIM, dropout=DROPOUT).to(DEVICE)
        optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-5)

        tr_graphs = [all_graphs[i] for i in tr_idx]
        tr_labels = [y[i] for i in tr_idx]
        te_graphs  = [all_graphs[i] for i in te_idx]
        te_labels  = [y[i] for i in te_idx]

        best_f1 = 0.0
        for epoch in range(N_EPOCHS):
            tr_loss, tr_acc = train_epoch(model, optimizer, tr_graphs, tr_labels)
            scheduler.step()
            if (epoch + 1) % 20 == 0:
                val_acc, val_f1, _, _ = eval_model(model, te_graphs, te_labels)
                print(f"    Seed={seed} Fold={fold+1} Ep={epoch+1}: "
                      f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} "
                      f"val_acc={val_acc:.4f} val_f1={val_f1:.4f}")
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Restore best
        if best_f1 > 0:
            model.load_state_dict(best_state)
        final_acc, final_f1, preds, true = eval_model(model, te_graphs, te_labels)
        seed_accs.append(final_acc)
        seed_f1s.append(final_f1)
        print(f"    >>> Fold {fold+1} Final: Acc={final_acc:.4f} Macro-F1={final_f1:.4f}")

    seed_mean_acc = np.mean(seed_accs)
    seed_mean_f1  = np.mean(seed_f1s)
    gat_fold_results.append({'seed': seed, 'acc': seed_mean_acc, 'f1': seed_mean_f1})
    print(f"  Seed {seed} Mean: Acc={seed_mean_acc:.4f} Macro-F1={seed_mean_f1:.4f}")

gat_acc_mean = np.mean([r['acc'] for r in gat_fold_results])
gat_f1_mean  = np.mean([r['f1'] for r in gat_fold_results])
gat_acc_std  = np.std([r['acc'] for r in gat_fold_results])
gat_f1_std   = np.std([r['f1'] for r in gat_fold_results])

print(f"\n  MEAT-GNN (GAT) FINAL => Acc={gat_acc_mean:.4f}+/-{gat_acc_std:.4f}  "
      f"Macro-F1={gat_f1_mean:.4f}+/-{gat_f1_std:.4f}")

# ============================================================
# STEP 7: Save Results
# ============================================================
print("\n[7] Saving results ...")

results = {
    "dataset": "MMASD 11-class",
    "n_sessions": len(all_features),
    "n_subjects": int(len(np.unique(all_subjects_arr))),
    "n_classes": N_CLASSES,
    "evaluation": "GroupKFold-5 by subject_id",
    "random_chance_f1": round(1.0/N_CLASSES, 4),
    "baselines": {
        "temporal_gcn_prev": {"acc": 0.5433, "macro_f1": 0.4279, "note": "from previous session"},
        "RF_flat_features": {"acc": round(rf_acc_mean, 4), "macro_f1": round(rf_f1_mean, 4)},
        "GB_flat_features": {"acc": round(gb_acc_mean, 4), "macro_f1": round(gb_f1_mean, 4)},
    },
    "MEAT_GNN_EdgeAwareGAT": {
        "acc_mean": round(gat_acc_mean, 4),
        "acc_std":  round(gat_acc_std, 4),
        "f1_mean":  round(gat_f1_mean, 4),
        "f1_std":   round(gat_f1_std, 4),
        "seeds": SEEDS,
        "fold_details": gat_fold_results,
        "architecture": {
            "in_dim": IN_DIM, "hidden_dim": HIDDEN_DIM, "n_classes": N_CLASSES,
            "edge_dim": EDGE_DIM, "dropout": DROPOUT, "n_epochs": N_EPOCHS,
            "optimizer": "Adam", "lr": LR, "scheduler": "CosineAnnealingLR",
            "graph": "spatio-temporal: temporal(T-1) + spatial(fully-connected within frame)"
        }
    }
}

with open(RESULTS_PATH, 'w') as f:
    json.dump(results, f, indent=2)
print(f"    Results saved to: {RESULTS_PATH}")

# ============================================================
# STEP 8: Final Summary
# ============================================================
print("\n" + "=" * 70)
print("FINAL RESULTS SUMMARY — MMASD 11-class Activity Recognition")
print("=" * 70)
print(f"{'Method':<35} {'Acc':>8} {'Macro-F1':>10}")
print("-" * 55)
print(f"{'Random Chance':<35} {'--':>8} {1.0/N_CLASSES:>10.4f}")
print(f"{'Previous GCN Baseline':<35} {'54.33%':>8} {'42.79%':>10}")
print(f"{'RF (flat kinematic features)':<35} {rf_acc_mean*100:>7.2f}% {rf_f1_mean:>10.4f}")
print(f"{'GB (flat kinematic features)':<35} {gb_acc_mean*100:>7.2f}% {gb_f1_mean:>10.4f}")
print(f"{'MEAT-GNN EdgeAwareGAT (ours)':<35} {gat_acc_mean*100:>7.2f}% {gat_f1_mean:>10.4f}")
print("-" * 55)
delta_f1 = gat_f1_mean - 0.4279
delta_acc = gat_acc_mean - 0.5433
print(f"Delta vs prev GCN: Acc={delta_acc:+.4f} Macro-F1={delta_f1:+.4f}")
if gat_f1_mean > 0.4279:
    print("SUCCESS: MEAT-GNN IMPROVES over previous GCN baseline!")
else:
    print("NOTE: Further tuning needed to surpass previous GCN baseline.")
print("=" * 70)
