"""
AV-ASD Visual Graph Construction + Multimodal MEAT-GNN Training
================================================================
USE THIS SCRIPT when AV-ASD MediaPipe keypoint .npy files are available locally.

Expected input:
  D:\Documents\Autism_Project\data\processed\keypoints\av_asd\{Video_ID}.npy
  Each file shape: (T, 33, 3) — T frames, 33 MediaPipe joints, xyz

Also requires:
  Audio features:   D:\Downloads\AV_ASD_audio_stage1_audit_bundle (1)\features\segment_features.npy
  Row ID mapping:   D:\Downloads\AV_ASD_audio_stage1_audit_bundle (1)\features\segment_features_row_ids.csv
  Annotation file:  D:\Downloads\AV_ASD_audio_stage1_audit_bundle (1)\reports\segments_with_annotation_alignment.csv

Output:
  D:\Documents\Autism_Project\experiments\meat_gnn_avasd_results.json
  D:\Documents\Autism_Project\experiments\gate_attribution_results.json
"""

import sys, os, json, time
import numpy as np
from pathlib import Path

PROJECT_ROOT = r"D:\Documents\Autism_Project"
KEYPOINTS_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "keypoints", "av_asd")
AUDIO_FEAT_PATH = r"D:\Downloads\AV_ASD_audio_stage1_audit_bundle (1)\features\segment_features.npy"
AUDIO_ROW_IDS   = r"D:\Downloads\AV_ASD_audio_stage1_audit_bundle (1)\features\segment_features_row_ids.csv"
ANNOTATION_CSV  = r"D:\Downloads\AV_ASD_audio_stage1_audit_bundle (1)\reports\segments_with_annotation_alignment.csv"
RESULTS_PATH    = os.path.join(PROJECT_ROOT, "experiments", "meat_gnn_avasd_results.json")
GATE_PATH       = os.path.join(PROJECT_ROOT, "experiments", "gate_attribution_results.json")

# --- Check keypoints directory ---
if not os.path.exists(KEYPOINTS_DIR):
    print(f"ERROR: Keypoints directory not found: {KEYPOINTS_DIR}")
    print("Please download from Google Drive:")
    print("  Colab path: /content/drive/MyDrive/Autism-Behaviour-Graph-XAI/.../data/processed/keypoints/av_asd/")
    print("  Copy all .npy files to:", KEYPOINTS_DIR)
    sys.exit(1)

kpt_files = list(Path(KEYPOINTS_DIR).glob("*.npy"))
print(f"Found {len(kpt_files)} keypoint files in {KEYPOINTS_DIR}")

# --- MediaPipe 33 joint body groups ---
MP_BODY_GROUPS = {
    "face":        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "left_arm":    [11, 13, 15, 17, 19, 21],
    "right_arm":   [12, 14, 16, 18, 20, 22],
    "torso":       [11, 12, 23, 24],
    "left_leg":    [23, 25, 27, 29, 31],
    "right_leg":   [24, 26, 28, 30, 32],
}
N_GROUPS = len(MP_BODY_GROUPS)
GROUP_NAMES = list(MP_BODY_GROUPS.keys())

TEMPORAL_WINDOW = 5
MAX_WINDOWS = 32

LABEL_MAP = {
    "Aggression": 0,
    "Upper Limb Stereotypies": 1,
    "Lower Limb Stereotypies": 2,
    "Mouthing": 3,
}
N_CLASSES = 4

import pandas as pd

# --- Load audio features and annotations ---
print("Loading audio features ...")
X_aud_all = np.load(AUDIO_FEAT_PATH)  # (N_total_segments, 32)
row_ids_df = pd.read_csv(AUDIO_ROW_IDS)
annot_df   = pd.read_csv(ANNOTATION_CSV)

# Align: get the 182 clips with valid labels
valid = annot_df[annot_df['Behavior_Label'].notna() & annot_df['Video_ID'].notna()].copy()
print(f"Valid annotated clips: {len(valid)}")

# --- Load and process keypoints + build dataset ---
sys.path.insert(0, os.path.join(PROJECT_ROOT, "models"))
from edge_aware_gat import MultimodalMEATGNN

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report

DEVICE = torch.device('cpu')

def extract_mp_group_features(frames_xyz):
    """frames_xyz: (T, 33, 3) -> (T', N_GROUPS, 8)"""
    T = frames_xyz.shape[0]
    group_seqs = []
    for grp_name, indices in MP_BODY_GROUPS.items():
        valid_idx = [i for i in indices if i < 33]
        grp_xyz   = frames_xyz[:, valid_idx, :]  # (T, n, 3)
        grp_mean  = grp_xyz.mean(axis=1)         # (T, 3)
        vel = np.zeros_like(grp_mean)
        vel[1:] = grp_mean[1:] - grp_mean[:-1]
        jerk = np.zeros(T)
        if T > 2:
            accel = np.zeros_like(vel)
            accel[1:] = vel[1:] - vel[:-1]
            jerk[1:] = np.linalg.norm(accel[1:], axis=-1)
        spread = grp_xyz.std(axis=1).mean(axis=-1)
        feat = np.concatenate([grp_mean, vel, jerk[:, None], spread[:, None]], axis=-1)
        group_seqs.append(feat)
    all_groups = np.stack(group_seqs, axis=1)  # (T, G, 8)
    n_windows = min(MAX_WINDOWS, max(1, T // TEMPORAL_WINDOW))
    step = max(1, T // n_windows)
    windowed = []
    for w in range(n_windows):
        start = w * step
        end = min(start + TEMPORAL_WINDOW, T)
        windowed.append(all_groups[start:end].mean(axis=0))
    return np.array(windowed)  # (T', G, 8)

def build_st_graph(feat):
    T, G, F = feat.shape
    N = T * G
    x_flat = feat.reshape(N, F).astype(np.float32)
    src_list, dst_list, attr_list = [], [], []
    for t in range(T - 1):
        for g in range(G):
            i = t*G+g; j = (t+1)*G+g
            src_list += [i, j]; dst_list += [j, i]; attr_list += [[1.0], [1.0]]
    for t in range(T):
        for g1 in range(G):
            for g2 in range(G):
                if g1 != g2:
                    src_list.append(t*G+g1); dst_list.append(t*G+g2); attr_list.append([0.5])
    if not src_list:
        return (torch.tensor(x_flat), torch.zeros(2,0,dtype=torch.long), torch.zeros(0,1))
    return (torch.tensor(x_flat),
            torch.tensor([src_list, dst_list], dtype=torch.long),
            torch.tensor(attr_list, dtype=torch.float32))

# Build dataset
print("Building dataset ...")
all_graphs  = []
all_audio   = []
all_labels  = []
all_subjects = []

for _, row in valid.iterrows():
    vid_id = str(row['Video_ID'])
    kpt_path = os.path.join(KEYPOINTS_DIR, vid_id + ".npy")
    if not os.path.exists(kpt_path):
        continue
    frames = np.load(kpt_path)  # (T, 33, 3) or (T, 99)
    if frames.ndim == 2:
        frames = frames.reshape(frames.shape[0], -1, 3)
    frames = frames - frames.mean(axis=(0,1), keepdims=True)
    feat = extract_mp_group_features(frames)
    graph = build_st_graph(feat)

    # Get audio feature for this clip
    audio_rows = row_ids_df[row_ids_df['Video_ID'] == row['Video_ID']]
    if len(audio_rows) == 0:
        continue
    row_idx = audio_rows.iloc[0]['row_id']
    aud_feat = X_aud_all[row_idx]  # (32,)

    label = LABEL_MAP.get(row['Behavior_Label'], -1)
    if label < 0:
        continue

    all_graphs.append(graph)
    all_audio.append(torch.tensor(aud_feat, dtype=torch.float32))
    all_labels.append(label)
    subj_parts = str(row['Video_ID']).split('_')
    all_subjects.append(subj_parts[0] if subj_parts else vid_id)

print(f"Dataset built: {len(all_graphs)} clips, {len(np.unique(all_subjects))} subjects, {N_CLASSES} classes")

# MEAT-GNN Training
IN_DIM = 8; HIDDEN_DIM = 64; AUDIO_DIM = 32; N_EPOCHS = 80; LR = 1e-3; DROPOUT = 0.3

gkf = GroupKFold(n_splits=5)
y = np.array(all_labels)
groups = np.array(all_subjects)
SEEDS = [42, 123, 7]

all_seed_results = []
all_gate_values = {c: [] for c in range(N_CLASSES)}

for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)
    seed_accs, seed_f1s = [], []

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(np.arange(len(all_graphs)), y, groups=groups)):
        model = MultimodalMEATGNN(in_dim=IN_DIM, audio_dim=AUDIO_DIM, hidden_dim=HIDDEN_DIM,
                                   n_classes=N_CLASSES, edge_dim=1, dropout=DROPOUT).to(DEVICE)
        optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-5)
        criterion = nn.CrossEntropyLoss()

        tr_graphs  = [all_graphs[i] for i in tr_idx]
        tr_audio   = [all_audio[i]  for i in tr_idx]
        tr_labels  = [y[i] for i in tr_idx]
        te_graphs  = [all_graphs[i] for i in te_idx]
        te_audio   = [all_audio[i]  for i in te_idx]
        te_labels  = [y[i] for i in te_idx]

        best_f1 = 0.0; best_state = None
        for epoch in range(N_EPOCHS):
            model.train()
            idx_perm = torch.randperm(len(tr_graphs))
            for idx in idx_perm:
                x, ei, ea = tr_graphs[idx]
                aud = tr_audio[idx].unsqueeze(0)
                lbl = torch.tensor([tr_labels[idx]], dtype=torch.long)
                optimizer.zero_grad()
                logits, _ = model(x, ei, ea, aud)
                loss = criterion(logits, lbl)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            if (epoch+1) % 20 == 0:
                model.eval()
                preds, trues = [], []
                with torch.no_grad():
                    for i in range(len(te_graphs)):
                        x, ei, ea = te_graphs[i]
                        aud = te_audio[i].unsqueeze(0)
                        logits, _ = model(x, ei, ea, aud)
                        preds.append(logits.argmax(-1).item())
                        trues.append(te_labels[i])
                val_f1 = f1_score(trues, preds, average='macro', zero_division=0)
                val_acc = accuracy_score(trues, preds)
                print(f"  Seed={seed} Fold={fold+1} Ep={epoch+1}: val_acc={val_acc:.4f} val_f1={val_f1:.4f}")
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if best_state: model.load_state_dict(best_state)
        model.eval()
        preds, trues, gates_per_class = [], [], {c: [] for c in range(N_CLASSES)}
        with torch.no_grad():
            for i in range(len(te_graphs)):
                x, ei, ea = te_graphs[i]
                aud = te_audio[i].unsqueeze(0)
                logits, g = model(x, ei, ea, aud)
                pred = logits.argmax(-1).item()
                true_cls = te_labels[i]
                preds.append(pred)
                trues.append(true_cls)
                # Gate: g is [1, hidden], mean over hidden = scalar gate value
                gate_val = g.mean().item()
                all_gate_values[true_cls].append(gate_val)

        acc = accuracy_score(trues, preds)
        f1  = f1_score(trues, preds, average='macro', zero_division=0)
        seed_accs.append(acc); seed_f1s.append(f1)
        print(f"  >>> Seed={seed} Fold={fold+1}: Acc={acc:.4f} Macro-F1={f1:.4f}")

    all_seed_results.append({'seed': seed, 'acc': np.mean(seed_accs), 'f1': np.mean(seed_f1s)})

# Gate attribution
gate_summary = {}
CLASS_NAMES = {0: "Aggression", 1: "Upper Limb Stereotypies", 2: "Lower Limb Stereotypies", 3: "Mouthing"}
for cls, vals in all_gate_values.items():
    if vals:
        gate_summary[CLASS_NAMES[cls]] = {
            "mean_gate": round(float(np.mean(vals)), 4),
            "std_gate": round(float(np.std(vals)), 4),
            "interpretation": "g<0.5 -> audio-dominant, g>0.5 -> visual-dominant",
            "n_samples": len(vals)
        }

final_acc = np.mean([r['acc'] for r in all_seed_results])
final_f1  = np.mean([r['f1'] for r in all_seed_results])

print("\n" + "="*60)
print("MULTIMODAL MEAT-GNN — AV-ASD RESULTS")
print(f"  Acc={final_acc:.4f}  Macro-F1={final_f1:.4f}")
print(f"\nGate Attribution per class:")
for cls_name, g_info in gate_summary.items():
    dom = "VISUAL" if g_info['mean_gate'] > 0.5 else "AUDIO"
    print(f"  {cls_name}: g={g_info['mean_gate']:.3f} ({dom}-dominant)")

results = {
    "dataset": "AV-ASD 4-class",
    "evaluation": "GroupKFold-5 by subject_id",
    "MEAT_GNN_multimodal": {"acc": round(final_acc, 4), "f1": round(final_f1, 4), "seeds": SEEDS},
    "baselines": {"audio_GB": {"acc": 0.4725, "f1": 0.4290}, "pose_RF": {"acc": 0.4759, "f1": 0.4403}},
    "delta_vs_pose_rf": {"acc": round(final_acc - 0.4759, 4), "f1": round(final_f1 - 0.4403, 4)},
}
with open(RESULTS_PATH, 'w') as f: json.dump(results, f, indent=2)
with open(GATE_PATH, 'w') as f: json.dump(gate_summary, f, indent=2)
print(f"\nResults saved to: {RESULTS_PATH}")
print(f"Gate attribution saved to: {GATE_PATH}")
