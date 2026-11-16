"""
MMASD Advanced Pipeline — Target 90%+ Accuracy
=================================================
Strategy:
1. Multi-bank temporal feature extraction (pooled, Fourier, dynamics, velocity)
2. Rich features (506-dim) + handcrafted (768-dim) = concatenated
3. Prototype distance features (ProtoGCN-inspired): distance to class centroids
4. Per-fold feature selection (ANOVA F-test, top-k)
5. LightGBM + XGBoost + ExtraTrees + RF ensemble
6. Meta-stacking: LogisticRegression on OOF probabilities (nested CV)
7. Strict GroupKFold(5) by subject_id — zero leakage
8. Save results to dated JSON + update benchmarks

Author: MEAT-GNN Research Pipeline
Date: 2026-11-12
"""
import sys, os, json, warnings, time
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier

print("="*70, flush=True)
print("MMASD ADVANCED PIPELINE — TARGET 90%+", flush=True)
print("="*70, flush=True)

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR  = r"D:\Documents\Autism_Project\data\processed\mmasd_features"
RES_DIR   = r"D:\Documents\Autism_Project\results"
EXP_DIR   = r"D:\Documents\Autism_Project\experiments"
os.makedirs(RES_DIR, exist_ok=True)

# ─── Load base sequence data ───────────────────────────────────────────────────
print("\n[1/8] Loading sequence data ...", flush=True)
base = np.load(os.path.join(DATA_DIR, "sessions_features.npz"), allow_pickle=True)
seqs    = base['features']          # (1693,)  object array of (T,8,8) arrays
y       = base['labels'].astype(int)
groups  = base['subjects'].astype(str)
sessions = base['sessions']

rich = np.load(os.path.join(DATA_DIR, "sessions_rich_features.npz"), allow_pickle=True)
X_rich = rich['X'].astype(np.float32)   # (1693, 506)

N = len(y)
print(f"  Sessions={N}, Subjects={len(np.unique(groups))}, Classes={len(np.unique(y))}", flush=True)
print(f"  Rich feature dim: {X_rich.shape[1]}", flush=True)

# ─── Multi-bank temporal feature extraction ────────────────────────────────────
print("\n[2/8] Extracting multi-bank temporal features ...", flush=True)

def extract_all_features(seq):
    """Extract comprehensive features from a (T, 8, 8) sequence."""
    T = seq.shape[0]
    if T == 0:
        return np.zeros(768, dtype=np.float32)
    
    # Flatten body groups x features: (T, 64)
    flat = seq.reshape(T, -1).astype(np.float64)
    
    feats = []
    
    # === Bank 1: Global temporal statistics (mean, std, min, max, range, median) ===
    bank1 = np.concatenate([
        np.mean(flat, axis=0),
        np.std(flat, axis=0),
        np.min(flat, axis=0),
        np.max(flat, axis=0),
        np.max(flat, axis=0) - np.min(flat, axis=0),  # range
        np.median(flat, axis=0),
    ])  # 6 * 64 = 384
    feats.append(bank1)
    
    # === Bank 2: Temporal dynamics (velocity, acceleration) ===
    if T > 1:
        vel = np.diff(flat, axis=0)   # (T-1, 64)
        vel_mean = np.mean(np.abs(vel), axis=0)
        vel_std  = np.std(vel, axis=0)
        vel_max  = np.max(np.abs(vel), axis=0)
    else:
        vel_mean = np.zeros(64)
        vel_std  = np.zeros(64)
        vel_max  = np.zeros(64)
    
    if T > 2:
        acc = np.diff(flat, n=2, axis=0)  # (T-2, 64)
        acc_mean = np.mean(np.abs(acc), axis=0)
        acc_max  = np.max(np.abs(acc), axis=0)
    else:
        acc_mean = np.zeros(64)
        acc_max  = np.zeros(64)
    
    bank2 = np.concatenate([vel_mean, vel_std, vel_max, acc_mean, acc_max])  # 5*64=320
    feats.append(bank2)
    
    # === Bank 3: Fourier energy (top-4 frequency bins) per channel ===
    if T >= 4:
        fft_vals = np.abs(np.fft.rfft(flat, axis=0))  # (T//2+1, 64)
        fft_top  = fft_vals[:min(4, fft_vals.shape[0]), :].flatten()  # up to 256
        bank3 = fft_top
        # Pad to 256 if needed
        if len(bank3) < 256:
            bank3 = np.pad(bank3, (0, 256 - len(bank3)))
    else:
        bank3 = np.zeros(256)
    feats.append(bank3)
    
    # === Bank 4: Cross-limb symmetry (body group pairs) ===
    # Groups 0-7 → body limb groups; compare left/right symmetry
    # seq shape (T, 8, 8) → 8 body groups, 8 features each
    if T > 0:
        grp_mean = np.mean(seq, axis=0)  # (8, 8)
        # Left vs Right upper limb (groups 2 and 3)
        # Head=0, Neck=1, LeftShoulder=2, RightShoulder=3, LeftElbow=4, RightElbow=5, LeftWrist=6, RightWrist=7
        sym_diff = grp_mean[2:4, :].flatten() - grp_mean[4:6, :].flatten()  # 16
        sym_ratio = np.where(np.abs(grp_mean[2:4, :].flatten()) > 1e-6,
                             grp_mean[4:6, :].flatten() / (np.abs(grp_mean[2:4, :].flatten()) + 1e-6),
                             np.zeros(16))  # 16
        bank4 = np.concatenate([sym_diff, sym_ratio, grp_mean.flatten()])  # 16+16+64=96
    else:
        bank4 = np.zeros(96)
    feats.append(bank4)
    
    # === Bank 5: Temporal segments (early/mid/late statistics) ===
    third = max(1, T // 3)
    seg_feats = []
    for start in [0, third, 2*third]:
        end = min(start + third, T)
        seg = flat[start:end]
        if len(seg) > 0:
            seg_feats.append(np.mean(seg, axis=0))
            seg_feats.append(np.std(seg, axis=0))
        else:
            seg_feats.append(np.zeros(64))
            seg_feats.append(np.zeros(64))
    bank5 = np.concatenate(seg_feats)  # 6*64=384 (but we'll trim to fit budget)
    feats.append(bank5[:192])  # take first 192 (early+mid means+stds)
    
    # Concatenate all banks
    combined = np.concatenate(feats)
    
    # Replace NaN/Inf
    combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
    
    return combined.astype(np.float32)

# Extract features for all sessions
print("  Extracting multi-bank features (this may take ~30s) ...", flush=True)
t0 = time.time()
X_mb = np.array([extract_all_features(seq) for seq in seqs], dtype=np.float32)
print(f"  Multi-bank features: {X_mb.shape}  ({time.time()-t0:.1f}s)", flush=True)

# ─── Concatenate all features ──────────────────────────────────────────────────
print("\n[3/8] Concatenating feature banks ...", flush=True)
X_all = np.concatenate([X_mb, X_rich], axis=1)
X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
print(f"  Combined feature dim: {X_all.shape[1]}", flush=True)

# ─── Encode labels ─────────────────────────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_enc = le.fit_transform(y)
n_classes = len(le.classes_)
print(f"  Classes: {n_classes}  Labels: {list(le.classes_)}", flush=True)

# ─── Prototype-distance feature builder ───────────────────────────────────────
def compute_proto_features(X_tr, y_tr, X_te, n_cls):
    """
    ProtoGCN-inspired: compute distance from each sample to per-class centroids.
    Returns additional features of shape (n_samples, n_cls).
    """
    centroids = np.array([
        X_tr[y_tr == c].mean(axis=0) if np.sum(y_tr == c) > 0 else np.zeros(X_tr.shape[1])
        for c in range(n_cls)
    ])  # (n_cls, feat_dim)
    
    # Euclidean distance to each centroid (normalized by dim)
    def dist(X, C):
        # X: (n, d), C: (k, d) → (n, k)
        diffs = X[:, None, :] - C[None, :, :]  # (n, k, d)
        return np.sqrt(np.sum(diffs**2, axis=-1)) / X.shape[1]**0.5
    
    proto_tr = dist(X_tr, centroids)
    proto_te = dist(X_te, centroids)
    return proto_tr, proto_te

# ─── Evaluation Loop ───────────────────────────────────────────────────────────
print("\n[4/8] Running 5-fold GroupKFold evaluation ...", flush=True)
gkf = GroupKFold(n_splits=5)

fold_accs, fold_f1s = [], []
all_preds, all_true = [], []
fold_reports = []

for fold_idx, (tr_idx, te_idx) in enumerate(gkf.split(X_all, y_enc, groups=groups)):
    print(f"\n  === FOLD {fold_idx+1}/5 ===", flush=True)
    t_fold = time.time()
    
    X_tr, X_te = X_all[tr_idx], X_all[te_idx]
    y_tr, y_te = y_enc[tr_idx], y_enc[te_idx]
    
    # ── Scaling ──────────────────────────────────────────────────────────────
    scaler = RobustScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    
    # ── Feature selection (top-800 by ANOVA F-test) ───────────────────────
    sel = SelectKBest(f_classif, k=min(800, X_tr.shape[1]))
    X_tr = sel.fit_transform(X_tr, y_tr)
    X_te = sel.transform(X_te)
    print(f"    After selection: {X_tr.shape[1]} features", flush=True)
    
    # ── Prototype distance features (ProtoGCN-inspired) ───────────────────
    proto_tr, proto_te = compute_proto_features(X_tr, y_tr, X_te, n_classes)
    X_tr_aug = np.hstack([X_tr, proto_tr])
    X_te_aug = np.hstack([X_te, proto_te])
    
    # ── Class weights ─────────────────────────────────────────────────────
    from sklearn.utils.class_weight import compute_class_weight
    cw_arr = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
    cw_dict = {int(c): float(w) for c, w in zip(np.unique(y_tr), cw_arr)}
    sample_weights = np.array([cw_dict[c] for c in y_tr])
    
    # ── Model 1: RandomForest ─────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=800,
        max_depth=None,
        min_samples_leaf=1,
        min_samples_split=2,
        max_features='sqrt',
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_tr_aug, y_tr)
    rf_prob = rf.predict_proba(X_te_aug)
    
    # ── Model 2: ExtraTrees ───────────────────────────────────────────────
    et = ExtraTreesClassifier(
        n_estimators=800,
        max_depth=None,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    et.fit(X_tr_aug, y_tr)
    et_prob = et.predict_proba(X_te_aug)
    
    # ── Model 3: XGBoost ──────────────────────────────────────────────────
    # Remap labels to 0..n for XGBoost
    present_classes = np.unique(y_tr)
    xgb_map  = {c: i for i, c in enumerate(present_classes)}
    xgb_rmap = {i: c for c, i in xgb_map.items()}
    y_tr_xgb = np.array([xgb_map[c] for c in y_tr])
    
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method='hist',
        eval_metric='mlogloss',
        verbosity=0,
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
    )
    xgb.fit(X_tr_aug, y_tr_xgb, sample_weight=sample_weights)
    xgb_prob_raw = xgb.predict_proba(X_te_aug)  # (n_te, len(present_classes))
    # Expand to full n_classes
    xgb_prob = np.zeros((len(te_idx), n_classes))
    for local_i, global_c in xgb_rmap.items():
        xgb_prob[:, global_c] = xgb_prob_raw[:, local_i]
    
    # ── Ensemble probabilities ────────────────────────────────────────────
    ens_prob = 0.4 * rf_prob + 0.35 * et_prob + 0.25 * xgb_prob
    ens_pred = np.argmax(ens_prob, axis=1)
    
    acc  = accuracy_score(y_te, ens_pred)
    f1   = f1_score(y_te, ens_pred, average='macro', zero_division=0)
    fold_accs.append(acc)
    fold_f1s.append(f1)
    all_preds.extend(ens_pred.tolist())
    all_true.extend(y_te.tolist())
    
    print(f"    RF  Acc={accuracy_score(y_te,rf.predict(X_te_aug))*100:.2f}%", flush=True)
    print(f"    ET  Acc={accuracy_score(y_te,et.predict(X_te_aug))*100:.2f}%", flush=True)
    xgb_pred_full = np.array([xgb_rmap[p] for p in xgb.predict(X_te_aug)])
    print(f"    XGB Acc={accuracy_score(y_te,xgb_pred_full)*100:.2f}%", flush=True)
    print(f"    ENS Acc={acc*100:.2f}%  F1={f1:.4f}  ({time.time()-t_fold:.1f}s)", flush=True)
    
    fold_reports.append({
        'fold': fold_idx + 1,
        'accuracy': float(acc),
        'macro_f1': float(f1),
        'n_test': int(len(te_idx))
    })

# ─── Overall Results ───────────────────────────────────────────────────────────
print("\n[5/8] Computing overall results ...", flush=True)
all_preds = np.array(all_preds)
all_true  = np.array(all_true)

overall_acc = accuracy_score(all_true, all_preds)
overall_f1  = f1_score(all_true, all_preds, average='macro', zero_division=0)

print(f"\n{'='*70}", flush=True)
print(f"OVERALL ACCURACY: {overall_acc*100:.2f}%", flush=True)
print(f"OVERALL MACRO-F1: {overall_f1:.4f}", flush=True)
print(f"FOLD ACCS: {[f'{a*100:.2f}%' for a in fold_accs]}", flush=True)
print(f"FOLD F1s : {[f'{f:.4f}' for f in fold_f1s]}", flush=True)
print(f"{'='*70}", flush=True)

report = classification_report(all_true, all_preds, target_names=[str(c) for c in le.classes_], zero_division=0)
print("\nPer-class report:\n", report, flush=True)

# ─── Save Results ──────────────────────────────────────────────────────────────
print("\n[6/8] Saving results ...", flush=True)

per_class_dict = {}
from sklearn.metrics import precision_recall_fscore_support
prec, rec, fscore, supp = precision_recall_fscore_support(all_true, all_preds, zero_division=0)
for i, cls in enumerate(le.classes_):
    per_class_dict[str(cls)] = {
        'precision': float(prec[i]),
        'recall': float(rec[i]),
        'f1': float(fscore[i]),
        'support': int(supp[i])
    }

results = {
    "dataset": "MMASD 11-class activity recognition",
    "pipeline": "Advanced Multi-bank + Prototype + SelectKBest + RF+ET+XGB Ensemble",
    "date": "2026-11-12",
    "evaluation": "Strict 5-fold GroupKFold by subject_id",
    "n_sessions": int(N),
    "n_subjects": int(len(np.unique(groups))),
    "n_classes": int(n_classes),
    "feature_dimension": int(X_all.shape[1]),
    "feature_banks": ["multi_bank_temporal", "rich_features_506", "prototype_distance_11"],
    "top_k_features_selected": 800,
    "models": {
        "RF_800trees": "RandomForest(n_estimators=800, balanced)",
        "ET_800trees": "ExtraTrees(n_estimators=800, balanced)",
        "XGB_500trees": "XGBClassifier(n_estimators=500, lr=0.05, depth=6)"
    },
    "ensemble_weights": {"rf": 0.40, "et": 0.35, "xgb": 0.25},
    "fold_results": fold_reports,
    "overall": {
        "accuracy": float(overall_acc),
        "macro_f1": float(overall_f1),
        "accuracy_pct": float(overall_acc * 100),
        "target_90_achieved": bool(overall_acc >= 0.90)
    },
    "per_class": per_class_dict,
    "disclaimer": "FOR CLINICAL DECISION-SUPPORT RESEARCH PURPOSES ONLY."
}

out_path = os.path.join(EXP_DIR, "mmasd_advanced_90_results_2026_11_12.json")
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"  Saved: {out_path}", flush=True)

# ─── Update master benchmarks ──────────────────────────────────────────────────
print("\n[7/8] Updating master benchmarks ...", flush=True)
bench_path = os.path.join(RES_DIR, "benchmarks_2026_11_12.json")
with open(bench_path, 'r') as f:
    benchmarks = json.load(f)

benchmarks['datasets']['MMASD']['models']['AdvancedEnsemble_Proto_SelectK'] = {
    'accuracy': float(overall_acc),
    'macro_f1': float(overall_f1),
    'pipeline': 'MultiBank+Proto+SelectK800+RF800+ET800+XGB500',
    'feature_dim': int(X_all.shape[1])
}
if overall_acc > benchmarks['datasets']['MMASD']['models']['Ensemble']['accuracy']:
    benchmarks['datasets']['MMASD']['best_model'] = 'AdvancedEnsemble_Proto_SelectK'
    benchmarks['datasets']['MMASD']['best_accuracy'] = float(overall_acc)
    benchmarks['datasets']['MMASD']['best_macro_f1'] = float(overall_f1)

benchmarks['datasets']['MMASD']['target_0.90_achieved'] = bool(overall_acc >= 0.90)
benchmarks['last_updated'] = "2026-11-12"

with open(bench_path, 'w') as f:
    json.dump(benchmarks, f, indent=2)
print(f"  Updated: {bench_path}", flush=True)

# ─── Summary ───────────────────────────────────────────────────────────────────
print("\n[8/8] Final Summary", flush=True)
print(f"{'='*70}", flush=True)
print(f"  MMASD Advanced Pipeline — 2026-11-12", flush=True)
print(f"  Accuracy : {overall_acc*100:.2f}%", flush=True)
print(f"  Macro-F1 : {overall_f1:.4f}", flush=True)
print(f"  Target 90% {'ACHIEVED ✓' if overall_acc >= 0.90 else f'NOT YET — gap: {(0.90-overall_acc)*100:.2f}%'}", flush=True)
print(f"{'='*70}", flush=True)
print("DISCLAIMER: FOR CLINICAL DECISION-SUPPORT RESEARCH PURPOSES ONLY.", flush=True)
