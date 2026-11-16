"""
AV-ASD Hyper-Tuned 3-Class Pipeline
=====================================
Target: Push 74.61% → 85%+ on AV-ASD 3-class

Key insight from previous run:
- 3-class segment-level got 74.61% (huge improvement!)
- Hyper-reactivity class has only 55 samples → very low recall (11%)
- Upper Limb Stereotypies dominates (700 samples → 85% F1)
- Aggressive Behavior moderate (206 samples → 57% F1)

Strategy:
1. Undersample Upper Limb Stereotypies (cap at 300 per fold)
2. Use cost-sensitive learning more aggressively
3. Try different feature engineering (MFCC-based aggregation)
4. Per-class threshold calibration
5. Run only 3-class (cleaner problem)

Date: 2026-11-12
"""
import sys, os, csv, ast, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_recall_fscore_support
from xgboost import XGBClassifier

print("="*70, flush=True)
print("AV-ASD HYPER-TUNED 3-CLASS PIPELINE — TARGET 80%+", flush=True)
print("="*70, flush=True)

FEAT_DIR = r"D:\Downloads\AV_ASD_audio_stage1_audit_bundle (1)\features"
REP_DIR  = r"D:\Downloads\AV_ASD_audio_stage1_audit_bundle (1)\reports"
RES_DIR  = r"D:\Documents\Autism_Project\results"
EXP_DIR  = r"D:\Documents\Autism_Project\experiments"

# ─── Load data ─────────────────────────────────────────────────────────────────
print("\n[1] Loading data ...", flush=True)
X_audio_all = np.load(os.path.join(FEAT_DIR, "segment_features.npy"))
feat_ids = list(csv.DictReader(open(os.path.join(FEAT_DIR, "segment_features_row_ids.csv"))))
feat_id_to_idx = {r['segment_id']: i for i, r in enumerate(feat_ids)}
manifest = list(csv.DictReader(open(os.path.join(REP_DIR, "final_segment_manifest.csv"))))

CLS3 = ['Aggressive Behavior', 'Hyper- or Hyporeactivity to Sensory Input', 'Upper Limb Stereotypies']
PRIORITY = ['Aggressive Behavior', 'Self-Spinning or Spinning Objects', 'Upper Limb Stereotypies', 'Hyper- or Hyporeactivity to Sensory Input']

def get_label(row, cls_name):
    key = f'annotation__{cls_name}'
    val = row.get(key, '[]')
    try:
        entries = ast.literal_eval(val)
        if entries:
            return int(entries[0]['value'])
    except:
        pass
    return 0

def get_primary_label(labels):
    for cls in PRIORITY:
        if labels.get(cls, 0) == 1:
            return cls
    return None

# Build 3-class dataset
segments = []
for row in manifest:
    if row.get('feature_row_available') != 'True':
        continue
    if int(row.get('annotation_match_count', '0')) == 0:
        continue
    seg_id = row['segment_id']
    if seg_id not in feat_id_to_idx:
        continue
    
    all_labels = {
        'Aggressive Behavior': get_label(row, 'Aggressive Behavior'),
        'Hyper- or Hyporeactivity to Sensory Input': get_label(row, 'Hyper- or Hyporeactivity to Sensory Input'),
        'Upper Limb Stereotypies': get_label(row, 'Upper Limb Stereotypies'),
        'Self-Spinning or Spinning Objects': get_label(row, 'Self-Spinning or Spinning Objects'),
    }
    
    pl = get_primary_label(all_labels)
    if pl is None or pl not in CLS3:
        continue
    
    segments.append({
        'seg_id': seg_id,
        'video_id': row['video_id'],
        'feat_idx': feat_id_to_idx[seg_id],
        'primary_label': pl,
        'duration': float(row.get('duration_sec', 1.0))
    })

label_map = {c: i for i, c in enumerate(CLS3)}
X_raw = np.array([X_audio_all[s['feat_idx']] for s in segments], dtype=np.float32)
y_arr = np.array([label_map[s['primary_label']] for s in segments], dtype=int)
grp   = np.array([s['video_id'] for s in segments])
dur   = np.array([s['duration'] for s in segments], dtype=np.float32)

print(f"  3-class segments: {len(segments)}", flush=True)
from collections import Counter
print(f"  Distribution: {Counter(s['primary_label'] for s in segments)}", flush=True)
print(f"  Unique videos: {len(np.unique(grp))}", flush=True)

# ─── Advanced Feature Engineering ─────────────────────────────────────────────
print("\n[2] Advanced feature engineering ...", flush=True)

def engineer_features_v2(X, dur_arr):
    """Enhanced feature engineering for AV-ASD audio segments."""
    # Base features (32 dims)
    # Log transform
    X_log = np.log1p(np.abs(X))
    # Squared
    X_sq = X ** 2
    # L2-normalized
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6)
    # Relative energy (L1-normalized)
    X_rel = X / (np.abs(X).sum(axis=1, keepdims=True) + 1e-6)
    
    # Row-wise statistics
    row_mean = X.mean(axis=1, keepdims=True)
    row_std  = X.std(axis=1, keepdims=True)
    row_max  = X.max(axis=1, keepdims=True)
    row_min  = X.min(axis=1, keepdims=True)
    row_range= (row_max - row_min)
    row_skew = np.mean((X - row_mean)**3, axis=1, keepdims=True) / (row_std**3 + 1e-6)
    row_kurt = np.mean((X - row_mean)**4, axis=1, keepdims=True) / (row_std**4 + 1e-6)
    
    # Half-band ratios: low vs high frequency bands
    half = X.shape[1] // 2
    low_energy  = np.abs(X[:, :half]).sum(axis=1, keepdims=True)
    high_energy = np.abs(X[:, half:]).sum(axis=1, keepdims=True)
    band_ratio  = low_energy / (high_energy + 1e-6)
    
    # Duration feature
    dur_feat = dur_arr.reshape(-1, 1)
    
    # Pairwise feature differences (adjacent features)
    X_diff = np.diff(X, axis=1)  # (N, 31)
    X_diff_abs = np.abs(X_diff)
    
    combined = np.hstack([
        X,          # 32
        X_log,      # 32
        X_sq,       # 32
        X_norm,     # 32
        X_rel,      # 32
        row_mean, row_std, row_max, row_min, row_range, row_skew, row_kurt,  # 7
        low_energy, high_energy, band_ratio,  # 3
        dur_feat,   # 1
        X_diff,     # 31
        X_diff_abs, # 31
    ])
    return np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)

X_eng = engineer_features_v2(X_raw, dur)
print(f"  Engineered features: {X_eng.shape[1]} dims", flush=True)

# ─── Prototype features ─────────────────────────────────────────────────────
def compute_proto_features(X_tr, y_tr, X_te, n_cls):
    centroids = np.array([
        X_tr[y_tr == c].mean(axis=0) if np.sum(y_tr == c) > 0 else np.zeros(X_tr.shape[1])
        for c in range(n_cls)
    ])
    def dist(X, C):
        diffs = X[:, None, :] - C[None, :, :]
        return np.sqrt(np.sum(diffs**2, axis=-1)) / (X.shape[1]**0.5 + 1e-6)
    return dist(X_tr, centroids), dist(X_te, centroids)

# ─── Evaluation ───────────────────────────────────────────────────────────────
print("\n[3] Running 5-fold GroupKFold evaluation ...", flush=True)
gkf = GroupKFold(n_splits=5)
n_cls = 3

fold_accs, fold_f1s = [], []
all_preds, all_true = [], []

for fold_idx, (tr_idx, te_idx) in enumerate(gkf.split(X_eng, y_arr, groups=grp)):
    X_tr, X_te = X_eng[tr_idx], X_eng[te_idx]
    y_tr, y_te = y_arr[tr_idx], y_arr[te_idx]
    
    if len(np.unique(y_tr)) < 2:
        print(f"  Fold {fold_idx+1}: Skipped", flush=True)
        continue
    
    # Scale
    scaler = RobustScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    
    # Feature selection (top-60 features)
    k = min(60, X_tr.shape[1], X_tr.shape[0]//5)
    sel = SelectKBest(f_classif, k=k)
    X_tr = sel.fit_transform(X_tr, y_tr)
    X_te = sel.transform(X_te)
    
    # Prototype features (ProtoGCN-inspired)
    proto_tr, proto_te = compute_proto_features(X_tr, y_tr, X_te, n_cls)
    X_tr_aug = np.hstack([X_tr, proto_tr])
    X_te_aug = np.hstack([X_te, proto_te])
    
    # Class weights — extra boost for minority class (Hyper-reactivity)
    from sklearn.utils.class_weight import compute_class_weight
    cw_arr = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
    # Boost Hyper-reactivity (class 1) even more
    cw_dict = {int(c): float(w) for c, w in zip(np.unique(y_tr), cw_arr)}
    if 1 in cw_dict:
        cw_dict[1] = cw_dict[1] * 1.5  # extra 50% weight for Hyper-reactivity
    sw = np.array([cw_dict.get(c, 1.0) for c in y_tr])
    
    # RF
    rf = RandomForestClassifier(n_estimators=800, max_features='sqrt',
                                 max_depth=None, class_weight='balanced',
                                 n_jobs=-1, random_state=42)
    rf.fit(X_tr_aug, y_tr)
    rf_prob = rf.predict_proba(X_te_aug)
    
    # ET
    et = ExtraTreesClassifier(n_estimators=800, max_features='sqrt',
                               max_depth=None, class_weight='balanced',
                               n_jobs=-1, random_state=42)
    et.fit(X_tr_aug, y_tr)
    et_prob = et.predict_proba(X_te_aug)
    
    # XGBoost
    present = np.unique(y_tr)
    xmap  = {c: i for i, c in enumerate(present)}
    xrmap = {i: c for c, i in xmap.items()}
    y_tr_x = np.array([xmap[c] for c in y_tr])
    
    xgb = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.06,
                         subsample=0.85, colsample_bytree=0.85,
                         min_child_weight=1, reg_alpha=0.3, reg_lambda=1.0,
                         tree_method='hist', verbosity=0, random_state=42,
                         n_jobs=-1, use_label_encoder=False)
    xgb.fit(X_tr_aug, y_tr_x, sample_weight=sw)
    xgb_prob_raw = xgb.predict_proba(X_te_aug)
    xgb_prob = np.zeros((len(te_idx), n_cls))
    for li, gc in xrmap.items():
        if gc < n_cls:
            xgb_prob[:, gc] = xgb_prob_raw[:, li]
    
    # Soft voting ensemble
    ens = 0.35 * rf_prob + 0.35 * et_prob + 0.30 * xgb_prob
    preds = np.argmax(ens, axis=1)
    
    acc = accuracy_score(y_te, preds)
    f1  = f1_score(y_te, preds, average='macro', zero_division=0)
    fold_accs.append(acc)
    fold_f1s.append(f1)
    all_preds.extend(preds.tolist())
    all_true.extend(y_te.tolist())
    print(f"  Fold {fold_idx+1}: Acc={acc*100:.2f}% F1={f1:.4f} (n_test={len(te_idx)})", flush=True)

overall_acc = accuracy_score(all_true, all_preds)
overall_f1  = f1_score(all_true, all_preds, average='macro', zero_division=0)

print(f"\n{'='*70}", flush=True)
print(f"AV-ASD 3-CLASS HYPER-TUNED OVERALL", flush=True)
print(f"Accuracy : {overall_acc*100:.2f}%  Macro-F1 : {overall_f1:.4f}", flush=True)
print(f"Previous best: 74.61% (Acc)  55.90% (old pipeline)", flush=True)
print(f"{'='*70}", flush=True)
print(classification_report(all_true, all_preds, target_names=CLS3, zero_division=0), flush=True)

# ─── Save ─────────────────────────────────────────────────────────────────────
prec, rec, fscore, supp = precision_recall_fscore_support(all_true, all_preds, zero_division=0)
results = {
    "dataset": "AV-ASD 3-class Hyper-tuned Segment-Level",
    "date": "2026-11-12",
    "evaluation": "Strict 5-fold GroupKFold by video_id",
    "n_samples": len(segments),
    "n_videos": int(len(np.unique(grp))),
    "feature_dim_raw": 32,
    "feature_dim_engineered": int(X_eng.shape[1]),
    "fold_results": [{"fold": i+1, "accuracy": float(a), "macro_f1": float(f)}
                      for i, (a, f) in enumerate(zip(fold_accs, fold_f1s))],
    "overall": {
        "accuracy": float(overall_acc),
        "macro_f1": float(overall_f1),
        "accuracy_pct": float(overall_acc * 100),
        "target_90_achieved": bool(overall_acc >= 0.90)
    },
    "per_class": {
        CLS3[i]: {"precision": float(prec[i]), "recall": float(rec[i]),
                  "f1": float(fscore[i]), "support": int(supp[i])}
        for i in range(3)
    },
    "disclaimer": "FOR CLINICAL DECISION-SUPPORT RESEARCH PURPOSES ONLY."
}

out_path = os.path.join(EXP_DIR, "avasd_3class_hyper_results_2026_11_12.json")
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved: {out_path}", flush=True)

# Update benchmarks
bench_path = os.path.join(RES_DIR, "benchmarks_2026_11_12.json")
with open(bench_path, 'r') as f:
    benchmarks = json.load(f)
benchmarks['datasets']['AV_ASD_3_Class_Stimming']['hyper_tuned_segment_level'] = {
    'accuracy': float(overall_acc), 'macro_f1': float(overall_f1),
    'n_samples': len(segments), 'pipeline': 'EngineeredAudio+Proto+SelectK+RF800+ET800+XGB400'
}
benchmarks['last_updated'] = "2026-11-12"
with open(bench_path, 'w') as f:
    json.dump(benchmarks, f, indent=2)
print("Updated master benchmarks", flush=True)
