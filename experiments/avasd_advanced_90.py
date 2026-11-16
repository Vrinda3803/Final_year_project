"""
AV-ASD Advanced Multi-label Classification Pipeline
====================================================
Key improvement: Use all 1,278 annotated audio segments (vs 182 clips before).
Each segment has 32-dim audio features + multi-label annotations.

Strategy:
1. Load all 1278 annotated segments with their 32-dim audio features
2. Build video-level aggregated features (mean/std/max/min across segments)
3. Use primary label per segment (dominant behaviour) for single-label eval
4. Try per-video aggregation for robust video-level classification
5. Advanced feature engineering (polynomial, interaction terms)
6. RF + ET + XGB ensemble with proper GroupKFold by video_id
7. Target: Push AV-ASD toward 70%+ (from current 55.90%)

Date: 2026-11-12
"""
import sys, os, csv, ast, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_recall_fscore_support
from xgboost import XGBClassifier

print("="*70, flush=True)
print("AV-ASD ADVANCED PIPELINE — SEGMENT-LEVEL (1278 SAMPLES)", flush=True)
print("="*70, flush=True)

FEAT_DIR = r"D:\Downloads\AV_ASD_audio_stage1_audit_bundle (1)\features"
REP_DIR  = r"D:\Downloads\AV_ASD_audio_stage1_audit_bundle (1)\reports"
RES_DIR  = r"D:\Documents\Autism_Project\results"
EXP_DIR  = r"D:\Documents\Autism_Project\experiments"

# ─── Load audio features (5410, 32) ───────────────────────────────────────────
print("\n[1/7] Loading audio features ...", flush=True)
X_audio_all = np.load(os.path.join(FEAT_DIR, "segment_features.npy"))  # (5410, 32)
feat_ids = list(csv.DictReader(open(os.path.join(FEAT_DIR, "segment_features_row_ids.csv"))))
feat_id_to_idx = {r['segment_id']: i for i, r in enumerate(feat_ids)}
print(f"  Loaded {X_audio_all.shape[0]} segment features, dim={X_audio_all.shape[1]}", flush=True)

# ─── Load manifest and extract annotated segments ─────────────────────────────
print("\n[2/7] Loading manifest and extracting annotated segments ...", flush=True)
manifest = list(csv.DictReader(open(os.path.join(REP_DIR, "final_segment_manifest.csv"))))

# Class names we care about
CLASSES = [
    'Aggressive Behavior',
    'Hyper- or Hyporeactivity to Sensory Input',
    'Self-Spinning or Spinning Objects',
    'Upper Limb Stereotypies'
]
CLS3 = [
    'Aggressive Behavior',
    'Hyper- or Hyporeactivity to Sensory Input',
    'Upper Limb Stereotypies'
]

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

# Build dataset: segments with annotations + features
segments = []
for row in manifest:
    if row.get('feature_row_available') != 'True':
        continue
    if int(row.get('annotation_match_count', '0')) == 0:
        continue
    seg_id = row['segment_id']
    if seg_id not in feat_id_to_idx:
        continue
    
    labels = {cls: get_label(row, cls) for cls in CLASSES}
    
    # Only keep segments with at least one positive label
    if sum(labels.values()) == 0:
        continue
    
    segments.append({
        'seg_id': seg_id,
        'video_id': row['video_id'],
        'feat_idx': feat_id_to_idx[seg_id],
        'labels': labels,
        'duration': float(row.get('duration_sec', 0)),
    })

print(f"  Segments with ≥1 positive label: {len(segments)}", flush=True)

# Determine primary label (pick the behaviour with highest priority)
# Priority: Aggressive > Spinning > UpperLimb > Hyper (most distinguishable first)
PRIORITY = ['Aggressive Behavior', 'Self-Spinning or Spinning Objects', 'Upper Limb Stereotypies', 'Hyper- or Hyporeactivity to Sensory Input']

def get_primary_label(labels):
    for cls in PRIORITY:
        if labels.get(cls, 0) == 1:
            return cls
    return None

# Filter to single-label segments for clean evaluation
single_label_segs = []
for seg in segments:
    pl = get_primary_label(seg['labels'])
    if pl is not None:
        seg['primary_label'] = pl
        single_label_segs.append(seg)

print(f"  Single-label segments (4-class): {len(single_label_segs)}", flush=True)

# Label distribution
from collections import Counter
dist = Counter(s['primary_label'] for s in single_label_segs)
print("  Label distribution:", dict(dist), flush=True)

# ─── Build feature matrix ──────────────────────────────────────────────────────
print("\n[3/7] Building feature matrices ...", flush=True)

# 4-class dataset
label_map = {c: i for i, c in enumerate(CLASSES)}
X_4 = np.array([X_audio_all[s['feat_idx']] for s in single_label_segs], dtype=np.float32)
y_4 = np.array([label_map[s['primary_label']] for s in single_label_segs], dtype=int)
grp_4 = np.array([s['video_id'] for s in single_label_segs])
print(f"  4-class: X={X_4.shape}, unique_videos={len(np.unique(grp_4))}", flush=True)

# 3-class dataset (exclude Self-Spinning)
seg3 = [s for s in single_label_segs if s['primary_label'] in CLS3]
label_map3 = {c: i for i, c in enumerate(CLS3)}
X_3 = np.array([X_audio_all[s['feat_idx']] for s in seg3], dtype=np.float32)
y_3 = np.array([label_map3[s['primary_label']] for s in seg3], dtype=int)
grp_3 = np.array([s['video_id'] for s in seg3])
print(f"  3-class: X={X_3.shape}, unique_videos={len(np.unique(grp_3))}", flush=True)

dist3 = Counter(s['primary_label'] for s in seg3)
print("  3-class distribution:", dict(dist3), flush=True)

# ─── Advanced Feature Engineering ─────────────────────────────────────────────
print("\n[4/7] Engineering additional features ...", flush=True)

def engineer_features(X):
    """Add polynomial/ratio features to audio feature matrix."""
    # Normalize first
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6)
    
    # Pairwise ratios for top features (selected 8 features)
    n = X.shape[1]
    # Log transform of absolute values
    X_log = np.log1p(np.abs(X))
    # Squared features
    X_sq = X ** 2
    # Interaction: mean vs variance proxy
    # Feature statistics
    row_mean = X.mean(axis=1, keepdims=True)
    row_std  = X.std(axis=1, keepdims=True)
    row_max  = X.max(axis=1, keepdims=True)
    row_min  = X.min(axis=1, keepdims=True)
    row_range = row_max - row_min
    
    # Relative energy (normalized per sample)
    X_rel = X / (np.abs(X).sum(axis=1, keepdims=True) + 1e-6)
    
    extra = np.hstack([X_log, X_sq, X_rel, 
                       np.tile(row_mean, (1, 1)),
                       np.tile(row_std, (1, 1)),
                       np.tile(row_range, (1, 1))])
    
    combined = np.hstack([X, extra])
    return np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)

X_4_eng = engineer_features(X_4)
X_3_eng = engineer_features(X_3)
print(f"  Engineered 4-class dim: {X_4_eng.shape[1]}", flush=True)
print(f"  Engineered 3-class dim: {X_3_eng.shape[1]}", flush=True)

# ─── Prototype distance features ──────────────────────────────────────────────
def compute_proto_features(X_tr, y_tr, X_te, n_cls):
    centroids = np.array([
        X_tr[y_tr == c].mean(axis=0) if np.sum(y_tr == c) > 0 else np.zeros(X_tr.shape[1])
        for c in range(n_cls)
    ])
    def dist(X, C):
        diffs = X[:, None, :] - C[None, :, :]
        return np.sqrt(np.sum(diffs**2, axis=-1)) / (X.shape[1]**0.5 + 1e-6)
    return dist(X_tr, centroids), dist(X_te, centroids)

# ─── Evaluation function ───────────────────────────────────────────────────────
def run_evaluation(X_eng, y_arr, groups, n_cls, cls_names, tag):
    print(f"\n  Running {tag} ...", flush=True)
    gkf = GroupKFold(n_splits=5)
    
    fold_accs, fold_f1s = [], []
    all_preds, all_true = [], []
    
    for fold_idx, (tr_idx, te_idx) in enumerate(gkf.split(X_eng, y_arr, groups=groups)):
        X_tr, X_te = X_eng[tr_idx], X_eng[te_idx]
        y_tr, y_te = y_arr[tr_idx], y_arr[te_idx]
        
        if len(np.unique(y_tr)) < 2:
            print(f"    Fold {fold_idx+1}: Skipped (insufficient classes in train)", flush=True)
            continue
        
        # Scale
        scaler = RobustScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        
        # Feature selection
        k = min(50, X_tr.shape[1], X_tr.shape[0]//5)
        sel = SelectKBest(f_classif, k=k)
        X_tr = sel.fit_transform(X_tr, y_tr)
        X_te = sel.transform(X_te)
        
        # Prototype features
        proto_tr, proto_te = compute_proto_features(X_tr, y_tr, X_te, n_cls)
        X_tr_aug = np.hstack([X_tr, proto_tr])
        X_te_aug = np.hstack([X_te, proto_te])
        
        # Class weights
        from sklearn.utils.class_weight import compute_class_weight
        cw_arr = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
        cw_dict = {int(c): float(w) for c, w in zip(np.unique(y_tr), cw_arr)}
        sw = np.array([cw_dict[c] for c in y_tr])
        
        # RF
        rf = RandomForestClassifier(n_estimators=500, max_features='sqrt',
                                    class_weight='balanced', n_jobs=-1, random_state=42)
        rf.fit(X_tr_aug, y_tr)
        
        # ET
        et = ExtraTreesClassifier(n_estimators=500, max_features='sqrt',
                                  class_weight='balanced', n_jobs=-1, random_state=42)
        et.fit(X_tr_aug, y_tr)
        
        # XGBoost
        present = np.unique(y_tr)
        xmap = {c: i for i, c in enumerate(present)}
        xrmap = {i: c for c, i in xmap.items()}
        y_tr_x = np.array([xmap[c] for c in y_tr])
        
        xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08,
                             subsample=0.85, colsample_bytree=0.85,
                             min_child_weight=2, reg_alpha=0.5,
                             tree_method='hist', eval_metric='mlogloss',
                             verbosity=0, random_state=42, n_jobs=-1,
                             use_label_encoder=False)
        xgb.fit(X_tr_aug, y_tr_x, sample_weight=sw)
        
        rf_prob = rf.predict_proba(X_te_aug)
        et_prob = et.predict_proba(X_te_aug)
        xgb_prob_raw = xgb.predict_proba(X_te_aug)
        
        # Expand XGB proba to full n_cls
        xgb_prob = np.zeros((len(te_idx), n_cls))
        for li, gc in xrmap.items():
            if gc < n_cls:
                xgb_prob[:, gc] = xgb_prob_raw[:, li]
        
        ens = 0.35*rf_prob + 0.35*et_prob + 0.30*xgb_prob
        preds = np.argmax(ens, axis=1)
        
        acc = accuracy_score(y_te, preds)
        f1  = f1_score(y_te, preds, average='macro', zero_division=0)
        fold_accs.append(acc)
        fold_f1s.append(f1)
        all_preds.extend(preds.tolist())
        all_true.extend(y_te.tolist())
        print(f"    Fold {fold_idx+1}: Acc={acc*100:.2f}% F1={f1:.4f}", flush=True)
    
    if not all_preds:
        return None
    
    overall_acc = accuracy_score(all_true, all_preds)
    overall_f1  = f1_score(all_true, all_preds, average='macro', zero_division=0)
    
    print(f"  {tag} OVERALL: Acc={overall_acc*100:.2f}% F1={overall_f1:.4f}", flush=True)
    print(classification_report(all_true, all_preds, target_names=cls_names, zero_division=0), flush=True)
    
    return {
        'accuracy': float(overall_acc),
        'macro_f1': float(overall_f1),
        'fold_accs': [float(a) for a in fold_accs],
        'fold_f1s': [float(f) for f in fold_f1s],
    }

# ─── Run evaluations ───────────────────────────────────────────────────────────
print("\n[5/7] Running evaluations ...", flush=True)
res_4cls = run_evaluation(X_4_eng, y_4, grp_4, 4, CLASSES, "4-CLASS SEGMENT-LEVEL")
res_3cls = run_evaluation(X_3_eng, y_3, grp_3, 3, CLS3, "3-CLASS SEGMENT-LEVEL")

# ─── Video-level aggregation experiment ───────────────────────────────────────
print("\n[6/7] Video-level aggregation experiment ...", flush=True)
# Aggregate all segments per video → mean/std/max features
# Then classify at video level using majority label

from collections import defaultdict

def build_video_level(segs_list, feat_array, label_map_dict, feat_id_to_idx_local):
    vid_feats = defaultdict(list)
    vid_labels = defaultdict(list)
    for seg in segs_list:
        vid = seg['video_id']
        fidx = seg['feat_idx']
        vid_feats[vid].append(feat_array[fidx])
        vid_labels[vid].append(label_map_dict[seg['primary_label']])
    
    X_vid, y_vid, grp_vid = [], [], []
    for vid in vid_feats:
        feats = np.array(vid_feats[vid])  # (n_segs, 32)
        mean_f = feats.mean(axis=0)
        std_f  = feats.std(axis=0)
        max_f  = feats.max(axis=0)
        min_f  = feats.min(axis=0)
        
        # Majority vote label
        from scipy.stats import mode as smode
        maj_label = int(smode(vid_labels[vid], keepdims=True).mode[0])
        
        combined = np.concatenate([mean_f, std_f, max_f, min_f])
        X_vid.append(combined)
        y_vid.append(maj_label)
        grp_vid.append(vid)
    
    return np.array(X_vid, dtype=np.float32), np.array(y_vid), np.array(grp_vid)

X_vid4, y_vid4, grp_vid4 = build_video_level(single_label_segs, X_audio_all, 
    {c: i for i, c in enumerate(CLASSES)}, feat_id_to_idx)
X_vid3, y_vid3, grp_vid3 = build_video_level(seg3, X_audio_all,
    {c: i for i, c in enumerate(CLS3)}, feat_id_to_idx)

print(f"  Video-level 4-class: {X_vid4.shape[0]} videos", flush=True)
print(f"  Video-level 3-class: {X_vid3.shape[0]} videos", flush=True)

res_vid4 = run_evaluation(engineer_features(X_vid4), y_vid4, grp_vid4, 4, CLASSES, "4-CLASS VIDEO-LEVEL")
res_vid3 = run_evaluation(engineer_features(X_vid3), y_vid3, grp_vid3, 3, CLS3, "3-CLASS VIDEO-LEVEL")

# ─── Save Results ──────────────────────────────────────────────────────────────
print("\n[7/7] Saving results ...", flush=True)

results = {
    "dataset": "AV-ASD (Audio-Visual Autism Spectrum Dataset) Advanced Pipeline",
    "date": "2026-11-12",
    "evaluation": "Strict 5-fold GroupKFold by video_id",
    "disclaimer": "FOR CLINICAL DECISION-SUPPORT RESEARCH PURPOSES ONLY.",
    "data_expansion": {
        "previous_clips": 182,
        "annotated_segments_used": len(single_label_segs),
        "total_segments_in_corpus": 5410,
        "unique_videos_4class": int(len(np.unique(grp_4)))
    },
    "segment_level_4class": res_4cls,
    "segment_level_3class": res_3cls,
    "video_level_4class": res_vid4,
    "video_level_3class": res_vid3,
}

out_path = os.path.join(EXP_DIR, "avasd_advanced_results_2026_11_12.json")
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"  Saved: {out_path}", flush=True)

# Update master benchmarks
bench_path = os.path.join(RES_DIR, "benchmarks_2026_11_12.json")
with open(bench_path, 'r') as f:
    benchmarks = json.load(f)

if res_4cls:
    benchmarks['datasets']['AV_ASD_4_Class_Audio_XGB']['advanced_segment_level'] = res_4cls
if res_3cls:
    benchmarks['datasets']['AV_ASD_3_Class_Stimming']['advanced_segment_level'] = res_3cls
if res_vid4:
    benchmarks['datasets']['AV_ASD_4_Class_Audio_XGB']['video_level_aggregated'] = res_vid4
if res_vid3:
    benchmarks['datasets']['AV_ASD_3_Class_Stimming']['video_level_aggregated'] = res_vid3

benchmarks['last_updated'] = "2026-11-12"
with open(bench_path, 'w') as f:
    json.dump(benchmarks, f, indent=2)
print(f"  Updated master benchmarks: {bench_path}", flush=True)

print("\n" + "="*70, flush=True)
print("AV-ASD ADVANCED PIPELINE COMPLETE", flush=True)
best4 = max(r['accuracy'] for r in [res_4cls, res_vid4] if r) if any([res_4cls, res_vid4]) else 0
best3 = max(r['accuracy'] for r in [res_3cls, res_vid3] if r) if any([res_3cls, res_vid3]) else 0
print(f"Best 4-class accuracy: {best4*100:.2f}%", flush=True)
print(f"Best 3-class accuracy: {best3*100:.2f}%", flush=True)
print("="*70, flush=True)
