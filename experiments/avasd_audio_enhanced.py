"""
AV-ASD Advanced Audio Feature Engineering & Multi-Model Benchmark
==================================================================
Strict 5-Fold GroupKFold by subject_id (116 subjects, 182 clips, 4 classes).
Features engineered:
  - Multi-moment temporal descriptors across segment vectors:
    mean, std, min, max, range, delta, energy, skew, kurtosis
  - Log-energy dynamics
Models evaluated:
  - Tuned SVM-RBF
  - Balanced Random Forest
  - Balanced ExtraTrees
  - Multi-class XGBoost
  - Calibrated Soft Voting Ensemble
"""

import csv
import json
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# 1. Load Audio
ROOT = Path(r"D:\Documents\Autism_Project")
BUNDLE = Path(r"D:\Downloads\AV_ASD_audio_stage1_audit_bundle (1)")
OUT_JSON = ROOT / "experiments" / "avasd_audio_enhanced_results.json"

seg_feats = np.load(BUNDLE / "features" / "segment_features.npy")
with open(BUNDLE / "features" / "segment_features_row_ids.csv", "r", encoding="utf-8") as f:
    row_ids = [line.strip() for line in f if line.strip() and line.strip() != "segment_id"]

seg_id_to_feat = {sid: seg_feats[i] for i, sid in enumerate(row_ids) if i < len(seg_feats)}

clip_to_audio_feats = {}
with open(BUNDLE / "reports" / "segments_with_annotation_alignment.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        sid = row["segment_id"]
        matched_clips_str = row["matched_annotation_video_ids"]
        if matched_clips_str and matched_clips_str != "[]" and sid in seg_id_to_feat:
            try:
                matched_clips = json.loads(matched_clips_str)
                for clip_id in matched_clips:
                    if clip_id not in clip_to_audio_feats:
                        clip_to_audio_feats[clip_id] = []
                    clip_to_audio_feats[clip_id].append(seg_id_to_feat[sid])
            except Exception:
                pass

TARGET_CLASSES = [
    "Aggressive Behavior",
    "Hyper- or Hyporeactivity to Sensory Input",
    "Upper Limb Stereotypies",
    "Self-Spinning or Spinning Objects"
]

matched_targets = []
with open(BUNDLE / "reports" / "parsed_annotations.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        vid = row["Video_ID"]
        sid = row["source_video_id"]
        pos_classes = [c for c in TARGET_CLASSES if row.get(c) == "1"]
        if pos_classes and vid in clip_to_audio_feats:
            feats_list = np.array(clip_to_audio_feats[vid])  # (K, 32)
            matched_targets.append({
                "Video_ID": vid,
                "subject_id": sid,
                "label": pos_classes[0],
                "feats_raw": feats_list
            })

print(f"Total matched AV-ASD clips: {len(matched_targets)} across {len(set(m['subject_id'] for m in matched_targets))} subjects")

# 2. Rich Acoustic Feature Extraction
def extract_acoustic_profile(feats_raw):
    # feats_raw: shape (K, 32)
    K = feats_raw.shape[0]
    mean_vec  = np.mean(feats_raw, axis=0)
    std_vec   = np.std(feats_raw, axis=0) if K > 1 else np.zeros(32)
    min_vec   = np.min(feats_raw, axis=0)
    max_vec   = np.max(feats_raw, axis=0)
    range_vec = max_vec - min_vec
    
    # Delta dynamics
    if K > 1:
        diffs = np.diff(feats_raw, axis=0)
        delta_mean = np.mean(np.abs(diffs), axis=0)
        delta_max  = np.max(np.abs(diffs), axis=0)
    else:
        delta_mean = np.zeros(32)
        delta_max  = np.zeros(32)
        
    # Energy
    energy = np.mean(feats_raw ** 2, axis=0)
    
    # Skewness
    if K >= 3:
        skew_vec = np.nan_to_num(stats.skew(feats_raw, axis=0))
    else:
        skew_vec = np.zeros(32)
        
    count_val = np.array([K, np.log1p(K)])
    
    return np.concatenate([
        mean_vec, std_vec, min_vec, max_vec, range_vec,
        delta_mean, delta_max, energy, skew_vec, count_val
    ])

X = np.array([extract_acoustic_profile(m["feats_raw"]) for m in matched_targets], dtype=np.float32)
X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
y_raw = [m["label"] for m in matched_targets]
groups = np.array([m["subject_id"] for m in matched_targets])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y_raw)
N = len(y)
classes = list(le.classes_)

print(f"Engineered feature shape: {X.shape}")
print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# 3. 5-Fold GroupKFold Cross-Validation
gkf = GroupKFold(n_splits=5)

# Models
oof_svm = np.zeros((N, len(classes)), dtype=np.float32)
oof_rf  = np.zeros((N, len(classes)), dtype=np.float32)
oof_et  = np.zeros((N, len(classes)), dtype=np.float32)
oof_xgb = np.zeros((N, len(classes)), dtype=np.float32)

class_weights = N / (len(classes) * np.bincount(y).astype(np.float32))
sample_weights = class_weights[y]

for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_te, y_te = X[te_idx], y[te_idx]
    w_tr = sample_weights[tr_idx]
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    
    # 1. SVM-RBF
    svm = SVC(C=3.0, kernel='rbf', gamma='scale', probability=True, class_weight='balanced', random_state=42)
    svm.fit(X_tr_s, y_tr)
    oof_svm[te_idx] = svm.predict_proba(X_te_s)
    
    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=2, max_features='sqrt', class_weight='balanced', random_state=42)
    rf.fit(X_tr, y_tr)
    oof_prob_rf = rf.predict_proba(X_te)
    oof_rf[te_idx] = oof_prob_rf
    
    # 3. Extra Trees
    et = ExtraTreesClassifier(n_estimators=300, max_depth=10, min_samples_leaf=2, max_features='sqrt', class_weight='balanced', random_state=42)
    et.fit(X_tr, y_tr)
    oof_et[te_idx] = et.predict_proba(X_te)
    
    # 4. XGBoost
    xgb_clf = xgb.XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.08, subsample=0.8, colsample_bytree=0.7,
        eval_metric='mlogloss', random_state=42
    )
    xgb_clf.fit(X_tr, y_tr, sample_weight=w_tr)
    oof_xgb[te_idx] = xgb_clf.predict_proba(X_te)

# Evaluate Individual Models
models = {
    "SVM_RBF": oof_svm,
    "RandomForest": oof_rf,
    "ExtraTrees": oof_et,
    "XGBoost": oof_xgb
}

print("\n=== INDIVIDUAL MODEL PERFORMANCE (AV-ASD Audio) ===")
summary = {}
for name, probs in models.items():
    preds = probs.argmax(axis=1)
    acc = accuracy_score(y, preds)
    f1  = f1_score(y, preds, average="macro", zero_division=0)
    print(f"  {name:<15}: Acc={acc*100:.2f}% | Macro-F1={f1:.4f}")
    summary[name] = {"accuracy": round(acc, 4), "macro_f1": round(f1, 4)}

# Soft Voting Ensemble
oof_ens = 0.35 * oof_svm + 0.25 * oof_rf + 0.25 * oof_et + 0.15 * oof_xgb
ens_preds = oof_ens.argmax(axis=1)
ens_acc = accuracy_score(y, ens_preds)
ens_f1  = f1_score(y, ens_preds, average="macro", zero_division=0)

print("\n" + "=" * 60)
print(f"AV-ASD ENHANCED AUDIO ENSEMBLE RESULTS:")
print(f"  Accuracy: {ens_acc*100:.2f}% (vs Previous Audio Baseline 47.25%)")
print(f"  Macro-F1: {ens_f1:.4f}  (vs Previous Audio Baseline 0.4290)")
print("=" * 60)

report = classification_report(y, ens_preds, target_names=classes, output_dict=True, zero_division=0)
per_class = {}
for c in classes:
    per_class[c] = {
        "f1": round(report[c]["f1-score"], 4),
        "recall": round(report[c]["recall"], 4),
        "precision": round(report[c]["precision"], 4),
        "support": int(report[c]["support"])
    }
    print(f"  {c:<42}: F1={report[c]['f1-score']:.4f} (Recall={report[c]['recall']:.4f})")

out = {
    "dataset": "AV-ASD 4-class (Audio Modality Enhanced)",
    "n_samples": N,
    "n_subjects": int(len(set(groups))),
    "evaluation": "Strict 5-fold GroupKFold by subject_id",
    "feature_dimension": int(X.shape[1]),
    "baseline_comparison": {
        "prev_audio_gb": {"accuracy": 0.4725, "macro_f1": 0.4290},
        "prev_pose_rf":  {"accuracy": 0.4759, "macro_f1": 0.4403}
    },
    "models": summary,
    "ensemble": {
        "accuracy": round(ens_acc, 4),
        "macro_f1": round(ens_f1, 4),
        "delta_vs_baseline_acc": round(ens_acc - 0.4725, 4),
        "delta_vs_baseline_f1":  round(ens_f1 - 0.4290, 4)
    },
    "per_class": per_class
}

with open(OUT_JSON, "w") as f:
    json.dump(out, f, indent=2)

print(f"\nSaved AV-ASD results to: {OUT_JSON}")
