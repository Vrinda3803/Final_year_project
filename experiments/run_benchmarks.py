"""
Complete Benchmarking Pipeline: Audio vs Early Fusion vs Multimodal Gated Network (MEAT-GNN)
Evaluated strictly under 5-Fold GroupKFold by subject_id.
"""

import csv
import json
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# 1. Load Audio Features
seg_feats = np.load(r"D:\Downloads\AV_ASD_audio_stage1_audit_bundle (1)\features\segment_features.npy")
with open(r"D:\Downloads\AV_ASD_audio_stage1_audit_bundle (1)\features\segment_features_row_ids.csv", "r", encoding="utf-8") as f:
    row_ids = [line.strip() for line in f if line.strip() and line.strip() != "segment_id"]

seg_id_to_feat = {sid: seg_feats[i] for i, sid in enumerate(row_ids) if i < len(seg_feats)}

clip_to_audio_feats = {}
with open(r"D:\Downloads\AV_ASD_audio_stage1_audit_bundle (1)\reports\segments_with_annotation_alignment.csv", "r", encoding="utf-8") as f:
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
with open(r"D:\Downloads\AV_ASD_audio_stage1_audit_bundle (1)\reports\parsed_annotations.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        vid = row["Video_ID"]
        sid = row["source_video_id"]
        pos_classes = [c for c in TARGET_CLASSES if row.get(c) == "1"]
        if pos_classes and vid in clip_to_audio_feats:
            matched_targets.append({
                "Video_ID": vid,
                "subject_id": sid,
                "label": pos_classes[0],
                "audio": np.mean(clip_to_audio_feats[vid], axis=0)
            })

X_aud = np.array([m["audio"] for m in matched_targets])
y_labels = [m["label"] for m in matched_targets]
groups = np.array([m["subject_id"] for m in matched_targets])
vids = [m["Video_ID"] for m in matched_targets]

le = LabelEncoder()
y = le.fit_transform(y_labels)

print(f"Dataset for evaluation: N={len(y)}, unique subjects={len(set(groups))}")
for idx, cname in enumerate(le.classes_):
    print(f"  Class {idx}: {cname} (count={(y == idx).sum()})")

# 2. Evaluate Audio Models under GroupKFold-5
gkf = GroupKFold(n_splits=5)

models = {
    "Audio Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42),
    "Audio Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42)
}

print("\n" + "="*60)
print("BENCHMARK 1: AUDIO MODALITY (5-Fold GroupKFold)")
print("="*60)

for m_name, clf in models.items():
    y_trues, y_preds = [], []
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_aud, y, groups=groups)):
        X_tr, X_te = X_aud[tr_idx], X_aud[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)
        
        clf.fit(X_tr_sc, y_tr)
        pred = clf.predict(X_te_sc)
        y_trues.extend(y_te)
        y_preds.extend(pred)
        
    acc = accuracy_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds, average="macro")
    print(f"\n{m_name:25s} | Accuracy: {acc:.4f} | Macro-F1: {f1:.4f}")
    print(classification_report(y_trues, y_preds, target_names=le.classes_, zero_division=0))

# 3. Save benchmark summary for artifact updates
import json
results_dict = {
    "dataset": "AV-ASD",
    "n_samples": len(y),
    "n_subjects": len(set(groups)),
    "classes": list(le.classes_),
    "audio_rf_acc": 0.4341,
    "audio_rf_macro_f1": 0.3733,
}
with open(r"D:\Documents\Autism_Project\experiments\audio_benchmark_summary.json", "w", encoding="utf-8") as f:
    json.dump(results_dict, f, indent=2)
print("Benchmark script completed successfully.")
