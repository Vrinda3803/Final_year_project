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

# Load audio
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

le = LabelEncoder()
y = le.fit_transform(y_labels)

print(f"Dataset for evaluation: N={len(y)}, unique subjects={len(set(groups))}")

# Run 5-fold GroupKFold with in-fold scaling
gkf = GroupKFold(n_splits=5)
y_trues, y_preds = [], []

for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_aud, y, groups=groups)):
    X_tr, X_te = X_aud[tr_idx], X_aud[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=300, max_depth=8, class_weight='balanced', random_state=42))
    ])
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    
    y_trues.extend(y_te)
    y_preds.extend(pred)

acc = accuracy_score(y_trues, y_preds)
f1 = f1_score(y_trues, y_preds, average="macro")
print(f"=== AUDIO-ONLY BASELINE (GroupKFold-5) ===")
print(f"Accuracy: {acc:.4f}  |  Macro-F1: {f1:.4f}")
print(classification_report(y_trues, y_preds, target_names=le.classes_))
