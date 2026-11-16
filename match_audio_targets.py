import csv
import json
import numpy as np

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

print(f"Total target clips WITH AUDIO: {len(matched_targets)}")
from collections import Counter
c = Counter([m["label"] for m in matched_targets])
for k, v in c.items():
    print(f"  {k}: {v}")
