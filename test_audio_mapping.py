import csv
import json
import numpy as np

# Load segment features and row IDs
seg_feats = np.load(r"D:\Downloads\AV_ASD_audio_stage1_audit_bundle (1)\features\segment_features.npy")
with open(r"D:\Downloads\AV_ASD_audio_stage1_audit_bundle (1)\features\segment_features_row_ids.csv", "r", encoding="utf-8") as f:
    row_ids = [line.strip() for line in f if line.strip() and line.strip() != "segment_id"]

seg_id_to_feat = {sid: seg_feats[i] for i, sid in enumerate(row_ids) if i < len(seg_feats)}
print(f"Total mapped audio segments: {len(seg_id_to_feat)}")

# Map segments to Video_ID from alignment manifest
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

print(f"Total video clips with matched audio features: {len(clip_to_audio_feats)}")

# Check mean audio feature shape
for clip, feats in list(clip_to_audio_feats.items())[:5]:
    mean_feat = np.mean(feats, axis=0)
    print(f"Clip: {clip} -> {len(feats)} segments, pooled feature shape: {mean_feat.shape}")
