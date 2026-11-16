"""AV-ASD audio-only multi-label clip benchmark with grouped nested tuning.

The original AV-ASD task is multi-label, while earlier local experiments
collapsed labels to a single primary class. This script restores clip-level
multi-label targets, pools the available 32-D VAD-segment descriptors to each
annotated clip, groups all clips from a source video together, and tunes
regularization and class thresholds only inside outer-training groups.

This is an audio-only modality ablation, not a reproduction of the published
audio-visual systems: the local audit bundle has no aligned video/pose features
or verified target-speaker identity.
"""
from __future__ import annotations

import ast
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".codex-deps"))
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

BUNDLE = Path(r"D:\Downloads\AV_ASD_audio_stage1_audit_bundle (1)")
FEATURES = BUNDLE / "features" / "segment_features.npy"
FEATURE_IDS = BUNDLE / "features" / "segment_features_row_ids.csv"
ALIGNMENT = BUNDLE / "reports" / "segments_with_annotation_alignment.csv"
ANNOTATIONS = BUNDLE / "reports" / "parsed_annotations.csv"
OUT_JSON = ROOT / "results" / "avasd_multilabel_audio_grouped_2026_11_12.json"
OUT_FOLDS = ROOT / "results" / "avasd_multilabel_audio_grouped_folds_2026_11_12.csv"
OUT_OOF = ROOT / "results" / "avasd_multilabel_audio_grouped_oof_2026_11_12.csv"

SEED = 42
N_FOLDS = 5
BEHAVIOR_LABELS = [
    "Absence or Avoidance of Eye Contact",
    "Aggressive Behavior",
    "Hyper- or Hyporeactivity to Sensory Input",
    "Non-Responsiveness to Verbal Interaction",
    "Non-Typical Language",
    "Object Lining-Up",
    "Self-Hitting or Self-Injurious Behavior",
    "Self-Spinning or Spinning Objects",
    "Upper Limb Stereotypies",
]
ALL_LABELS = BEHAVIOR_LABELS + ["Background"]
C_VALUES = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60)


def parse_list(value: str) -> list:
    try:
        v = ast.literal_eval(value or "[]")
        return v if isinstance(v, list) else []
    except (ValueError, SyntaxError, TypeError):
        return []


def clip_descriptor(rows: list[dict], feature_by_id: dict[str, np.ndarray]) -> np.ndarray:
    events = []
    for row in rows:
        seg = feature_by_id.get(row.get("segment_id", ""))
        if seg is None:
            continue
        try:
            start = float(row.get("start_sec") or 0.0)
            duration = max(0.0, float(row.get("duration_sec") or 0.0))
        except ValueError:
            start, duration = 0.0, 0.0
        events.append((start, duration, seg))
    events.sort(key=lambda z: z[0])
    x = np.stack([e[2] for e in events]).astype(np.float32)
    duration = np.asarray([e[1] for e in events], dtype=np.float32)
    parts = [x.mean(0), x.std(0), x.min(0), x.max(0)]
    # Ordered temporal bins preserve coarse progression without requiring raw audio.
    for index_group in np.array_split(np.arange(len(x)), 4):
        # Short clips can leave a temporal bin empty; encode it as zero rather
        # than allowing NaNs to enter the training matrix.
        parts.append(x[index_group].mean(0) if len(index_group) else np.zeros(x.shape[1], dtype=np.float32))
    parts.append(np.asarray([
        np.log1p(len(events)),
        float(np.mean(duration)),
        float(np.std(duration)),
    ], dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


def load_clips():
    x_all = np.load(FEATURES).astype(np.float32)
    with FEATURE_IDS.open(newline="", encoding="utf-8-sig") as f:
        feat_rows = list(csv.DictReader(f))
    feat_index = {r["segment_id"]: i for i, r in enumerate(feat_rows)}
    feature_by_id = {seg_id: x_all[i] for seg_id, i in feat_index.items() if i < len(x_all)}

    # Each audio segment may overlap more than one annotated clip. All linked
    # clips share their source-video group, preventing cross-fold segment reuse.
    segments_by_clip: dict[str, list[dict]] = defaultdict(list)
    with ALIGNMENT.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("segment_id") not in feature_by_id:
                continue
            if row.get("annotation_alignment_status") != "matched_temporally":
                continue
            for clip_id in parse_list(row.get("matched_annotation_video_ids", "[]")):
                segments_by_clip[str(clip_id)].append(row)

    items = []
    with ANNOTATIONS.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            clip_id = row.get("Video_ID", "")
            segs = segments_by_clip.get(clip_id, [])
            if not segs:
                continue
            y = []
            for label in ALL_LABELS:
                try:
                    y.append(int(row.get(label, "0") or 0))
                except ValueError:
                    y.append(0)
            items.append({
                "clip_id": clip_id,
                "group": row.get("source_video_id") or clip_id,
                "X": clip_descriptor(segs, feature_by_id),
                "y": np.asarray(y, dtype=np.int64),
                "n_segments": len(segs),
            })
    X = np.stack([r["X"] for r in items])
    Y = np.stack([r["y"] for r in items])
    groups = np.asarray([r["group"] for r in items])
    return items, X, Y, groups


def fit_models(X_fit, Y_fit, X_eval, c_value: float) -> np.ndarray:
    scaler = StandardScaler().fit(X_fit)
    a = scaler.transform(X_fit)
    b = scaler.transform(X_eval)
    probs = np.zeros((len(X_eval), len(ALL_LABELS)), dtype=np.float64)
    for j in range(len(ALL_LABELS)):
        y = Y_fit[:, j]
        if len(np.unique(y)) < 2:
            probs[:, j] = float(y[0]) if len(y) else 0.0
            continue
        clf = LogisticRegression(
            C=c_value, class_weight="balanced", solver="liblinear",
            max_iter=2000, random_state=SEED,
        )
        clf.fit(a, y)
        probs[:, j] = clf.predict_proba(b)[:, 1]
    return probs


def choose_thresholds(y_true: np.ndarray, probs: np.ndarray) -> np.ndarray:
    thresholds = np.full(len(ALL_LABELS), 0.5, dtype=np.float64)
    for j in range(len(ALL_LABELS)):
        if len(np.unique(y_true[:, j])) < 2 or int(y_true[:, j].sum()) < 2:
            continue
        best_f, best_t = -1.0, 0.5
        for t in THRESHOLDS:
            pred = (probs[:, j] >= t).astype(int)
            f = f1_score(y_true[:, j], pred, zero_division=0)
            if f > best_f:
                best_f, best_t = f, t
        thresholds[j] = best_t
    return thresholds


def metric_block(y_true: np.ndarray, probs: np.ndarray, thresholds: np.ndarray) -> dict:
    pred = (probs >= thresholds[None, :]).astype(np.int64)
    pred05 = (probs >= 0.5).astype(np.int64)
    f1s = f1_score(y_true, pred, average=None, zero_division=0)
    f105 = f1_score(y_true, pred05, average=None, zero_division=0)
    prec, rec, _, support = precision_recall_fscore_support(
        y_true, pred, average=None, zero_division=0
    )
    behavior = np.arange(len(BEHAVIOR_LABELS))
    try:
        ap = average_precision_score(y_true[:, behavior], probs[:, behavior], average="macro")
    except ValueError:
        ap = float("nan")
    return {
        "threshold_tuned": {
            "behavior_macro_f1": float(np.mean(f1s[behavior])),
            "all10_macro_f1": float(np.mean(f1s)),
            "behavior_micro_f1": float(f1_score(y_true[:, behavior], pred[:, behavior], average="micro", zero_division=0)),
            "behavior_macro_average_precision": float(ap),
            "exact_match_10": float(np.mean(np.all(y_true == pred, axis=1))),
            "per_label": {
                name: {"precision": float(prec[j]), "recall": float(rec[j]), "f1": float(f1s[j]),
                      "f1_at_0_5": float(f105[j]), "support": int(support[j]), "threshold": float(thresholds[j])}
                for j, name in enumerate(ALL_LABELS)
            },
        },
        "fixed_0_5": {
            "behavior_macro_f1": float(np.mean(f105[behavior])),
            "all10_macro_f1": float(np.mean(f105)),
            "behavior_micro_f1": float(f1_score(y_true[:, behavior], pred05[:, behavior], average="micro", zero_division=0)),
        },
    }


def main():
    items, X, Y, groups = load_clips()
    outer = GroupKFold(n_splits=N_FOLDS)
    oof_probs = np.zeros_like(Y, dtype=np.float64)
    oof_thresholds = np.full_like(Y, 0.5, dtype=np.float64)
    fold_rows = []
    oof_rows = []
    for fold, (outer_train, test) in enumerate(outer.split(X, Y, groups), 1):
        # Inner grouped CV: hold out one of four folds from outer training.
        inner_groups = groups[outer_train]
        inner = GroupKFold(n_splits=4)
        inner_fit_local, inner_val_local = next(inner.split(X[outer_train], Y[outer_train], inner_groups))
        fit_idx = outer_train[inner_fit_local]
        val_idx = outer_train[inner_val_local]

        best = None
        for c in C_VALUES:
            val_probs = fit_models(X[fit_idx], Y[fit_idx], X[val_idx], c)
            th = choose_thresholds(Y[val_idx], val_probs)
            pred = (val_probs >= th[None, :]).astype(int)
            val_macro = float(np.mean(f1_score(Y[val_idx, :len(BEHAVIOR_LABELS)],
                                                pred[:, :len(BEHAVIOR_LABELS)], average=None, zero_division=0)))
            key = (val_macro, -abs(np.log10(c)))
            if best is None or key > best[0]:
                best = (key, c, th, val_macro)
        _, chosen_c, chosen_thresholds, inner_score = best
        probs = fit_models(X[outer_train], Y[outer_train], X[test], chosen_c)
        oof_probs[test] = probs
        oof_thresholds[test] = chosen_thresholds[None, :]
        fold_metrics = metric_block(Y[test], probs, chosen_thresholds)
        fold_rows.append({
            "fold": fold, "train_clips": len(outer_train), "test_clips": len(test),
            "train_source_video_groups": len(np.unique(groups[outer_train])),
            "test_source_video_groups": len(np.unique(groups[test])),
            "selected_C": chosen_c, "inner_validation_behavior_macro_f1": inner_score,
            "behavior_macro_f1_tuned": fold_metrics["threshold_tuned"]["behavior_macro_f1"],
            "behavior_macro_f1_fixed_0_5": fold_metrics["fixed_0_5"]["behavior_macro_f1"],
            "behavior_micro_f1_tuned": fold_metrics["threshold_tuned"]["behavior_micro_f1"],
            "all10_macro_f1_tuned": fold_metrics["threshold_tuned"]["all10_macro_f1"],
        })
        for ix in test:
            row = {"fold": fold, "clip_id": items[ix]["clip_id"], "source_video_id": groups[ix],
                   "n_audio_segments": items[ix]["n_segments"]}
            for j, label in enumerate(ALL_LABELS):
                row[f"true__{j}"] = int(Y[ix, j])
                row[f"prob__{j}"] = float(probs[np.where(test == ix)[0][0], j])
                row[f"pred__{j}"] = int(probs[np.where(test == ix)[0][0], j] >= chosen_thresholds[j])
            oof_rows.append(row)
        print(f"fold={fold} C={chosen_c} val_macroF1={inner_score:.3f} "
              f"test_macroF1={fold_metrics['threshold_tuned']['behavior_macro_f1']:.3f}", flush=True)

    # Thresholds differ by outer fold. Pool OOF predictions using the threshold
    # selected inside the corresponding outer-training partition.
    threshold_pred = (oof_probs >= oof_thresholds).astype(int)
    fixed_pred = (oof_probs >= 0.5).astype(int)
    behavior = slice(0, len(BEHAVIOR_LABELS))
    pooled = {
        "threshold_tuned": {
            "behavior_macro_f1": float(np.mean(f1_score(Y[:, behavior], threshold_pred[:, behavior], average=None, zero_division=0))),
            "all10_macro_f1": float(np.mean(f1_score(Y, threshold_pred, average=None, zero_division=0))),
            "behavior_micro_f1": float(f1_score(Y[:, behavior], threshold_pred[:, behavior], average="micro", zero_division=0)),
            "behavior_macro_average_precision": float(average_precision_score(Y[:, behavior], oof_probs[:, behavior], average="macro")),
            "exact_match_10": float(np.mean(np.all(Y == threshold_pred, axis=1))),
            "per_label": {},
        },
        "fixed_0_5": {
            "behavior_macro_f1": float(np.mean(f1_score(Y[:, behavior], fixed_pred[:, behavior], average=None, zero_division=0))),
            "all10_macro_f1": float(np.mean(f1_score(Y, fixed_pred, average=None, zero_division=0))),
            "behavior_micro_f1": float(f1_score(Y[:, behavior], fixed_pred[:, behavior], average="micro", zero_division=0)),
        },
    }
    p, r, f, n = precision_recall_fscore_support(Y, threshold_pred, average=None, zero_division=0)
    f05 = f1_score(Y, fixed_pred, average=None, zero_division=0)
    for j, label in enumerate(ALL_LABELS):
        pooled["threshold_tuned"]["per_label"][label] = {
            "precision": float(p[j]), "recall": float(r[j]), "f1": float(f[j]),
            "f1_at_0_5": float(f05[j]), "support": int(n[j]),
            "threshold_mean_across_outer_folds": float(np.mean([rr for rr in oof_thresholds[:, j]])),
        }

    OUT_FOLDS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FOLDS.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fold_rows[0])); w.writeheader(); w.writerows(fold_rows)
    with OUT_OOF.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(oof_rows[0])); w.writeheader(); w.writerows(oof_rows)

    class_counts = {label: int(Y[:, j].sum()) for j, label in enumerate(ALL_LABELS)}
    result = {
        "study": "AV-ASD audio-only, clip-level multi-label grouped benchmark",
        "status": "completed local modality ablation; not a reproduction of multimodal AV-ASD systems",
        "research_base": {
            "dataset_task_paper": "Hear Me, See Me, Understand Me: Audio-Visual Autism Behavior Recognition (IEEE TMM 2025)",
            "recent_context": "Multimodal behavioral analysis for autism spectrum disorder assessment (Pattern Recognition 2026)",
            "adaptation_note": "The local bundle provides only 32-D acoustic features. It lacks aligned visual/pose features, raw video, and verified target-speaker identity; this experiment preserves the published multi-label task but evaluates audio only.",
        },
        "data": {
            "annotated_clips_with_audio_features": len(items),
            "unique_source_video_groups": int(len(np.unique(groups))),
            "feature_dimension_per_segment": int(np.load(FEATURES, mmap_mode="r").shape[1]),
            "pooled_clip_feature_dimension": int(X.shape[1]),
            "label_counts": class_counts,
            "label_cardinality_mean_9_behaviors": float(Y[:, :len(BEHAVIOR_LABELS)].sum(1).mean()),
            "participant_identity": "unverified; grouping is by source_video_id, not confirmed participant",
            "speaker_identity": "unverified; diarization was disabled and target_participant_verified is false",
        },
        "method": {
            "clip_features": "mean, standard deviation, minimum, maximum and four ordered temporal-bin means over aligned VAD segment descriptors, plus segment count and duration statistics",
            "classifier": "one binary logistic regression per label; class-balanced loss",
            "nested_selection": "C and per-label decision thresholds selected using grouped validation videos inside each outer-training fold",
            "outer_evaluation": "five-fold GroupKFold by source_video_id; all clips from one source video stay in one fold",
            "thresholds": list(THRESHOLDS), "C_values": list(C_VALUES), "seed": SEED,
        },
        "metrics": {"pooled_oof": pooled, "fold_results": fold_rows},
        "limitations": [
            "This is audio-only and cannot support a multimodal performance claim.",
            "Only 227 of 928 annotated clips had usable aligned audio features in the local bundle.",
            "Source-video grouping is not verified participant-independent evaluation.",
            "The available 32-D features are MFCC/spectral summaries rather than pretrained speech representations.",
            "The target participant in the audio is not verified; audio may include caregiver or ambient sound.",
        ],
        "files": {"fold_csv": str(OUT_FOLDS.relative_to(ROOT)), "oof_csv": str(OUT_OOF.relative_to(ROOT))},
    }
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"data": result["data"], "metrics": pooled, "json": str(OUT_JSON)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
