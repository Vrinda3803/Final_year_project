"""Audio-only AV-ASD video-level experiment with PPO adaptive segment sampling.

The available local AV-ASD bundle contains 32-dimensional segment audio
features and annotations, but no aligned visual/pose feature matrix. This
script therefore tests the same sampling policy on audio segments only; it
must not be described as a multimodal AV-ASD result.

Run from the project root after installing NumPy, scikit-learn, and PyTorch:
    python experiments/avasd_adaptive_sampling_rl.py
"""
from __future__ import annotations

import ast
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".codex-deps"))
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mmasd_adaptive_sampling_rl import (  # shared PPO and equal-budget baselines
    DEVICE, SAMPLE_BUDGET, SEED, _aggregate, _baseline_select, _group_folds,
    _policy_select, _train_policy, LinearSoftmax,
)

BUNDLE = Path(r"D:\Downloads\AV_ASD_audio_stage1_audit_bundle (1)")
FEATURES = BUNDLE / "features" / "segment_features.npy"
FEATURE_IDS = BUNDLE / "features" / "segment_features_row_ids.csv"
MANIFEST = BUNDLE / "reports" / "final_segment_manifest.csv"
OUT_JSON = ROOT / "results" / "avasd_adaptive_sampling_rl_2026_11_12.json"
OUT_CSV = ROOT / "results" / "avasd_adaptive_sampling_rl_folds_2026_11_12.csv"
CLASSES = ["Aggressive Behavior", "Hyper- or Hyporeactivity to Sensory Input",
           "Upper Limb Stereotypies"]
ALL_PRIORITY = ["Aggressive Behavior", "Self-Spinning or Spinning Objects",
                "Upper Limb Stereotypies", "Hyper- or Hyporeactivity to Sensory Input"]
METHODS = ("uniform", "random", "heuristic", "ppo")


def _first_annotation(row: dict[str, str], cls: str) -> int:
    try:
        values = ast.literal_eval(row.get(f"annotation__{cls}", "[]"))
        return int(values[0]["value"]) if values else 0
    except (ValueError, SyntaxError, KeyError, TypeError):
        return 0


def _read_videos() -> tuple[list[np.ndarray], np.ndarray, np.ndarray, dict[str, Any]]:
    if not all(p.exists() for p in (FEATURES, FEATURE_IDS, MANIFEST)):
        raise FileNotFoundError(
            "The AV-ASD audio feature/annotation bundle is not present at "
            f"{BUNDLE}; update BUNDLE to the local audit bundle path."
        )
    x_all = np.load(FEATURES).astype(np.float32)
    with FEATURE_IDS.open(newline="", encoding="utf-8") as f:
        feature_rows = list(csv.DictReader(f))
    feat_idx = {r["segment_id"]: i for i, r in enumerate(feature_rows)}
    by_video: dict[str, list[tuple[float, np.ndarray, int]]] = defaultdict(list)
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("feature_row_available") != "True" or not row.get("segment_id"):
                continue
            if row["segment_id"] not in feat_idx:
                continue
            active = [c for c in ALL_PRIORITY if _first_annotation(row, c) == 1]
            if not active:
                continue
            primary = next((c for c in ALL_PRIORITY if c in active), None)
            # Match the earlier project task definition: assign the four-class
            # priority first, then exclude the spinning class for this 3-class task.
            if primary is None or primary not in CLASSES:
                continue
            try:
                start = float(row.get("start_sec", 0.0))
            except ValueError:
                start = 0.0
            by_video[row["video_id"]].append(
                (start, x_all[feat_idx[row["segment_id"]]], CLASSES.index(primary))
            )
    videos, labels, video_ids = [], [], []
    for video_id, events in sorted(by_video.items()):
        events.sort(key=lambda item: item[0])
        if not events:
            continue
        videos.append(np.asarray([e[1] for e in events], dtype=np.float32))
        counts = Counter(e[2] for e in events)
        labels.append(counts.most_common(1)[0][0])
        video_ids.append(video_id)
    return videos, np.asarray(labels, dtype=np.int64), np.asarray(video_ids), {
        "videos": len(videos), "annotated_segments": int(sum(map(len, videos))),
        "classes": {CLASSES[i]: int(np.sum(np.asarray(labels) == i)) for i in range(len(CLASSES))},
        "audio_feature_dim": int(x_all.shape[1]),
    }


def main() -> None:
    videos, labels, video_ids, data_info = _read_videos()
    # One sample per video; video IDs are held out. Participant IDs in this
    # feature bundle are not verified, so this is not subject-independent CV.
    folds = _group_folds(video_ids, SEED)
    fold_rows: list[dict[str, Any]] = []
    for fold, (tr, te) in enumerate(folds, 1):
        train_concat = np.concatenate([videos[i] for i in tr], axis=0)
        mean, std = train_concat.mean(0), train_concat.std(0)
        std[std < 1e-6] = 1.0
        seqs = [(v - mean) / std for v in videos]
        policy = _train_policy([seqs[i] for i in tr], SEED + fold)
        for method in METHODS:
            xtr, xte = [], []
            selected_counts = []
            for i in tr:
                ids = _policy_select(policy, seqs[i]) if method == "ppo" else _baseline_select(
                    seqs[i], method, SEED + fold + i)
                xtr.append(_aggregate(seqs[i], ids))
                selected_counts.append(len(ids))
            for i in te:
                ids = _policy_select(policy, seqs[i]) if method == "ppo" else _baseline_select(
                    seqs[i], method, SEED + fold + i)
                xte.append(_aggregate(seqs[i], ids))
                selected_counts.append(len(ids))
            xtr, xte = np.asarray(xtr), np.asarray(xte)
            xm, xs = xtr.mean(0), xtr.std(0)
            xs[xs < 1e-6] = 1.0
            xtr, xte = (xtr - xm) / xs, (xte - xm) / xs
            classes = np.unique(labels[tr])
            to_local = {int(c): j for j, c in enumerate(classes)}
            ytr = np.asarray([to_local[int(y)] for y in labels[tr]])
            clf = LinearSoftmax(xtr.shape[1], len(classes), SEED + fold)
            clf.fit(xtr, ytr)
            pred = classes[clf.predict(xte)]
            acc = float(np.mean(labels[te] == pred))
            class_f1 = []
            for c in np.unique(np.concatenate([labels[te], pred])):
                tp = int(np.sum((labels[te] == c) & (pred == c)))
                fp = int(np.sum((labels[te] != c) & (pred == c)))
                fn = int(np.sum((labels[te] == c) & (pred != c)))
                denom = 2 * tp + fp + fn
                class_f1.append(2 * tp / denom if denom else 0.0)
            fold_rows.append({
                "fold": fold, "method": method, "train_videos": len(tr),
                "test_videos": len(te), "budget_segments": SAMPLE_BUDGET,
                "mean_segments_selected": float(np.mean(selected_counts)),
                "accuracy": acc, "macro_f1": float(np.mean(class_f1)),
            })
        print(f"Completed AV-ASD outer fold {fold}/5", flush=True)

    summary: dict[str, Any] = {
        "study": "AV-ASD audio-only PPO adaptive segment sampling",
        "status": "preliminary execution completed; independent replication and review are still required",
        "data": data_info,
        "evaluation": "5-fold video-grouped cross-validation; each video contributes one majority-label sample",
        "participant_independence": "not established; participant identity is not verified in the local bundle",
        "modality_limit": "Only 32-dimensional audio descriptors are available locally. This experiment is audio-only and is not an audio-visual or multimodal result.",
        "sampling_budget": SAMPLE_BUDGET,
        "reward": "training-video segment-descriptor novelty minus small selection cost; not a clinical outcome",
        "fold_results": fold_rows,
        "aggregate": {},
    }
    for method in METHODS:
        rows = [r for r in fold_rows if r["method"] == method]
        summary["aggregate"][method] = {
            "accuracy_mean": float(np.mean([r["accuracy"] for r in rows])),
            "accuracy_sd": float(np.std([r["accuracy"] for r in rows], ddof=1)),
            "macro_f1_mean": float(np.mean([r["macro_f1"] for r in rows])),
            "macro_f1_sd": float(np.std([r["macro_f1"] for r in rows], ddof=1)),
        }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fold_rows[0]))
        writer.writeheader()
        writer.writerows(fold_rows)
    print(f"Saved {OUT_JSON}")
    print(f"Saved {OUT_CSV}")
    for method, values in summary["aggregate"].items():
        print(f"{method}: accuracy={values['accuracy_mean']:.4f}±{values['accuracy_sd']:.4f}; "
              f"macro-F1={values['macro_f1_mean']:.4f}±{values['macro_f1_sd']:.4f}")


if __name__ == "__main__":
    main()
