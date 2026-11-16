"""Nested subject-independent MMASD baseline using cached temporal features.

Hyperparameters are chosen with GroupKFold inside each outer training fold.
Outer test groups are not used for selection. The output is deliberately kept
separate from the MEAT-GNN result file.
"""

import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold


ROOT = Path(r"D:\Documents\Autism_Project")
CACHE = ROOT / "data" / "processed" / "mmasd_features" / "sessions_features.npz"
OUT = ROOT / "experiments" / "mmasd_nested_rf_results.json"
SEED = 42
OUTER_SPLITS = 5
INNER_SPLITS = 3


def summarize(sequence):
    """Summarize per-session temporal body-group features (T, 8, 8)."""
    sequence = np.asarray(sequence, dtype=np.float32)
    third = max(1, len(sequence) // 3)
    early = sequence[:third].mean(axis=0)
    late = sequence[-third:].mean(axis=0)
    blocks = (
        sequence.mean(axis=0),
        sequence.std(axis=0),
        early,
        late,
        late - early,
    )
    return np.concatenate([block.ravel() for block in blocks])


def score_params(x, y, groups, params):
    splitter = GroupKFold(n_splits=min(INNER_SPLITS, len(np.unique(groups))))
    scores = []
    for train_idx, valid_idx in splitter.split(x, y, groups):
        model = RandomForestClassifier(
            **params, n_jobs=2, random_state=SEED
        )
        model.fit(x[train_idx], y[train_idx])
        prediction = model.predict(x[valid_idx])
        scores.append(
            f1_score(y[valid_idx], prediction, average="macro", zero_division=0)
        )
    return float(np.mean(scores)), [float(value) for value in scores]


def main():
    started = time.time()
    data = np.load(CACHE, allow_pickle=True)
    x = np.asarray([summarize(row) for row in data["features"]], dtype=np.float32)
    y = data["labels"].astype(np.int64)
    groups = data["subjects"].astype(str)
    labels = sorted(np.unique(y).tolist())

    # This compact search is informed by the exploratory screen, but every
    # choice is re-evaluated using only the outer training subjects.
    candidates = [
        {"n_estimators": 300, "max_features": "sqrt", "max_depth": None,
         "min_samples_leaf": 1, "class_weight": "balanced"},
        {"n_estimators": 500, "max_features": "sqrt", "max_depth": None,
         "min_samples_leaf": 1, "class_weight": "balanced"},
        {"n_estimators": 400, "max_features": 0.5, "max_depth": None,
         "min_samples_leaf": 1, "class_weight": "balanced"},
        {"n_estimators": 400, "max_features": "sqrt", "max_depth": 24,
         "min_samples_leaf": 1, "class_weight": "balanced"},
    ]

    outer = GroupKFold(n_splits=OUTER_SPLITS)
    oof = np.full(y.shape, -1, dtype=np.int64)
    folds = []
    for fold, (train_idx, test_idx) in enumerate(outer.split(x, y, groups), 1):
        train_groups = groups[train_idx]
        inner_scores = []
        for candidate in candidates:
            mean_score, by_fold = score_params(
                x[train_idx], y[train_idx], train_groups, candidate
            )
            inner_scores.append({
                "params": candidate,
                "mean_macro_f1": mean_score,
                "fold_macro_f1": by_fold,
            })
        selected = max(inner_scores, key=lambda row: row["mean_macro_f1"])
        model = RandomForestClassifier(
            **selected["params"], n_jobs=2, random_state=SEED
        )
        model.fit(x[train_idx], y[train_idx])
        prediction = model.predict(x[test_idx])
        oof[test_idx] = prediction
        fold_f1 = f1_score(y[test_idx], prediction, average="macro", zero_division=0)
        fold_acc = accuracy_score(y[test_idx], prediction)
        folds.append({
            "fold": fold,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "train_subjects": int(len(np.unique(train_groups))),
            "test_subjects": int(len(np.unique(groups[test_idx]))),
            "selected_params": selected["params"],
            "inner_selection_macro_f1": selected["mean_macro_f1"],
            "test_macro_f1": float(fold_f1),
            "test_accuracy": float(fold_acc),
        })
        print(
            f"Outer fold {fold}/{OUTER_SPLITS}: Macro-F1={fold_f1:.4f}, "
            f"accuracy={fold_acc:.4f}, selected={selected['params']}",
            flush=True,
        )

    if np.any(oof < 0):
        raise RuntimeError("Some samples did not receive an outer-fold prediction")

    report = classification_report(
        y, oof, labels=labels, output_dict=True, zero_division=0
    )
    result = {
        "dataset": "MMASD 11-class activity recognition",
        "method": "RandomForest on pooled temporal kinematic summaries",
        "evaluation": "Nested GroupKFold by subject; 5 outer folds, 3 inner folds",
        "n_sessions": int(len(y)),
        "n_subjects": int(len(np.unique(groups))),
        "n_classes": int(len(labels)),
        "feature_shape": list(x.shape),
        "seed": SEED,
        "outer_oof_accuracy": float(accuracy_score(y, oof)),
        "outer_oof_macro_f1": float(f1_score(y, oof, average="macro", zero_division=0)),
        "outer_fold_macro_f1_mean": float(np.mean([row["test_macro_f1"] for row in folds])),
        "outer_fold_macro_f1_std": float(np.std([row["test_macro_f1"] for row in folds])),
        "outer_fold_accuracy_mean": float(np.mean([row["test_accuracy"] for row in folds])),
        "per_class": {
            str(label): {
                "precision": float(report[str(label)]["precision"]),
                "recall": float(report[str(label)]["recall"]),
                "f1": float(report[str(label)]["f1-score"]),
                "support": int(report[str(label)]["support"]),
            }
            for label in labels
        },
        "folds": folds,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"Nested outer OOF: Macro-F1={result['outer_oof_macro_f1']:.4f}, "
        f"accuracy={result['outer_oof_accuracy']:.4f}; saved {OUT}",
        flush=True,
    )


if __name__ == "__main__":
    main()
