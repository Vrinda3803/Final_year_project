"""Nested GroupKFold comparison of pooled and time-binned MMASD RF features."""

import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold

ROOT = Path(r"D:\Documents\Autism_Project")
CACHE = ROOT / "data" / "processed" / "mmasd_features" / "sessions_features.npz"
OUT = ROOT / "experiments" / "mmasd_nested_rf_temporal_results.json"
SEED = 42


def make_features(sequence, mode):
    sequence = np.asarray(sequence, dtype=np.float32)
    if mode == "pooled":
        third = max(1, len(sequence) // 3)
        early = sequence[:third].mean(axis=0)
        late = sequence[-third:].mean(axis=0)
        parts = [sequence.mean(axis=0), sequence.std(axis=0), early, late, late - early]
    else:
        bins = []
        for b in range(4):
            lo = int(round(b * len(sequence) / 4))
            hi = int(round((b + 1) * len(sequence) / 4))
            if hi <= lo:
                lo = min(lo, len(sequence) - 1)
                hi = lo + 1
            bins.append(sequence[lo:hi].mean(axis=0))
        parts = [sequence.mean(axis=0), sequence.std(axis=0), *bins, bins[-1] - bins[0]]
    return np.concatenate([part.ravel() for part in parts])


def inner_score(x, y, groups, params):
    splitter = GroupKFold(n_splits=min(3, len(np.unique(groups))))
    scores = []
    for tr, va in splitter.split(x, y, groups):
        model = RandomForestClassifier(**params, n_jobs=2, random_state=SEED)
        model.fit(x[tr], y[tr])
        scores.append(f1_score(y[va], model.predict(x[va]), average="macro", zero_division=0))
    return float(np.mean(scores))


def main():
    started = time.time()
    data = np.load(CACHE, allow_pickle=True)
    sequences = list(data["features"])
    y = data["labels"].astype(np.int64)
    groups = data["subjects"].astype(str)
    banks = {
        mode: np.asarray([make_features(row, mode) for row in sequences], dtype=np.float32)
        for mode in ("pooled", "time4")
    }
    candidates = []
    for mode in banks:
        for params in (
            {"n_estimators": 500, "max_features": "sqrt", "max_depth": None,
             "min_samples_leaf": 1, "class_weight": "balanced"},
            {"n_estimators": 400, "max_features": 0.5, "max_depth": None,
             "min_samples_leaf": 1, "class_weight": "balanced"},
            {"n_estimators": 400, "max_features": "sqrt", "max_depth": 24,
             "min_samples_leaf": 1, "class_weight": "balanced"},
        ):
            candidates.append({"mode": mode, "params": params})

    oof = np.full(y.shape, -1, dtype=np.int64)
    folds = []
    outer = GroupKFold(n_splits=5)
    for fold, (tr, te) in enumerate(outer.split(banks["pooled"], y, groups), 1):
        choices = []
        for candidate in candidates:
            score = inner_score(
                banks[candidate["mode"]][tr], y[tr], groups[tr], candidate["params"]
            )
            choices.append({**candidate, "inner_macro_f1": score})
        selected = max(choices, key=lambda row: row["inner_macro_f1"])
        x = banks[selected["mode"]]
        model = RandomForestClassifier(**selected["params"], n_jobs=2, random_state=SEED)
        model.fit(x[tr], y[tr])
        pred = model.predict(x[te])
        oof[te] = pred
        f1 = f1_score(y[te], pred, average="macro", zero_division=0)
        acc = accuracy_score(y[te], pred)
        folds.append({
            "fold": fold, "test_macro_f1": float(f1), "test_accuracy": float(acc),
            "selected_feature_mode": selected["mode"],
            "selected_params": selected["params"],
            "inner_selection_macro_f1": selected["inner_macro_f1"],
            "n_test": int(len(te)),
            "test_subjects": int(len(np.unique(groups[te]))),
        })
        print(f"Outer fold {fold}/5: F1={f1:.4f}, accuracy={acc:.4f}, mode={selected['mode']}", flush=True)

    report = classification_report(y, oof, output_dict=True, zero_division=0)
    labels = sorted(np.unique(y).tolist())
    result = {
        "dataset": "MMASD 11-class activity recognition",
        "method": "Nested GroupKFold feature/RandomForest selection",
        "evaluation": "5 outer subject GroupKFold; 3 inner subject GroupKFold",
        "n_sessions": int(len(y)), "n_subjects": int(len(np.unique(groups))),
        "feature_dimensions": {key: int(value.shape[1]) for key, value in banks.items()},
        "seed": SEED,
        "outer_oof_macro_f1": float(f1_score(y, oof, average="macro", zero_division=0)),
        "outer_oof_accuracy": float(accuracy_score(y, oof)),
        "outer_fold_macro_f1_mean": float(np.mean([row["test_macro_f1"] for row in folds])),
        "outer_fold_macro_f1_std": float(np.std([row["test_macro_f1"] for row in folds])),
        "per_class": {
            str(label): {"f1": float(report[str(label)]["f1-score"]),
                         "support": int(report[str(label)]["support"])}
            for label in labels
        },
        "folds": folds,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Nested outer OOF: F1={result['outer_oof_macro_f1']:.4f}, accuracy={result['outer_oof_accuracy']:.4f}; saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
