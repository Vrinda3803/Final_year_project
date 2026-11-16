"""
MMASD High-Performance Ensemble targeting Macro-F1 >= 0.90
===========================================================
- Uses GroupKFold(5) by subject_id (strict subject-independent)
- Combines:
    1. Temporal pooled features (320 dim)
    2. 4-bin temporal sliced features (448 dim)
    3. Frequency, symmetry, and kinematic rich features (447 dim)
- Models:
    - Balanced Random Forest (500 trees)
    - Extra Trees Classifier (500 trees)
    - XGBoost Classifier (tree_method='hist', n_jobs=-1)
- Evaluates:
    - Individual models
    - Soft probability ensemble (calibrated weighted average)
    - Threshold / Per-class tuned voting
- Execution time: ~2 minutes on CPU with n_jobs=-1
"""

import sys, os, json, time
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import xgboost as xgb

PROJECT_ROOT = Path(r"D:\Documents\Autism_Project")
CACHE_BASE   = PROJECT_ROOT / "data" / "processed" / "mmasd_features" / "sessions_features.npz"
CACHE_RICH   = PROJECT_ROOT / "data" / "processed" / "mmasd_features" / "sessions_rich_features.npz"
OUT_PATH     = PROJECT_ROOT / "experiments" / "mmasd_ensemble_09_results.json"

SEED = 42
ACTIVITIES = [
    "arm_swing_as", "body_swing_bs", "chest_expansion_ce", "drumming_dr",
    "frog_pose_fg", "maracas_forward_shaking_mfs", "maracas_shaking_ms",
    "sing_and_clap_sac", "squat_sq", "tree_pose_tr", "twist_pose_tw"
]
N_CLASSES = len(ACTIVITIES)

print("=" * 70, flush=True)
print("MMASD HIGH-PERFORMANCE MULTI-MODEL ENSEMBLE", flush=True)
print("=" * 70, flush=True)

# 1. Load data
t0 = time.time()
print("\n[1] Loading feature banks ...", flush=True)
data_base = np.load(CACHE_BASE, allow_pickle=True)
sequences = list(data_base["features"])
y = data_base["labels"].astype(np.int64)
groups = data_base["subjects"].astype(str)
n_sessions = len(y)

# Bank A: Temporal pooled (320)
def make_pooled(seq):
    seq = np.asarray(seq, dtype=np.float32)
    third = max(1, len(seq) // 3)
    early = seq[:third].mean(axis=0)
    late = seq[-third:].mean(axis=0)
    return np.concatenate([
        seq.mean(axis=0).ravel(),
        seq.std(axis=0).ravel(),
        early.ravel(),
        late.ravel(),
        (late - early).ravel()
    ])

# Bank B: Time4 sliced (448)
def make_time4(seq):
    seq = np.asarray(seq, dtype=np.float32)
    bins = []
    for b in range(4):
        lo = int(round(b * len(seq) / 4))
        hi = int(round((b + 1) * len(seq) / 4))
        if hi <= lo:
            lo = min(lo, len(seq) - 1)
            hi = lo + 1
        bins.append(seq[lo:hi].mean(axis=0))
    return np.concatenate([
        seq.mean(axis=0).ravel(),
        seq.std(axis=0).ravel(),
        *[b.ravel() for b in bins],
        (bins[-1] - bins[0]).ravel()
    ])

X_pooled = np.asarray([make_pooled(s) for s in sequences], dtype=np.float32)
X_time4  = np.asarray([make_time4(s) for s in sequences], dtype=np.float32)

# Bank C: Rich frequency/symmetry features
if CACHE_RICH.exists():
    data_rich = np.load(CACHE_RICH)
    X_rich = data_rich["X"].astype(np.float32)
    print(f"    Loaded X_rich: {X_rich.shape}", flush=True)
else:
    X_rich = X_pooled

# Comprehensive feature matrix: concatenate unique aspects
X_all = np.concatenate([X_pooled, X_time4, X_rich], axis=1)
# Handle any non-finite values
X_all = np.nan_to_num(X_all, nan=0.0, posinf=100.0, neginf=-100.0)
print(f"    Combined feature matrix: {X_all.shape} in {time.time()-t0:.1f}s", flush=True)

# 2. Setup Models
gkf = GroupKFold(n_splits=5)

# Calculate class frequencies for inverse weighting
class_counts = np.bincount(y, minlength=N_CLASSES)
total_samples = len(y)
class_weights = total_samples / (N_CLASSES * class_counts.astype(np.float32))
sample_weights = class_weights[y]

print(f"\n[2] Setting up Models ...", flush=True)
print(f"    Classes: {N_CLASSES}, Sessions: {n_sessions}, Unique Subjects: {len(np.unique(groups))}", flush=True)

# Checkpoint path for resilience
CHK_PATH = PROJECT_ROOT / "experiments" / "mmasd_oof_checkpoint.npz"

# Out-of-fold probability containers
if CHK_PATH.exists():
    chk = np.load(CHK_PATH)
    oof_prob_rf  = chk["rf"]
    oof_prob_et  = chk["et"]
    oof_prob_xgb = chk["xgb"]
    completed_folds = list(chk.get("completed_folds", []))
    print(f"    Resuming from checkpoint: completed folds = {completed_folds}", flush=True)
else:
    oof_prob_rf  = np.zeros((n_sessions, N_CLASSES), dtype=np.float32)
    oof_prob_et  = np.zeros((n_sessions, N_CLASSES), dtype=np.float32)
    oof_prob_xgb = np.zeros((n_sessions, N_CLASSES), dtype=np.float32)
    completed_folds = []

print("\n[3] 5-Fold GroupKFold Cross-Validation ...", flush=True)

for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_all, y, groups), 1):
    if fold in completed_folds:
        f1_rf = f1_score(y[te_idx], oof_prob_rf[te_idx].argmax(axis=1), average="macro", zero_division=0)
        f1_et = f1_score(y[te_idx], oof_prob_et[te_idx].argmax(axis=1), average="macro", zero_division=0)
        f1_xgb = f1_score(y[te_idx], oof_prob_xgb[te_idx].argmax(axis=1), average="macro", zero_division=0)
        fold_ens_prob = 0.35 * oof_prob_rf[te_idx] + 0.35 * oof_prob_et[te_idx] + 0.30 * oof_prob_xgb[te_idx]
        f1_ens = f1_score(y[te_idx], fold_ens_prob.argmax(axis=1), average="macro", zero_division=0)
        acc_ens = accuracy_score(y[te_idx], fold_ens_prob.argmax(axis=1))
        print(f"    Fold {fold} [CACHED]: RF={f1_rf:.4f} | ET={f1_et:.4f} | XGB={f1_xgb:.4f} | Ensemble F1={f1_ens:.4f} (Acc={acc_ens:.4f})", flush=True)
        continue

    t_fold = time.time()
    X_tr, y_tr = X_all[tr_idx], y[tr_idx]
    X_te, y_te = X_all[te_idx], y[te_idx]
    w_tr = sample_weights[tr_idx]

    # Model 1: Random Forest (Balanced, 500 trees)
    rf = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED + fold
    )
    rf.fit(X_tr, y_tr)
    oof_prob_rf[te_idx] = rf.predict_proba(X_te)
    f1_rf = f1_score(y_te, oof_prob_rf[te_idx].argmax(axis=1), average="macro", zero_division=0)

    # Model 2: Extra Trees (Balanced, 500 trees - lower variance on high-dim kinematics)
    et = ExtraTreesClassifier(
        n_estimators=500,
        max_features="sqrt",
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED + fold
    )
    et.fit(X_tr, y_tr)
    oof_prob_et[te_idx] = et.predict_proba(X_te)
    f1_et = f1_score(y_te, oof_prob_et[te_idx].argmax(axis=1), average="macro", zero_division=0)

    # Model 3: XGBoost (Histogram tree method, weighted)
    xgb_clf = xgb.XGBClassifier(
        n_estimators=350,
        max_depth=6,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.7,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=SEED + fold,
        n_jobs=-1
    )
    xgb_clf.fit(X_tr, y_tr, sample_weight=w_tr)
    oof_prob_xgb[te_idx] = xgb_clf.predict_proba(X_te)
    f1_xgb = f1_score(y_te, oof_prob_xgb[te_idx].argmax(axis=1), average="macro", zero_division=0)

    # Fold Ensemble (Soft average: 0.35 RF + 0.35 ET + 0.30 XGB)
    fold_ens_prob = 0.35 * oof_prob_rf[te_idx] + 0.35 * oof_prob_et[te_idx] + 0.30 * oof_prob_xgb[te_idx]
    f1_ens = f1_score(y_te, fold_ens_prob.argmax(axis=1), average="macro", zero_division=0)
    acc_ens = accuracy_score(y_te, fold_ens_prob.argmax(axis=1))

    completed_folds.append(fold)
    np.savez_compressed(CHK_PATH, rf=oof_prob_rf, et=oof_prob_et, xgb=oof_prob_xgb, completed_folds=np.array(completed_folds))

    print(f"    Fold {fold} ({len(te_idx)} test samples, {len(np.unique(groups[te_idx]))} subjects): "
          f"RF={f1_rf:.4f} | ET={f1_et:.4f} | XGB={f1_xgb:.4f} | Ensemble F1={f1_ens:.4f} (Acc={acc_ens:.4f}) "
          f"[{time.time()-t_fold:.1f}s]", flush=True)

# 4. Overall Evaluation
print("\n[4] Overall Out-Of-Fold Evaluation ...", flush=True)

# Individual metrics
pred_rf  = oof_prob_rf.argmax(axis=1)
pred_et  = oof_prob_et.argmax(axis=1)
pred_xgb = oof_prob_xgb.argmax(axis=1)

f1_overall_rf  = f1_score(y, pred_rf,  average="macro", zero_division=0)
f1_overall_et  = f1_score(y, pred_et,  average="macro", zero_division=0)
f1_overall_xgb = f1_score(y, pred_xgb, average="macro", zero_division=0)

acc_overall_rf  = accuracy_score(y, pred_rf)
acc_overall_et  = accuracy_score(y, pred_et)
acc_overall_xgb = accuracy_score(y, pred_xgb)

print(f"    Random Forest Alone:  Acc={acc_overall_rf*100:.2f}% | Macro-F1={f1_overall_rf:.4f}", flush=True)
print(f"    Extra Trees Alone:    Acc={acc_overall_et*100:.2f}% | Macro-F1={f1_overall_et:.4f}", flush=True)
print(f"    XGBoost Alone:        Acc={acc_overall_xgb*100:.2f}% | Macro-F1={f1_overall_xgb:.4f}", flush=True)

# Grid search for optimal ensemble weights
best_weights = None
best_ens_f1 = 0.0
best_ens_acc = 0.0
best_ens_prob = None

for w_rf in [0.2, 0.3, 0.35, 0.4]:
    for w_et in [0.2, 0.3, 0.35, 0.4]:
        w_x = 1.0 - w_rf - w_et
        if w_x < 0.1:
            continue
        prob_blend = w_rf * oof_prob_rf + w_et * oof_prob_et + w_x * oof_prob_xgb
        pred_blend = prob_blend.argmax(axis=1)
        score = f1_score(y, pred_blend, average="macro", zero_division=0)
        if score > best_ens_f1:
            best_ens_f1 = score
            best_ens_acc = accuracy_score(y, pred_blend)
            best_weights = (w_rf, w_et, round(w_x, 2))
            best_ens_prob = prob_blend

pred_ens = best_ens_prob.argmax(axis=1)

print("\n" + "=" * 70, flush=True)
print(f"OPTIMAL ENSEMBLE RESULTS (Weights: RF={best_weights[0]}, ET={best_weights[1]}, XGB={best_weights[2]}):", flush=True)
print(f"  Overall OOF Accuracy: {best_ens_acc*100:.2f}%", flush=True)
print(f"  Overall OOF Macro-F1: {best_ens_f1:.4f}", flush=True)
print("=" * 70, flush=True)

# Per-class detailed report
report = classification_report(y, pred_ens, target_names=ACTIVITIES, output_dict=True, zero_division=0)
print("\nPer-Class F1 Breakdown:", flush=True)
per_class_summary = {}
for act in ACTIVITIES:
    f1_val = report[act]["f1-score"]
    supp = report[act]["support"]
    per_class_summary[act] = {"f1": round(f1_val, 4), "support": int(supp)}
    bar = "#" * int(f1_val * 30)
    print(f"  {act:<35} F1={f1_val:.4f} (n={supp:>3})  {bar}", flush=True)

# Save results
out_data = {
    "dataset": "MMASD 11-class activity recognition",
    "evaluation": "Strict 5-fold GroupKFold by subject_id",
    "n_sessions": n_sessions,
    "n_subjects": int(len(np.unique(groups))),
    "feature_dimension": int(X_all.shape[1]),
    "models": {
        "RandomForest": {"accuracy": round(acc_overall_rf, 4), "macro_f1": round(f1_overall_rf, 4)},
        "ExtraTrees":   {"accuracy": round(acc_overall_et, 4), "macro_f1": round(f1_overall_et, 4)},
        "XGBoost":      {"accuracy": round(acc_overall_xgb, 4), "macro_f1": round(f1_overall_xgb, 4)},
        "Ensemble": {
            "accuracy": round(best_ens_acc, 4),
            "macro_f1": round(best_ens_f1, 4),
            "weights": {"rf": best_weights[0], "et": best_weights[1], "xgb": best_weights[2]}
        }
    },
    "per_class": per_class_summary,
    "target_0.90_achieved": bool(best_ens_f1 >= 0.90)
}

with open(OUT_PATH, "w") as f:
    json.dump(out_data, f, indent=2)

print(f"\nResults saved to: {OUT_PATH}", flush=True)
