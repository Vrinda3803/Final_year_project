"""
MMASD HIGH-PERFORMANCE PIPELINE — Targeting Macro-F1 >= 0.90
=============================================================
Strategy:
  1. Rich feature engineering: joint angles, FFT, velocity profiles, ROM, static/dynamic ratio
  2. SVM (RBF), XGBoost, RF, GB — individually evaluated
  3. Voting Ensemble (SVM + RF + GB + XGB)
  4. MEAT-GNN with improved features
  5. Final Stacking Ensemble (RF meta-learner on top of base predictions)
  6. Strict GroupKFold(5) by subject_id throughout

MMASD activities are biomechanically VERY distinct:
  - tree_pose_tr:    static balance, minimal motion
  - squat_sq:        large vertical amplitude
  - arm_swing_as:    lateral oscillation
  - drumming_dr:     high-freq bilateral arm motion
  - twist_pose_tw:   rotational torso motion
  ... etc.

These properties make high F1 achievable with the right features.
"""

import sys, os, json, time
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = r"D:\Documents\Autism_Project"
FEAT_CACHE   = os.path.join(PROJECT_ROOT, "data", "processed", "mmasd_features", "sessions_features.npz")
RICH_FEAT_CACHE = os.path.join(PROJECT_ROOT, "data", "processed", "mmasd_features", "sessions_rich_features.npz")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "experiments", "mmasd_highperf_results.json")

ACTIVITIES = [
    "arm_swing_as", "body_swing_bs", "chest_expansion_ce", "drumming_dr",
    "frog_pose_fg", "maracas_forward_shaking_mfs", "maracas_shaking_ms",
    "sing_and_clap_sac", "squat_sq", "tree_pose_tr", "twist_pose_tw"
]
N_CLASSES = len(ACTIVITIES)

print("=" * 70)
print("MMASD HIGH-PERFORMANCE PIPELINE — Target Macro-F1 >= 0.90")
print("=" * 70)

# ============================================================
# Load base cached features
# ============================================================
print("\n[1] Loading base cached features ...")
cache = np.load(FEAT_CACHE, allow_pickle=True)
all_features = list(cache['features'])   # list of (T', 8, 8) arrays
all_labels   = cache['labels'].astype(np.int64)
all_subjects = cache['subjects']
N = len(all_features)
print(f"    Sessions: {N}, Subjects: {len(np.unique(all_subjects))}, Classes: {N_CLASSES}")

# ============================================================
# Rich Feature Engineering
# Body groups: 0=head, 1=torso, 2=left_arm, 3=right_arm,
#              4=left_leg, 5=right_leg, 6=left_hand, 7=right_hand
# Base features per group: [mean_x, mean_y, mean_z, vel_x, vel_y, vel_z, jerk_mag, spread]
# ============================================================
print("\n[2] Computing rich features ...")

def compute_rich_features(feat_arr):
    """
    feat_arr: (T', 8, 8) — T' temporal windows, 8 body groups, 8 base features
    Returns: 1D rich feature vector

    Feature categories:
    A) Statistical summaries of base features: mean, std, min, max, range -> 8*8*5 = 320
    B) Velocity magnitude per group (norm of vel_xyz): mean, std, max -> 8*3 = 24
    C) Total kinetic energy proxy (sum of vel_sq over time) -> 8
    D) Range of motion per group (max - min of centroid positions): -> 8*3 = 24
    E) Temporal autocorrelation of centroid z (vertical): lag-1 -> 8
    F) FFT dominant frequency of centroid y (lateral), z (vertical) -> 8*2 = 16
    G) Static/dynamic ratio: std of position / (mean of jerk + eps) -> 8
    H) Bilateral symmetry: |left - right| for arms, legs -> 4 group pairs * 8 = 32
    I) Head-torso coupling: correlation of head and torso positions -> 3
    J) Temporal derivative of spread (intra-group joint spread rate) -> 8

    Total: ~447 features
    """
    T, G, F = feat_arr.shape
    feats = []

    # Positions: indices 0,1,2 of base feat
    pos  = feat_arr[:, :, 0:3]   # (T, G, 3)
    vel  = feat_arr[:, :, 3:6]   # (T, G, 3)
    jerk = feat_arr[:, :, 6]     # (T, G)
    sprd = feat_arr[:, :, 7]     # (T, G)

    # A) Statistical summaries over all 8 base features
    for stat_fn in [np.mean, np.std, np.min, np.max]:
        feats.append(stat_fn(feat_arr, axis=0).flatten())   # (8,8) -> 64 each
    # Range
    feats.append((np.max(feat_arr, axis=0) - np.min(feat_arr, axis=0)).flatten())

    # B) Velocity magnitude per group: |vel|
    vel_mag = np.linalg.norm(vel, axis=-1)  # (T, G)
    feats.append(vel_mag.mean(axis=0))       # (G,)
    feats.append(vel_mag.std(axis=0))        # (G,)
    feats.append(vel_mag.max(axis=0))        # (G,)

    # C) Total kinetic energy proxy
    feats.append((vel_mag ** 2).sum(axis=0))  # (G,)

    # D) Range of motion (max-min over T of centroid position)
    rom = pos.max(axis=0) - pos.min(axis=0)  # (G, 3)
    feats.append(rom.flatten())               # 24

    # E) Temporal autocorrelation (lag-1) of vertical position (axis 2)
    pos_z = pos[:, :, 2]  # (T, G)
    if T > 2:
        autocorr = np.array([
            np.corrcoef(pos_z[:-1, g], pos_z[1:, g])[0, 1]
            if np.std(pos_z[:, g]) > 1e-9 else 0.0
            for g in range(G)
        ])
    else:
        autocorr = np.zeros(G)
    autocorr = np.nan_to_num(autocorr)
    feats.append(autocorr)  # (G,)

    # F) FFT dominant frequency band energy (low 0-25%, mid 25-50%, high 50-100%)
    for axis_idx in range(3):  # x, y, z
        pos_axis = pos[:, :, axis_idx]  # (T, G)
        if T >= 4:
            fft_mag = np.abs(np.fft.rfft(pos_axis, axis=0))  # (T//2+1, G)
            n_freq = fft_mag.shape[0]
            low  = fft_mag[:n_freq//3].mean(axis=0)
            mid  = fft_mag[n_freq//3:2*n_freq//3].mean(axis=0)
            high = fft_mag[2*n_freq//3:].mean(axis=0)
        else:
            low = mid = high = np.zeros(G)
        feats.append(low); feats.append(mid); feats.append(high)  # 3 * G each

    # G) Static/dynamic ratio: low jerk + low vel = static activity (tree pose)
    static_ratio = sprd.std(axis=0) / (jerk.mean(axis=0) + 1e-6)
    static_ratio = np.clip(static_ratio, 0, 100)
    feats.append(static_ratio)  # (G,)

    # H) Bilateral symmetry: left vs right arm, left vs right leg, left vs right hand
    # Group indices: left_arm=2, right_arm=3, left_leg=4, right_leg=5, left_hand=6, right_hand=7
    pairs = [(2, 3), (4, 5), (6, 7)]
    for (l, r) in pairs:
        sym_pos  = np.abs(pos[:, l, :] - pos[:, r, :]).mean(axis=0)   # (3,)
        sym_vel  = np.abs(vel_mag[:, l] - vel_mag[:, r]).mean(keepdims=True)  # (1,)
        sym_jerk = np.abs(jerk[:, l] - jerk[:, r]).mean(keepdims=True)
        feats.append(sym_pos); feats.append(sym_vel); feats.append(sym_jerk)

    # I) Head-torso coupling: Pearson correlation of centroid positions
    for axis_idx in range(3):
        h = pos[:, 0, axis_idx]; t = pos[:, 1, axis_idx]
        if np.std(h) > 1e-9 and np.std(t) > 1e-9:
            corr = np.corrcoef(h, t)[0, 1]
        else:
            corr = 0.0
        feats.append(np.array([np.nan_to_num(corr)]))  # scalar

    # J) Temporal derivative of spread
    if T > 1:
        sprd_rate = np.abs(np.diff(sprd, axis=0)).mean(axis=0)  # (G,)
    else:
        sprd_rate = np.zeros(G)
    feats.append(sprd_rate)

    # K) Dominant body part activity (which group has highest vel_mag)
    group_activity = vel_mag.mean(axis=0)  # (G,)
    feats.append(group_activity / (group_activity.sum() + 1e-9))  # normalized -> (G,)

    # L) Temporal entropy of velocity (measure of regularity)
    for g in range(G):
        vm = vel_mag[:, g]
        if vm.sum() > 1e-9:
            vm_norm = vm / vm.sum()
            vm_norm = np.clip(vm_norm, 1e-12, 1.0)
            entropy = -np.sum(vm_norm * np.log(vm_norm))
        else:
            entropy = 0.0
        feats.append(np.array([entropy]))

    return np.concatenate([f.flatten() for f in feats]).astype(np.float32)


# Check if rich features already cached
if os.path.exists(RICH_FEAT_CACHE):
    print("    Loading cached rich features ...")
    rich_cache = np.load(RICH_FEAT_CACHE)
    X_rich = rich_cache['X']
    print(f"    Loaded: {X_rich.shape}")
else:
    print("    Computing rich features (1693 sessions) ...")
    t0 = time.time()
    X_rich = []
    for i, feat in enumerate(all_features):
        fa = np.array(feat, dtype=np.float32)
        rv = compute_rich_features(fa)
        X_rich.append(rv)
        if (i+1) % 200 == 0:
            print(f"    {i+1}/{N} done [{time.time()-t0:.0f}s]")

    X_rich = np.array(X_rich, dtype=np.float32)
    # Handle any NaN/Inf
    X_rich = np.nan_to_num(X_rich, nan=0.0, posinf=100.0, neginf=-100.0)
    np.savez_compressed(RICH_FEAT_CACHE, X=X_rich)
    print(f"    Done: {X_rich.shape}, time={time.time()-t0:.1f}s")
    print(f"    Cached to: {RICH_FEAT_CACHE}")

print(f"    Rich feature vector dimension: {X_rich.shape[1]}")

y = all_labels
groups = all_subjects

# ============================================================
# Imports for models
# ============================================================
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

try:
    import xgboost as xgb
    HAS_XGB = True
    print("    XGBoost available.")
except ImportError:
    HAS_XGB = False
    print("    XGBoost not available — skipping XGB model.")

gkf = GroupKFold(n_splits=5)

# ============================================================
# Helper: Run model with GroupKFold, return mean results
# ============================================================
def run_model_gkf(clf_name, make_clf_fn, X, y, groups, verbose=True):
    fold_accs, fold_f1s = [], []
    per_class_f1_sum = np.zeros(N_CLASSES)
    fold_preds_all, fold_true_all = [], []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups)):
        clf = make_clf_fn()
        clf.fit(X[tr], y[tr])
        p = clf.predict(X[te])
        acc = accuracy_score(y[te], p)
        f1  = f1_score(y[te], p, average='macro', zero_division=0)
        fold_accs.append(acc); fold_f1s.append(f1)
        # Accumulate per-class F1
        rpt = classification_report(y[te], p, output_dict=True, zero_division=0)
        for c in range(N_CLASSES):
            if str(c) in rpt:
                per_class_f1_sum[c] += rpt[str(c)]['f1-score']
        fold_preds_all.extend(p.tolist())
        fold_true_all.extend(y[te].tolist())
        if verbose:
            print(f"    Fold {fold+1}: Acc={acc:.4f} Macro-F1={f1:.4f}")

    mean_acc = float(np.mean(fold_accs))
    mean_f1  = float(np.mean(fold_f1s))
    per_class_mean = (per_class_f1_sum / 5).tolist()

    # Overall report
    overall_f1 = f1_score(fold_true_all, fold_preds_all, average='macro', zero_division=0)
    overall_acc = accuracy_score(fold_true_all, fold_preds_all)

    print(f"    {clf_name}: Acc={mean_acc:.4f}  Macro-F1={mean_f1:.4f}  "
          f"[Overall pooled: Acc={overall_acc:.4f} F1={overall_f1:.4f}]")
    return {
        "acc_mean": round(mean_acc, 4), "f1_mean": round(mean_f1, 4),
        "acc_std":  round(float(np.std(fold_accs)), 4),
        "f1_std":   round(float(np.std(fold_f1s)), 4),
        "fold_accs": [round(a,4) for a in fold_accs],
        "fold_f1s":  [round(f,4) for f in fold_f1s],
        "per_class_f1": {ACTIVITIES[c]: round(per_class_mean[c], 4) for c in range(N_CLASSES)}
    }

# ============================================================
# Model 1: SVM with RBF kernel (great for small datasets)
# ============================================================
print("\n[3] SVM RBF (StandardScaler + C=10, gamma=scale) ...")
def make_svm():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svm',    SVC(C=10.0, kernel='rbf', gamma='scale', probability=False, random_state=42))
    ])
svm_results = run_model_gkf("SVM-RBF", make_svm, X_rich, y, groups)

# ============================================================
# Model 2: SVM with higher C
# ============================================================
print("\n[4] SVM RBF C=50 ...")
def make_svm2():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svm',    SVC(C=50.0, kernel='rbf', gamma='scale', probability=False, random_state=42))
    ])
svm2_results = run_model_gkf("SVM-RBF-C50", make_svm2, X_rich, y, groups)

# ============================================================
# Model 3: Random Forest (rich features)
# ============================================================
print("\n[5] Random Forest (500 trees, rich features) ...")
def make_rf():
    return RandomForestClassifier(n_estimators=500, max_depth=20,
                                   min_samples_leaf=1, n_jobs=-1, random_state=42)
rf_results = run_model_gkf("RF-500", make_rf, X_rich, y, groups)

# ============================================================
# Model 4: Gradient Boosting
# ============================================================
print("\n[6] Gradient Boosting (200 trees, depth=6) ...")
def make_gb():
    return GradientBoostingClassifier(n_estimators=200, max_depth=6,
                                       learning_rate=0.08, subsample=0.8,
                                       random_state=42)
gb_results = run_model_gkf("GB-200", make_gb, X_rich, y, groups)

# ============================================================
# Model 5: XGBoost (if available)
# ============================================================
xgb_results = None
if HAS_XGB:
    print("\n[7] XGBoost ...")
    def make_xgb():
        return xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.08,
                                  subsample=0.8, colsample_bytree=0.8,
                                  use_label_encoder=False, eval_metric='mlogloss',
                                  random_state=42, n_jobs=-1)
    xgb_results = run_model_gkf("XGBoost", make_xgb, X_rich, y, groups)

# ============================================================
# Model 6: Voting Ensemble (SVM prob + RF + GB)
# ============================================================
print("\n[8] Stacking / Voting Ensemble ...")

# Use out-of-fold predictions from RF + GB + SVM for stacking
from sklearn.model_selection import cross_val_predict

# Get individual model out-of-fold predictions for stacking
print("    Computing out-of-fold predictions for stacking meta-features ...")

def get_oof_preds(make_fn, X, y, groups, gkf):
    preds = np.zeros(len(y), dtype=np.int64)
    for tr, te in gkf.split(X, y, groups=groups):
        m = make_fn()
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    return preds

oof_rf  = get_oof_preds(make_rf,  X_rich, y, groups, gkf)
oof_gb  = get_oof_preds(make_gb,  X_rich, y, groups, gkf)
oof_svm = get_oof_preds(make_svm2, X_rich, y, groups, gkf)

# Majority voting across RF, GB, SVM
from scipy import stats

vote_preds = stats.mode(np.stack([oof_rf, oof_gb, oof_svm], axis=1), axis=1)[0].flatten()
vote_acc = accuracy_score(y, vote_preds)
vote_f1  = f1_score(y, vote_preds, average='macro', zero_division=0)
print(f"    Hard Voting (RF+GB+SVM): Acc={vote_acc:.4f}  Macro-F1={vote_f1:.4f}")

# Per-fold voting (proper evaluation)
print("    Per-fold voting ensemble ...")
vote_fold_accs, vote_fold_f1s = [], []
for fold, (tr, te) in enumerate(gkf.split(X_rich, y, groups=groups)):
    # Train all three on tr, predict on te, majority vote
    m_rf  = make_rf(); m_rf.fit(X_rich[tr], y[tr])
    m_gb  = make_gb(); m_gb.fit(X_rich[tr], y[tr])
    m_svm = make_svm2(); m_svm.fit(X_rich[tr], y[tr])
    p_rf  = m_rf.predict(X_rich[te])
    p_gb  = m_gb.predict(X_rich[te])
    p_svm = m_svm.predict(X_rich[te])
    vote  = stats.mode(np.stack([p_rf, p_gb, p_svm], axis=1), axis=1)[0].flatten()
    acc = accuracy_score(y[te], vote)
    f1  = f1_score(y[te], vote, average='macro', zero_division=0)
    vote_fold_accs.append(acc); vote_fold_f1s.append(f1)
    print(f"    Fold {fold+1}: Acc={acc:.4f} Macro-F1={f1:.4f}")

vote_acc_mean = float(np.mean(vote_fold_accs))
vote_f1_mean  = float(np.mean(vote_fold_f1s))
print(f"    VOTING ENSEMBLE: Acc={vote_acc_mean:.4f}  Macro-F1={vote_f1_mean:.4f}")

# ============================================================
# XGB-augmented ensemble (if available)
# ============================================================
xgb_vote_results = None
if HAS_XGB:
    print("    Per-fold voting with XGB ...")
    vote4_fold_accs, vote4_fold_f1s = [], []
    best_per_class_f1 = {}
    for fold, (tr, te) in enumerate(gkf.split(X_rich, y, groups=groups)):
        m_rf  = make_rf();  m_rf.fit(X_rich[tr], y[tr])
        m_gb  = make_gb();  m_gb.fit(X_rich[tr], y[tr])
        m_svm = make_svm2(); m_svm.fit(X_rich[tr], y[tr])
        m_xgb = make_xgb(); m_xgb.fit(X_rich[tr], y[tr])
        p_rf  = m_rf.predict(X_rich[te])
        p_gb  = m_gb.predict(X_rich[te])
        p_svm = m_svm.predict(X_rich[te])
        p_xgb = m_xgb.predict(X_rich[te])
        vote4 = stats.mode(np.stack([p_rf, p_gb, p_svm, p_xgb], axis=1), axis=1)[0].flatten()
        acc = accuracy_score(y[te], vote4)
        f1  = f1_score(y[te], vote4, average='macro', zero_division=0)
        vote4_fold_accs.append(acc); vote4_fold_f1s.append(f1)
        rpt = classification_report(y[te], vote4, output_dict=True, zero_division=0)
        for c in range(N_CLASSES):
            k = str(c)
            if k in rpt:
                best_per_class_f1[ACTIVITIES[c]] = best_per_class_f1.get(ACTIVITIES[c], 0) + rpt[k]['f1-score']
        print(f"    Fold {fold+1}: Acc={acc:.4f} Macro-F1={f1:.4f}")

    xgb_vote_acc = float(np.mean(vote4_fold_accs))
    xgb_vote_f1  = float(np.mean(vote4_fold_f1s))
    best_per_class_f1 = {k: round(v/5, 4) for k, v in best_per_class_f1.items()}
    xgb_vote_results = {
        "acc_mean": round(xgb_vote_acc, 4), "f1_mean": round(xgb_vote_f1, 4),
        "per_class_f1": best_per_class_f1
    }
    print(f"    ENSEMBLE (RF+GB+SVM+XGB): Acc={xgb_vote_acc:.4f}  Macro-F1={xgb_vote_f1:.4f}")

# ============================================================
# Save Results
# ============================================================
print("\n[9] Saving results ...")
results = {
    "dataset": "MMASD 11-class",
    "n_sessions": N,
    "n_subjects": int(len(np.unique(all_subjects))),
    "n_classes": N_CLASSES,
    "evaluation": "GroupKFold-5 by subject_id — STRICT SUBJECT-INDEPENDENT",
    "feature_dim": int(X_rich.shape[1]),
    "random_chance_f1": round(1.0/N_CLASSES, 4),
    "target_f1": 0.90,
    "baselines_prev_sessions": {
        "audio_GB_avasd": {"acc": 0.4725, "f1": 0.4290},
        "pose_RF_avasd":  {"acc": 0.4759, "f1": 0.4403},
        "temporal_gcn_mmasd": {"acc": 0.5433, "f1": 0.4279}
    },
    "models": {
        "SVM_RBF_C10":    svm_results,
        "SVM_RBF_C50":    svm2_results,
        "RF_500trees":    rf_results,
        "GB_200trees":    gb_results,
        "XGBoost":        xgb_results,
        "Voting_RF_GB_SVM": {
            "acc_mean": round(vote_acc_mean, 4),
            "f1_mean":  round(vote_f1_mean, 4),
            "fold_accs": [round(a,4) for a in vote_fold_accs],
            "fold_f1s":  [round(f,4) for f in vote_fold_f1s]
        },
        "Voting_RF_GB_SVM_XGB": xgb_vote_results
    }
}

# Find best model
all_model_f1s = {
    "SVM_RBF_C10": svm_results["f1_mean"],
    "SVM_RBF_C50": svm2_results["f1_mean"],
    "RF_500trees":  rf_results["f1_mean"],
    "GB_200trees":  gb_results["f1_mean"],
    "Voting_RF_GB_SVM": vote_f1_mean,
}
if xgb_results: all_model_f1s["XGBoost"] = xgb_results["f1_mean"]
if xgb_vote_results: all_model_f1s["Voting_RF_GB_SVM_XGB"] = xgb_vote_results["f1_mean"]

best_model = max(all_model_f1s, key=all_model_f1s.get)
best_f1    = all_model_f1s[best_model]
results["best_model"] = {"name": best_model, "f1_mean": round(best_f1, 4)}
results["target_achieved"] = best_f1 >= 0.90

with open(RESULTS_PATH, 'w') as f:
    json.dump(results, f, indent=2)
print(f"    Saved: {RESULTS_PATH}")

# ============================================================
# Final Summary
# ============================================================
print("\n" + "=" * 70)
print("FINAL RESULTS — MMASD 11-class | GroupKFold-5 | Subject-Independent")
print("=" * 70)
print(f"{'Method':<35} {'Acc':>8} {'Macro-F1':>10}")
print("-" * 55)
print(f"{'Random Chance':<35} {'--':>8} {1/N_CLASSES:>10.4f}")
print(f"{'Prev Session Temporal GCN':<35} {'54.33%':>8} {'0.4279':>10}")
print(f"{'SVM RBF (C=10)':<35} {svm_results['acc_mean']*100:>7.2f}% {svm_results['f1_mean']:>10.4f}")
print(f"{'SVM RBF (C=50)':<35} {svm2_results['acc_mean']*100:>7.2f}% {svm2_results['f1_mean']:>10.4f}")
print(f"{'RF 500 trees':<35} {rf_results['acc_mean']*100:>7.2f}% {rf_results['f1_mean']:>10.4f}")
print(f"{'GB 200 trees':<35} {gb_results['acc_mean']*100:>7.2f}% {gb_results['f1_mean']:>10.4f}")
if xgb_results:
    print(f"{'XGBoost':<35} {xgb_results['acc_mean']*100:>7.2f}% {xgb_results['f1_mean']:>10.4f}")
print(f"{'Voting (RF+GB+SVM)':<35} {vote_acc_mean*100:>7.2f}% {vote_f1_mean:>10.4f}")
if xgb_vote_results:
    print(f"{'Voting (RF+GB+SVM+XGB)':<35} {xgb_vote_results['acc_mean']*100:>7.2f}% {xgb_vote_results['f1_mean']:>10.4f}")
print("-" * 55)
print(f"\nBEST MODEL: {best_model}")
print(f"BEST Macro-F1: {best_f1:.4f}")
print(f"TARGET (0.90): {'ACHIEVED!' if best_f1 >= 0.90 else f'NOT YET — gap = {0.90-best_f1:.4f}'}")
print("=" * 70)

if xgb_vote_results and xgb_vote_results.get("per_class_f1"):
    print("\nPer-Class F1 (Best Ensemble):")
    for act, f1v in sorted(xgb_vote_results["per_class_f1"].items(), key=lambda x: -x[1]):
        bar = "#" * int(f1v * 30)
        print(f"  {act:<42} {f1v:.4f}  {bar}")
