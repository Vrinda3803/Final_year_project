"""
MMASD Super Pipeline — Meta-Stacking + LightGBM-style Gradient Boosting
=========================================================================
Target: 90%+ accuracy on MMASD 11-class

Strategy:
1. Same feature extraction as advanced pipeline (1,754 dims)
2. Per-fold: train RF + ET + XGB on training data
3. Get OOF probabilities from an INNER 3-fold CV on training data
4. Stack: train LogisticRegression meta-learner on OOF proba concatenated
5. Final prediction: (average base models) * 0.5 + (meta-learner) * 0.5
6. Alternative: Try higher n_estimators and deeper XGB

Key insight: The weak classes are squat_sq (F1=0.5529) and maracas_forward_shaking_mfs (F1=0.5978)
→ These share upper-body features, need better discrimination
→ SMOTE-like oversampling in feature space for rare classes

Date: 2026-11-12
"""
import sys, os, json, warnings, time
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_recall_fscore_support
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

print("="*70, flush=True)
print("MMASD SUPER PIPELINE — META-STACKING + BOOSTED ENSEMBLE", flush=True)
print("="*70, flush=True)

DATA_DIR  = r"D:\Documents\Autism_Project\data\processed\mmasd_features"
RES_DIR   = r"D:\Documents\Autism_Project\results"
EXP_DIR   = r"D:\Documents\Autism_Project\experiments"

# ─── Load data ─────────────────────────────────────────────────────────────────
print("\n[1/9] Loading data ...", flush=True)
base = np.load(os.path.join(DATA_DIR, "sessions_features.npz"), allow_pickle=True)
seqs    = base['features']
y       = base['labels'].astype(int)
groups  = base['subjects'].astype(str)

rich = np.load(os.path.join(DATA_DIR, "sessions_rich_features.npz"), allow_pickle=True)
X_rich = rich['X'].astype(np.float32)

N = len(y)
print(f"  Sessions={N}, Subjects={len(np.unique(groups))}, Classes={len(np.unique(y))}", flush=True)

# ─── Feature extraction (same multi-bank as advanced pipeline) ──────────────
print("\n[2/9] Extracting multi-bank features ...", flush=True)

def extract_all_features(seq):
    T = seq.shape[0]
    if T == 0:
        return np.zeros(1248, dtype=np.float32)
    flat = seq.reshape(T, -1).astype(np.float64)
    feats = []

    # Bank 1: Global statistics (6 stats x 64 features = 384)
    feats.append(np.concatenate([
        np.mean(flat, axis=0), np.std(flat, axis=0),
        np.min(flat, axis=0), np.max(flat, axis=0),
        np.max(flat, axis=0) - np.min(flat, axis=0),
        np.median(flat, axis=0),
    ]))

    # Bank 2: Velocity and acceleration (5 stats x 64 = 320)
    vel = np.diff(flat, axis=0) if T > 1 else np.zeros((1, flat.shape[1]))
    acc = np.diff(flat, n=2, axis=0) if T > 2 else np.zeros((1, flat.shape[1]))
    feats.append(np.concatenate([
        np.mean(np.abs(vel), axis=0), np.std(vel, axis=0), np.max(np.abs(vel), axis=0),
        np.mean(np.abs(acc), axis=0), np.max(np.abs(acc), axis=0),
    ]))

    # Bank 3: Fourier energy (256 dims)
    if T >= 4:
        fft_vals = np.abs(np.fft.rfft(flat, axis=0))
        fft_top = fft_vals[:min(4, fft_vals.shape[0]), :].flatten()
        bank3 = np.pad(fft_top, (0, max(0, 256 - len(fft_top))))[:256]
    else:
        bank3 = np.zeros(256)
    feats.append(bank3)

    # Bank 4: Symmetry + group means (96 dims)
    grp_mean = np.mean(seq, axis=0)  # (8, 8)
    sym_diff = grp_mean[2:4, :].flatten() - grp_mean[4:6, :].flatten()
    sym_ratio = np.where(np.abs(grp_mean[2:4, :].flatten()) > 1e-6,
                         grp_mean[4:6, :].flatten() / (np.abs(grp_mean[2:4, :].flatten()) + 1e-6),
                         np.zeros(16))
    feats.append(np.concatenate([sym_diff, sym_ratio, grp_mean.flatten()]))  # 16+16+64=96

    # Bank 5: Temporal segments early/mid/late (192 dims)
    third = max(1, T // 3)
    seg_feats = []
    for start in [0, third, 2*third]:
        end = min(start + third, T)
        seg = flat[start:end]
        if len(seg) > 0:
            seg_feats.extend([np.mean(seg, axis=0), np.std(seg, axis=0)])
        else:
            seg_feats.extend([np.zeros(64), np.zeros(64)])
    feats.append(np.concatenate(seg_feats)[:192])

    combined = np.concatenate(feats)
    return np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

t0 = time.time()
X_mb = np.array([extract_all_features(seq) for seq in seqs], dtype=np.float32)
print(f"  Multi-bank: {X_mb.shape}  ({time.time()-t0:.1f}s)", flush=True)

X_all = np.concatenate([X_mb, X_rich], axis=1)
X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
print(f"  Combined feature dim: {X_all.shape[1]}", flush=True)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_enc = le.fit_transform(y)
n_classes = len(le.classes_)

# ─── Prototype features helper ─────────────────────────────────────────────────
def compute_proto_features(X_tr, y_tr, X_te, n_cls):
    centroids = np.array([
        X_tr[y_tr == c].mean(axis=0) if np.sum(y_tr == c) > 0 else np.zeros(X_tr.shape[1])
        for c in range(n_cls)
    ])
    def dist(X, C):
        diffs = X[:, None, :] - C[None, :, :]
        return np.sqrt(np.sum(diffs**2, axis=-1)) / (X.shape[1]**0.5 + 1e-6)
    return dist(X_tr, centroids), dist(X_te, centroids)

# ─── Main evaluation with meta-stacking ───────────────────────────────────────
print("\n[3/9] Meta-Stacking Evaluation (5-fold outer GroupKFold) ...", flush=True)
gkf_outer = GroupKFold(n_splits=5)

fold_accs, fold_f1s = [], []
all_preds, all_true = [], []

for fold_idx, (tr_idx, te_idx) in enumerate(gkf_outer.split(X_all, y_enc, groups=groups)):
    print(f"\n  === FOLD {fold_idx+1}/5 ===", flush=True)
    t_fold = time.time()

    X_tr, X_te = X_all[tr_idx], X_all[te_idx]
    y_tr, y_te = y_enc[tr_idx], y_enc[te_idx]
    grp_tr = groups[tr_idx]

    # Scale
    scaler = RobustScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)

    # Feature selection (top-900 features by ANOVA F-test)
    sel = SelectKBest(f_classif, k=min(900, X_tr_sc.shape[1]))
    X_tr_sel = sel.fit_transform(X_tr_sc, y_tr)
    X_te_sel  = sel.transform(X_te_sc)

    # Prototype features
    proto_tr, proto_te = compute_proto_features(X_tr_sel, y_tr, X_te_sel, n_classes)
    X_tr_aug = np.hstack([X_tr_sel, proto_tr])
    X_te_aug  = np.hstack([X_te_sel, proto_te])

    print(f"    Feature dim after sel+proto: {X_tr_aug.shape[1]}", flush=True)

    # Class weights
    from sklearn.utils.class_weight import compute_class_weight
    cw_arr = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
    cw_dict = {int(c): float(w) for c, w in zip(np.unique(y_tr), cw_arr)}
    sw_tr = np.array([cw_dict[c] for c in y_tr])

    # ── INNER FOLD (for meta-learner OOF probabilities) ──────────────────────
    # Use 3-fold CV on training data to get OOF proba for stacking
    gkf_inner = GroupKFold(n_splits=3)
    oof_rf  = np.zeros((len(tr_idx), n_classes))
    oof_et  = np.zeros((len(tr_idx), n_classes))
    oof_xgb = np.zeros((len(tr_idx), n_classes))

    for in_fold, (in_tr, in_te) in enumerate(gkf_inner.split(X_tr_aug, y_tr, groups=grp_tr)):
        X_in_tr, X_in_te = X_tr_aug[in_tr], X_tr_aug[in_te]
        y_in_tr, y_in_te = y_tr[in_tr], y_tr[in_te]
        sw_in = np.array([cw_dict.get(int(c), 1.0) for c in y_in_tr])

        # RF inner
        rf_in = RandomForestClassifier(n_estimators=500, max_features='sqrt',
                                        class_weight='balanced', n_jobs=-1, random_state=42)
        rf_in.fit(X_in_tr, y_in_tr)
        oof_rf[in_te] = rf_in.predict_proba(X_in_te)

        # ET inner
        et_in = ExtraTreesClassifier(n_estimators=500, max_features='sqrt',
                                      class_weight='balanced', n_jobs=-1, random_state=42)
        et_in.fit(X_in_tr, y_in_tr)
        oof_et[in_te] = et_in.predict_proba(X_in_te)

        # XGB inner
        present_in = np.unique(y_in_tr)
        xmap_in  = {c: i for i, c in enumerate(present_in)}
        xrmap_in = {i: c for c, i in xmap_in.items()}
        y_in_tr_x = np.array([xmap_in[c] for c in y_in_tr])
        xgb_in = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08,
                                subsample=0.85, colsample_bytree=0.85,
                                min_child_weight=1, tree_method='hist',
                                verbosity=0, random_state=42, n_jobs=-1,
                                use_label_encoder=False)
        xgb_in.fit(X_in_tr, y_in_tr_x, sample_weight=sw_in)
        xgb_prob_in = xgb_in.predict_proba(X_in_te)
        xgb_full_in = np.zeros((len(in_te), n_classes))
        for li, gc in xrmap_in.items():
            if gc < n_classes:
                xgb_full_in[:, gc] = xgb_prob_in[:, li]
        oof_xgb[in_te] = xgb_full_in

    print(f"    Inner folds done", flush=True)

    # Meta-learner features (concatenate OOF probas)
    meta_tr_feats = np.hstack([oof_rf, oof_et, oof_xgb])  # (n_tr, 3*n_classes)

    # ── TRAIN FULL BASE MODELS on all training data ───────────────────────────
    # RF
    rf = RandomForestClassifier(n_estimators=1000, max_features='sqrt',
                                 max_depth=None, min_samples_leaf=1,
                                 class_weight='balanced', n_jobs=-1, random_state=42)
    rf.fit(X_tr_aug, y_tr)
    rf_prob_te = rf.predict_proba(X_te_aug)

    # ET
    et = ExtraTreesClassifier(n_estimators=1000, max_features='sqrt',
                               max_depth=None, min_samples_leaf=1,
                               class_weight='balanced', n_jobs=-1, random_state=42)
    et.fit(X_tr_aug, y_tr)
    et_prob_te = et.predict_proba(X_te_aug)

    # XGB (full)
    present_cls = np.unique(y_tr)
    xmap  = {c: i for i, c in enumerate(present_cls)}
    xrmap = {i: c for c, i in xmap.items()}
    y_tr_x = np.array([xmap[c] for c in y_tr])
    xgb = XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.04,
                         subsample=0.85, colsample_bytree=0.80,
                         min_child_weight=1, reg_alpha=0.1, reg_lambda=1.0,
                         tree_method='hist', eval_metric='mlogloss',
                         verbosity=0, random_state=42, n_jobs=-1,
                         use_label_encoder=False)
    xgb.fit(X_tr_aug, y_tr_x, sample_weight=sw_tr)
    xgb_prob_raw = xgb.predict_proba(X_te_aug)
    xgb_prob_te = np.zeros((len(te_idx), n_classes))
    for li, gc in xrmap.items():
        if gc < n_classes:
            xgb_prob_te[:, gc] = xgb_prob_raw[:, li]

    print(f"    Base models trained", flush=True)

    # ── META-LEARNER ─────────────────────────────────────────────────────────
    # Train LogisticRegression on OOF probas → predict on base model probas on test
    meta_lr = LogisticRegression(C=0.5, max_iter=2000, solver='lbfgs',
                                   multi_class='multinomial', random_state=42,
                                   class_weight='balanced')
    meta_lr.fit(meta_tr_feats, y_tr)

    # For test: use full model probas as meta-learner input
    meta_te_feats = np.hstack([rf_prob_te, et_prob_te, xgb_prob_te])
    meta_pred_proba = meta_lr.predict_proba(meta_te_feats)

    # ── COMBINED PREDICTION ───────────────────────────────────────────────────
    # Weighted combination: 50% base ensemble + 50% meta-learner
    base_ens = 0.40 * rf_prob_te + 0.35 * et_prob_te + 0.25 * xgb_prob_te
    final_prob = 0.5 * base_ens + 0.5 * meta_pred_proba
    final_pred = np.argmax(final_prob, axis=1)

    acc  = accuracy_score(y_te, final_pred)
    f1   = f1_score(y_te, final_pred, average='macro', zero_division=0)
    fold_accs.append(acc)
    fold_f1s.append(f1)
    all_preds.extend(final_pred.tolist())
    all_true.extend(y_te.tolist())

    # Also check base ensemble alone
    base_pred = np.argmax(base_ens, axis=1)
    base_acc  = accuracy_score(y_te, base_pred)
    meta_pred = meta_lr.predict(meta_te_feats)
    meta_acc  = accuracy_score(y_te, meta_pred)

    print(f"    Base ENS Acc={base_acc*100:.2f}%", flush=True)
    print(f"    Meta-LR Acc={meta_acc*100:.2f}%", flush=True)
    print(f"    COMBINED Acc={acc*100:.2f}%  F1={f1:.4f}  ({time.time()-t_fold:.1f}s)", flush=True)

# ─── Overall Results ───────────────────────────────────────────────────────────
print("\n[4/9] Overall Results ...", flush=True)
all_preds = np.array(all_preds)
all_true  = np.array(all_true)

overall_acc = accuracy_score(all_true, all_preds)
overall_f1  = f1_score(all_true, all_preds, average='macro', zero_division=0)

print(f"\n{'='*70}", flush=True)
print(f"OVERALL ACCURACY : {overall_acc*100:.2f}%", flush=True)
print(f"OVERALL MACRO-F1 : {overall_f1:.4f}", flush=True)
print(f"FOLD ACCS: {[f'{a*100:.2f}%' for a in fold_accs]}", flush=True)
print(f"FOLD F1s : {[f'{f:.4f}' for f in fold_f1s]}", flush=True)
print(f"{'='*70}", flush=True)

report = classification_report(all_true, all_preds, target_names=[str(c) for c in le.classes_], zero_division=0)
print("\nPer-class report:\n", report, flush=True)

# ─── Save Results ──────────────────────────────────────────────────────────────
print("\n[5/9] Saving results ...", flush=True)
prec, rec, fscore, supp = precision_recall_fscore_support(all_true, all_preds, zero_division=0)
per_class_dict = {}
for i, cls in enumerate(le.classes_):
    per_class_dict[str(cls)] = {
        'precision': float(prec[i]), 'recall': float(rec[i]),
        'f1': float(fscore[i]), 'support': int(supp[i])
    }

results = {
    "dataset": "MMASD 11-class — Super Pipeline (Meta-Stacking)",
    "pipeline": "MultiBank+Proto+SelectK900+RF1000+ET1000+XGB600+MetaLR",
    "date": "2026-11-12",
    "evaluation": "5-fold outer GroupKFold + 3-fold inner GroupKFold",
    "n_sessions": int(N),
    "n_subjects": int(len(np.unique(groups))),
    "n_classes": int(n_classes),
    "feature_dimension": int(X_all.shape[1]),
    "fold_results": [
        {"fold": i+1, "accuracy": float(a), "macro_f1": float(f)}
        for i, (a, f) in enumerate(zip(fold_accs, fold_f1s))
    ],
    "overall": {
        "accuracy": float(overall_acc),
        "macro_f1": float(overall_f1),
        "accuracy_pct": float(overall_acc * 100),
        "target_90_achieved": bool(overall_acc >= 0.90)
    },
    "per_class": per_class_dict,
    "disclaimer": "FOR CLINICAL DECISION-SUPPORT RESEARCH PURPOSES ONLY."
}

out_path = os.path.join(EXP_DIR, "mmasd_super_meta_results_2026_11_12.json")
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"  Saved: {out_path}", flush=True)

# Update benchmarks
bench_path = os.path.join(RES_DIR, "benchmarks_2026_11_12.json")
with open(bench_path, 'r') as f:
    benchmarks = json.load(f)

benchmarks['datasets']['MMASD']['models']['SuperMeta_RF1000_ET1000_XGB600_MetaLR'] = {
    'accuracy': float(overall_acc),
    'macro_f1': float(overall_f1),
    'pipeline': 'MetaStacking(RF1000+ET1000+XGB600+MetaLR)',
    'feature_dim': int(X_all.shape[1])
}
benchmarks['datasets']['MMASD']['target_0.90_achieved'] = bool(overall_acc >= 0.90)
benchmarks['last_updated'] = "2026-11-12"
with open(bench_path, 'w') as f:
    json.dump(benchmarks, f, indent=2)
print(f"  Updated benchmarks", flush=True)

print(f"\n{'='*70}", flush=True)
print(f"  MMASD Super Pipeline — 2026-11-12", flush=True)
print(f"  Accuracy : {overall_acc*100:.2f}%", flush=True)
print(f"  Macro-F1 : {overall_f1:.4f}", flush=True)
print(f"  Target 90% {'ACHIEVED ✓' if overall_acc >= 0.90 else f'NOT YET — gap: {(0.90-overall_acc)*100:.2f}%'}", flush=True)
print(f"{'='*70}", flush=True)
print("DISCLAIMER: FOR CLINICAL DECISION-SUPPORT RESEARCH PURPOSES ONLY.", flush=True)
