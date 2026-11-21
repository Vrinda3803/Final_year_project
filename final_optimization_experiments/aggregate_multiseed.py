"""
Aggregate multi-seed results (seeds 42, 43, 44) for RC-BiGCN v2.
Run after run_multiseed_v2.py completes.
"""
import sys, os, json, csv
import numpy as np

sys.path.insert(0, r'D:\Documents\Autism_Project\.codex-deps')

from sklearn.metrics import accuracy_score, f1_score

OUT = r'D:\Documents\Autism_Project\final_optimization_experiments'
RI  = r'D:\Documents\Autism_Project\research_improvements'
DATA_PATH = r'D:\Documents\Autism_Project\data\processed\mmasd_openpose\mmasd_openpose_64frames.npz'

data = np.load(DATA_PATH, allow_pickle=True)
y_true = data['y'].astype(int)
classes = list(data['classes'])

SEEDS = [42, 43, 44]
seed_summaries = []
all_probs = []

print("Loading per-seed results...")
for seed in SEEDS:
    json_path = os.path.join(OUT, f'rc_bigcn_v2_seed{seed}_summary.json')
    npz_path  = os.path.join(OUT, f'rc_bigcn_v2_seed{seed}_oof.npz')
    if not os.path.exists(json_path):
        print(f"  Seed {seed}: MISSING (not yet computed)")
        continue
    j = json.load(open(json_path))
    d = np.load(npz_path)
    seed_summaries.append(j)
    all_probs.append(d['probs'])
    fold_accs = [fr['accuracy'] for fr in j['fold_results']]
    print(f"  Seed {seed}: pooled_acc={j['pooled_oof_acc']:.4f}  "
          f"mean_fold={j['mean_fold_acc']:.4f}±{j['std_fold_acc']:.4f}  "
          f"macro_f1={j['pooled_oof_macro_f1']:.4f}")
    print(f"    Folds: {[round(a,4) for a in fold_accs]}")

if len(seed_summaries) == 0:
    print("No seeds available yet. Run run_multiseed_v2.py first.")
    sys.exit(0)

# ── Per-seed metrics ───────────────────────────────────────────────────────
pooled_accs = [j['pooled_oof_acc'] for j in seed_summaries]
pooled_mf1s = [j['pooled_oof_macro_f1'] for j in seed_summaries]
mean_fold_accs = [j['mean_fold_acc'] for j in seed_summaries]
std_fold_accs = [j['std_fold_acc'] for j in seed_summaries]

print(f"\n{'='*55}")
print(f"MULTI-SEED SUMMARY ({len(seed_summaries)} seeds available)")
print(f"{'='*55}")
print(f"Pooled OOF Acc:    {np.mean(pooled_accs):.4f} ± {np.std(pooled_accs):.4f}")
print(f"Mean Fold Acc:     {np.mean(mean_fold_accs):.4f} ± {np.std(mean_fold_accs):.4f}")
print(f"Pooled Macro-F1:   {np.mean(pooled_mf1s):.4f} ± {np.std(pooled_mf1s):.4f}")

# ── Ensemble across seeds (mean probs) ────────────────────────────────────
if len(all_probs) > 1:
    ensemble_probs = np.mean(all_probs, axis=0)
    pred_ens = ensemble_probs.argmax(1)
    ens_acc = accuracy_score(y_true, pred_ens)
    ens_mf1 = f1_score(y_true, pred_ens, average='macro', zero_division=0)
    ens_wf1 = f1_score(y_true, pred_ens, average='weighted', zero_division=0)
    print(f"\nSeed-ensemble (mean probs):")
    print(f"  Pooled acc:    {ens_acc:.4f}")
    print(f"  Macro-F1:      {ens_mf1:.4f}")
    print(f"  Weighted-F1:   {ens_wf1:.4f}")
    np.savez(os.path.join(OUT, 'rc_bigcn_v2_multiseed_ensemble_oof.npz'),
             y_true=y_true, y_pred=pred_ens, probs=ensemble_probs)
    print("  Saved: rc_bigcn_v2_multiseed_ensemble_oof.npz")

# ── Per-fold breakdown across seeds ───────────────────────────────────────
print(f"\nPer-fold accuracy across seeds:")
for fold_i in range(5):
    fold_accs = [j['fold_results'][fold_i]['accuracy'] for j in seed_summaries
                 if fold_i < len(j['fold_results'])]
    print(f"  Fold {fold_i+1}: {[round(a,4) for a in fold_accs]}  mean={np.mean(fold_accs):.4f}±{np.std(fold_accs):.4f}")

# ── Save MULTISEED_RESULTS.csv ─────────────────────────────────────────────
csv_path = os.path.join(OUT, 'MULTISEED_RESULTS.csv')
rows = []
for j in seed_summaries:
    fold_accs_j = [fr['accuracy'] for fr in j['fold_results']]
    row = {'seed': j['seed'],
           'pooled_oof_acc': round(j['pooled_oof_acc'], 4),
           'mean_fold_acc': round(j['mean_fold_acc'], 4),
           'std_fold_acc': round(j['std_fold_acc'], 4),
           'pooled_oof_macro_f1': round(j['pooled_oof_macro_f1'], 4),
           'pooled_oof_weighted_f1': round(j['pooled_oof_weighted_f1'], 4),
           'pooled_oof_ece': round(j['pooled_oof_ece'], 4)}
    for i, acc in enumerate(fold_accs_j):
        row[f'fold{i+1}_acc'] = round(acc, 4)
    rows.append(row)

# Summary row
if len(seed_summaries) > 1:
    rows.append({
        'seed': 'MEAN',
        'pooled_oof_acc': round(np.mean(pooled_accs), 4),
        'mean_fold_acc': round(np.mean(mean_fold_accs), 4),
        'std_fold_acc': round(np.mean(std_fold_accs), 4),
        'pooled_oof_macro_f1': round(np.mean(pooled_mf1s), 4),
        'pooled_oof_weighted_f1': '',
        'pooled_oof_ece': '',
    })
    rows.append({
        'seed': 'STD',
        'pooled_oof_acc': round(np.std(pooled_accs), 4),
        'mean_fold_acc': round(np.std(mean_fold_accs), 4),
        'std_fold_acc': round(np.std(std_fold_accs), 4),
        'pooled_oof_macro_f1': round(np.std(pooled_mf1s), 4),
        'pooled_oof_weighted_f1': '',
        'pooled_oof_ece': '',
    })

with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction='ignore')
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"\nSaved: {csv_path}")
print("Done.")
