"""
Audit and Reconciliation Script for Stage 3 Part 1 & Part 2.
Audits all historical and Stage 1/Stage 2 OOF files, verifies sample counts,
recomputes fold-wise and pooled metrics, and reconciles all discrepancies.
"""
from __future__ import annotations
import sys, json, csv
from pathlib import Path
ROOT = Path("d:/Documents/Autism_Project")
sys.path.insert(0, str(ROOT / ".codex-deps"))
sys.path.insert(0, str(ROOT))

import numpy as np

CLASSES = ["as", "bs", "ce", "dr", "mfs", "ms", "sac", "sq", "tr", "tw"]

def audit_oof_file(file_path: Path, name: str):
    if not file_path.exists():
        return {"error": f"File not found: {file_path}"}
        
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            clip = r.get("clip") or r.get("clip_id") or r.get("clip_name") or ""
            grp = r.get("group") or r.get("subject_id") or r.get("subject") or ""
            fold = int(r["fold"])
            if "y_true" in r:
                y_t = int(r["y_true"])
            elif "true_label" in r:
                y_t = int(r["true_label"])
            elif "label" in r:
                y_t = int(r["label"])
            else:
                y_t = -1
                
            if "y_pred" in r:
                y_p = int(r["y_pred"])
            elif "predicted_label" in r:
                y_p = int(r["predicted_label"])
            elif "prediction" in r:
                y_p = int(r["prediction"])
            elif "pred" in r:
                y_p = int(r["pred"])
            else:
                y_p = -1
                
            conf = float(r.get("confidence") or r.get("confidence_cal") or r.get("conf") or 0.0)
            rows.append({
                "clip": clip,
                "group": grp,
                "fold": fold,
                "y_true": y_t,
                "y_pred": y_p,
                "confidence": conf
            })
            
    total_clips = len(rows)
    unique_clips = len(set(r["clip"] for r in rows))
    unique_groups = len(set(r["group"] for r in rows))
    
    # Fold-wise metrics
    folds = sorted(list(set(r["fold"] for r in rows)))
    fold_accs = []
    fold_macro_f1s = []
    fold_counts = {}
    
    for f in folds:
        f_rows = [r for r in rows if r["fold"] == f]
        fold_counts[f] = len(f_rows)
        y_true_f = np.array([r["y_true"] for r in f_rows])
        y_pred_f = np.array([r["y_pred"] for r in f_rows])
        
        acc_f = float(np.mean(y_true_f == y_pred_f))
        
        # Per-class F1 for fold
        f1_list_f = []
        for c in range(len(CLASSES)):
            tp = np.sum((y_true_f == c) & (y_pred_f == c))
            fp = np.sum((y_true_f != c) & (y_pred_f == c))
            fn = np.sum((y_true_f == c) & (y_pred_f != c))
            denom = 2 * tp + fp + fn
            f1 = float(2 * tp / denom) if denom > 0 else 0.0
            f1_list_f.append(f1)
        macro_f1_f = float(np.mean(f1_list_f))
        
        fold_accs.append(acc_f)
        fold_macro_f1s.append(macro_f1_f)
        
    mean_fold_acc = float(np.mean(fold_accs))
    std_fold_acc = float(np.std(fold_accs))
    mean_fold_macro_f1 = float(np.mean(fold_macro_f1s))
    std_fold_macro_f1 = float(np.std(fold_macro_f1s))
    
    # Pooled metrics
    all_y_true = np.array([r["y_true"] for r in rows])
    all_y_pred = np.array([r["y_pred"] for r in rows])
    pooled_acc = float(np.mean(all_y_true == all_y_pred))
    
    pooled_f1_list = []
    per_class_f1 = {}
    for c, c_name in enumerate(CLASSES):
        tp = np.sum((all_y_true == c) & (all_y_pred == c))
        fp = np.sum((all_y_true != c) & (all_y_pred == c))
        fn = np.sum((all_y_true == c) & (all_y_pred != c))
        denom = 2 * tp + fp + fn
        f1 = float(2 * tp / denom) if denom > 0 else 0.0
        pooled_f1_list.append(f1)
        per_class_f1[c_name] = f1
    pooled_macro_f1 = float(np.mean(pooled_f1_list))
    
    # Confusion matrix
    conf_mat = np.zeros((10, 10), dtype=int)
    for yt, yp in zip(all_y_true, all_y_pred):
        if 0 <= yt < 10 and 0 <= yp < 10:
            conf_mat[yt, yp] += 1
            
    # Targeted error pairs:
    # 3: dr, 6: sac
    # 5: ms, 4: mfs
    dr_total = int(np.sum(all_y_true == 3))
    dr_as_sac = int(conf_mat[3, 6])
    sac_total = int(np.sum(all_y_true == 6))
    sac_as_dr = int(conf_mat[6, 3])
    
    ms_total = int(np.sum(all_y_true == 5))
    ms_as_mfs = int(conf_mat[5, 4])
    mfs_total = int(np.sum(all_y_true == 4))
    mfs_as_ms = int(conf_mat[4, 5])
    
    return {
        "name": name,
        "file": str(file_path),
        "total_clips": total_clips,
        "unique_clips": unique_clips,
        "unique_groups": unique_groups,
        "fold_counts": fold_counts,
        "fold_accs": fold_accs,
        "fold_macro_f1s": fold_macro_f1s,
        "mean_fold_acc": mean_fold_acc,
        "std_fold_acc": std_fold_acc,
        "mean_fold_macro_f1": mean_fold_macro_f1,
        "std_fold_macro_f1": std_fold_macro_f1,
        "pooled_acc": pooled_acc,
        "pooled_macro_f1": pooled_macro_f1,
        "per_class_f1": per_class_f1,
        "confusion_matrix": conf_mat.tolist(),
        "dr_total": dr_total,
        "dr_as_sac": dr_as_sac,
        "dr_as_sac_pct": float(dr_as_sac / dr_total * 100) if dr_total > 0 else 0.0,
        "sac_total": sac_total,
        "sac_as_dr": sac_as_dr,
        "ms_total": ms_total,
        "ms_as_mfs": ms_as_mfs,
        "ms_as_mfs_pct": float(ms_as_mfs / ms_total * 100) if ms_total > 0 else 0.0,
        "mfs_total": mfs_total,
        "mfs_as_ms": mfs_as_ms
    }

def main():
    files_to_audit = [
        (ROOT / "results" / "mmasd_openpose_rbf_kernel_oof_2026_11_12.csv", "RBF_Kernel_Baseline"),
        (ROOT / "novel_contribution" / "results" / "metrics" / "ra_stgt_baseline_stgt_oof.csv", "E0_Baseline_STGT"),
        (ROOT / "novel_contribution" / "results" / "metrics" / "ra_stgt_full_oof.csv", "E1_RA_STGT_v1"),
        (ROOT / "research_stage2" / "experiments" / "e2_topology_only_oof.csv", "E2_Topology_Only"),
        (ROOT / "research_stage2" / "experiments" / "e3_phase_only_oof.csv", "E3_Phase_Only"),
        (ROOT / "research_stage2" / "experiments" / "e4_margin_only_oof.csv", "E4_Margin_Only"),
        (ROOT / "research_stage2" / "experiments" / "e5_full_pr_stgt_oof.csv", "E5_Full_PR_STGT")
    ]
    
    results = {}
    for p, name in files_to_audit:
        print(f"Auditing: {name} ({p})")
        res = audit_oof_file(p, name)
        results[name] = res
        if "error" not in res:
            print(f"  Clips: {res['total_clips']} (Unique: {res['unique_clips']}) | Groups: {res['unique_groups']}")
            print(f"  Mean Fold Acc: {res['mean_fold_acc']*100:.2f}% ± {res['std_fold_acc']*100:.2f}% | Pooled Acc: {res['pooled_acc']*100:.2f}%")
            print(f"  Mean Fold F1:  {res['mean_fold_macro_f1']:.4f} ± {res['std_fold_macro_f1']:.4f} | Pooled F1:  {res['pooled_macro_f1']:.4f}")
            print(f"  dr->sac: {res['dr_as_sac']}/{res['dr_total']} ({res['dr_as_sac_pct']:.1f}%) | ms->mfs: {res['ms_as_mfs']}/{res['ms_total']} ({res['ms_as_mfs_pct']:.1f}%)")
        else:
            print(f"  ERROR: {res['error']}")
        print("-" * 60)
        
    out_json = ROOT / "research_stage3" / "reconciled_audit_metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved reconciled audit metrics to: {out_json}")

if __name__ == "__main__":
    main()
