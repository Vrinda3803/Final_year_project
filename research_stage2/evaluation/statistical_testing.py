"""
Statistical Significance and Diagnostic Error Testing for Stage 2 (PR-STGT vs RA-STGT v1).
Computes paired t-test, Wilcoxon signed-rank test, Cohen's d, and confusable-pair resolution.
"""
from __future__ import annotations
import sys, json, csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".codex-deps"))
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy import stats

CLASSES = ["as", "bs", "ce", "dr", "mfs", "ms", "sac", "sq", "tr", "tw"]


def load_oof_csv(path: Path | str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            clip = r.get("clip") or r.get("clip_id") or ""
            grp = r.get("group") or r.get("subject_id") or ""
            fold = int(r["fold"])
            y_t = int(r["y_true"]) if "y_true" in r else int(r["true_label"])
            y_p = int(r["y_pred"]) if "y_pred" in r else int(r["predicted_label"])
            conf = float(r.get("confidence") or r.get("confidence_cal") or 0.0)
            rows.append({
                "clip": clip,
                "group": grp,
                "fold": fold,
                "y_true": y_t,
                "y_pred": y_p,
                "confidence": conf
            })
    return rows


def compute_fold_metrics(rows: list[dict]) -> dict:
    folds = sorted(list(set(r["fold"] for r in rows)))
    fold_accs = []
    fold_f1s = []
    
    for f in folds:
        f_rows = [r for r in rows if r["fold"] == f]
        y_true = np.array([r["y_true"] for r in f_rows])
        y_pred = np.array([r["y_pred"] for r in f_rows])
        
        acc = float(np.mean(y_true == y_pred))
        f1_list = []
        for c in range(len(CLASSES)):
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))
            denom = 2 * tp + fp + fn
            f1 = float(2 * tp / denom) if denom > 0 else 0.0
            f1_list.append(f1)
        macro_f1 = float(np.mean(f1_list))
        
        fold_accs.append(acc)
        fold_f1s.append(macro_f1)
        
    return {
        "folds": folds,
        "accs": np.array(fold_accs),
        "f1s": np.array(fold_f1s)
    }


def compute_confusion_stats(rows: list[dict]) -> dict:
    y_true = np.array([r["y_true"] for r in rows])
    y_pred = np.array([r["y_pred"] for r in rows])
    
    # Targeted pairs:
    # dr (3), sac (6)
    # ms (5), mfs (4)
    dr_mask = (y_true == 3)
    dr_total = int(np.sum(dr_mask))
    dr_as_sac = int(np.sum(dr_mask & (y_pred == 6)))
    
    sac_mask = (y_true == 6)
    sac_total = int(np.sum(sac_mask))
    sac_as_dr = int(np.sum(sac_mask & (y_pred == 3)))
    
    ms_mask = (y_true == 5)
    ms_total = int(np.sum(ms_mask))
    ms_as_mfs = int(np.sum(ms_mask & (y_pred == 4)))
    
    mfs_mask = (y_true == 4)
    mfs_total = int(np.sum(mfs_mask))
    mfs_as_ms = int(np.sum(mfs_mask & (y_pred == 5)))
    
    return {
        "dr_total": dr_total,
        "dr_as_sac": dr_as_sac,
        "dr_as_sac_pct": float(dr_as_sac / dr_total * 100) if dr_total > 0 else 0.0,
        "sac_total": sac_total,
        "sac_as_dr": sac_as_dr,
        "sac_as_dr_pct": float(sac_as_dr / sac_total * 100) if sac_total > 0 else 0.0,
        "ms_total": ms_total,
        "ms_as_mfs": ms_as_mfs,
        "ms_as_mfs_pct": float(ms_as_mfs / ms_total * 100) if ms_total > 0 else 0.0,
        "mfs_total": mfs_total,
        "mfs_as_ms": mfs_as_ms,
        "mfs_as_ms_pct": float(mfs_as_ms / mfs_total * 100) if mfs_total > 0 else 0.0
    }


def run_statistical_tests(e1_path: Path, e5_path: Path, output_json: Path):
    print(f"Loading E1 (RA-STGT v1) OOF from: {e1_path}")
    e1_rows = load_oof_csv(e1_path)
    print(f"Loading E5 (PR-STGT) OOF from: {e5_path}")
    e5_rows = load_oof_csv(e5_path)
    
    e1_m = compute_fold_metrics(e1_rows)
    e5_m = compute_fold_metrics(e5_rows)
    
    # 1. Paired t-test on Macro-F1
    t_stat_f1, p_val_f1 = stats.ttest_rel(e5_m["f1s"], e1_m["f1s"])
    # 2. Wilcoxon signed-rank test on Macro-F1
    w_stat_f1, p_val_w_f1 = stats.wilcoxon(e5_m["f1s"], e1_m["f1s"])
    
    # 3. Paired t-test on Accuracy
    t_stat_acc, p_val_acc = stats.ttest_rel(e5_m["accs"], e1_m["accs"])
    w_stat_acc, p_val_w_acc = stats.wilcoxon(e5_m["accs"], e1_m["accs"])
    
    # Cohen's d effect size for paired samples
    diff_f1 = e5_m["f1s"] - e1_m["f1s"]
    d_f1 = float(np.mean(diff_f1) / np.std(diff_f1, ddof=1)) if np.std(diff_f1, ddof=1) > 0 else 0.0
    
    # Confusion diagnostic stats
    e1_conf = compute_confusion_stats(e1_rows)
    e5_conf = compute_confusion_stats(e5_rows)
    
    results = {
        "e1_metrics": {
            "mean_acc": float(np.mean(e1_m["accs"])),
            "std_acc": float(np.std(e1_m["accs"])),
            "mean_f1": float(np.mean(e1_m["f1s"])),
            "std_f1": float(np.std(e1_m["f1s"])),
            "fold_accs": e1_m["accs"].tolist(),
            "fold_f1s": e1_m["f1s"].tolist()
        },
        "e5_metrics": {
            "mean_acc": float(np.mean(e5_m["accs"])),
            "std_acc": float(np.std(e5_m["accs"])),
            "mean_f1": float(np.mean(e5_m["f1s"])),
            "std_f1": float(np.std(e5_m["f1s"])),
            "fold_accs": e5_m["accs"].tolist(),
            "fold_f1s": e5_m["f1s"].tolist()
        },
        "statistical_tests": {
            "macro_f1": {
                "t_statistic": float(t_stat_f1),
                "p_value_ttest": float(p_val_f1),
                "wilcoxon_statistic": float(w_stat_f1),
                "p_value_wilcoxon": float(p_val_w_f1),
                "cohens_d": d_f1,
                "is_significant_p05": bool(p_val_f1 < 0.05)
            },
            "accuracy": {
                "t_statistic": float(t_stat_acc),
                "p_value_ttest": float(p_val_acc),
                "wilcoxon_statistic": float(w_stat_acc),
                "p_value_wilcoxon": float(p_val_w_acc),
                "is_significant_p05": bool(p_val_acc < 0.05)
            }
        },
        "confusable_pair_resolution": {
            "drumming_as_clapping": {
                "e1_count": e1_conf["dr_as_sac"],
                "e1_pct": e1_conf["dr_as_sac_pct"],
                "e5_count": e5_conf["dr_as_sac"],
                "e5_pct": e5_conf["dr_as_sac_pct"],
                "absolute_reduction": e1_conf["dr_as_sac"] - e5_conf["dr_as_sac"],
                "relative_reduction_pct": float((e1_conf["dr_as_sac"] - e5_conf["dr_as_sac"]) / e1_conf["dr_as_sac"] * 100) if e1_conf["dr_as_sac"] > 0 else 0.0
            },
            "clapping_as_drumming": {
                "e1_count": e1_conf["sac_as_dr"],
                "e1_pct": e1_conf["sac_as_dr_pct"],
                "e5_count": e5_conf["sac_as_dr"],
                "e5_pct": e5_conf["sac_as_dr_pct"],
                "absolute_reduction": e1_conf["sac_as_dr"] - e5_conf["sac_as_dr"]
            },
            "maracas_shaking_as_forward": {
                "e1_count": e1_conf["ms_as_mfs"],
                "e1_pct": e1_conf["ms_as_mfs_pct"],
                "e5_count": e5_conf["ms_as_mfs"],
                "e5_pct": e5_conf["ms_as_mfs_pct"],
                "absolute_reduction": e1_conf["ms_as_mfs"] - e5_conf["ms_as_mfs"],
                "relative_reduction_pct": float((e1_conf["ms_as_mfs"] - e5_conf["ms_as_mfs"]) / e1_conf["ms_as_mfs"] * 100) if e1_conf["ms_as_mfs"] > 0 else 0.0
            },
            "maracas_forward_as_shaking": {
                "e1_count": e1_conf["mfs_as_ms"],
                "e1_pct": e1_conf["mfs_as_ms_pct"],
                "e5_count": e5_conf["mfs_as_ms"],
                "e5_pct": e5_conf["mfs_as_ms_pct"],
                "absolute_reduction": e1_conf["mfs_as_ms"] - e5_conf["mfs_as_ms"]
            }
        }
    }
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("\nSTATISTICAL SIGNIFICANCE & DIAGNOSTIC RESULTS:")
    print(f"E1 Macro-F1: {results['e1_metrics']['mean_f1']:.4f} ± {results['e1_metrics']['std_f1']:.4f}")
    print(f"E5 Macro-F1: {results['e5_metrics']['mean_f1']:.4f} ± {results['e5_metrics']['std_f1']:.4f}")
    print(f"Paired t-test: t = {t_stat_f1:.4f}, p = {p_val_f1:.4e} (Significant at p<0.05: {results['statistical_tests']['macro_f1']['is_significant_p05']})")
    print(f"Wilcoxon W: W = {w_stat_f1:.4f}, p = {p_val_w_f1:.4e}")
    print(f"Cohen's d: {d_f1:.4f}")
    print("\nConfusable Pair Diagnostics:")
    print(f"Drumming misclassified as Clapping: {e1_conf['dr_as_sac']} ({e1_conf['dr_as_sac_pct']:.1f}%) -> {e5_conf['dr_as_sac']} ({e5_conf['dr_as_sac_pct']:.1f}%) | Reduction: {results['confusable_pair_resolution']['drumming_as_clapping']['relative_reduction_pct']:.1f}%")
    print(f"Maracas Shaking misclassified as Forward: {e1_conf['ms_as_mfs']} ({e1_conf['ms_as_mfs_pct']:.1f}%) -> {e5_conf['ms_as_mfs']} ({e5_conf['ms_as_mfs_pct']:.1f}%) | Reduction: {results['confusable_pair_resolution']['maracas_shaking_as_forward']['relative_reduction_pct']:.1f}%")
    return results


if __name__ == "__main__":
    e1_p = ROOT / "novel_contribution" / "results" / "metrics" / "ra_stgt_full_oof.csv"
    e2_p = ROOT / "research_stage2" / "experiments" / "e2_topology_only_oof.csv"
    e5_p = ROOT / "research_stage2" / "experiments" / "e5_full_pr_stgt_oof.csv"
    out_j = ROOT / "research_stage2" / "experiments" / "statistical_testing_results.json"
    
    if e5_p.exists() and e1_p.exists():
        print("=== Statistical Testing: E5 (Full PR-STGT) vs E1 (RA-STGT v1) ===")
        run_statistical_tests(e1_p, e5_p, out_j)
        
        if e2_p.exists():
            print("\n=== Statistical Testing: E2 (Topology Only) vs E1 (RA-STGT v1) ===")
            out_j2 = ROOT / "research_stage2" / "experiments" / "statistical_testing_e2_vs_e1.json"
            run_statistical_tests(e1_p, e2_p, out_j2)
    else:
        print(f"File not found: e1={e1_p.exists()}, e5={e5_p.exists()}")
