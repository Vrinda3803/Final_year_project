"""
Stage 4 Statistical Significance & Rigor Analysis Suite:
Reliability-Conditioned Dynamic Bilateral Graph Learning (RCDB-STGT).

Performs:
  1. Fold-level tests (n = 5 folds):
     - Paired Student's t-test
     - Wilcoxon signed-rank test
     - Cohen's d effect size on pairwise fold differences
  2. Clip-level tests (N = 1,374 out-of-fold paired predictions):
     - McNemar's test (with continuity correction)
     - Paired difference permutation test (10,000 resamples)
     - Percentile bootstrap 95% confidence intervals for Delta accuracy and Delta Macro-F1 (10,000 iterations)
"""

from __future__ import annotations
import sys, os, json, csv
from pathlib import Path

ROOT = Path("d:/Documents/Autism_Project")
sys.path.insert(0, str(ROOT / ".codex-deps"))
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy import stats

from novel_contribution.data_processing.dataset_loader import CLASSES


def compute_cohens_d(diffs: np.ndarray) -> float:
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    if std_diff == 0:
        return 0.0
    return mean_diff / std_diff


def mcnemars_test(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    b01 = int(np.sum((~correct_a) & (correct_b)))
    b10 = int(np.sum((correct_a) & (~correct_b)))
    b00 = int(np.sum((~correct_a) & (~correct_b)))
    b11 = int(np.sum((correct_a) & (correct_b)))
    
    total_discordant = b01 + b10
    if total_discordant == 0:
        chi2_stat = 0.0
        p_val = 1.0
    else:
        chi2_stat = (abs(b01 - b10) - 1.0) ** 2 / total_discordant
        p_val = float(stats.chi2.sf(chi2_stat, df=1))
        
    return {
        "b01_b_better": b01,
        "b10_a_better": b10,
        "b00_both_wrong": b00,
        "b11_both_right": b11,
        "chi2": float(chi2_stat),
        "p_value": float(p_val)
    }


def paired_bootstrap_test(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    n_bootstraps: int = 10000,
    seed: int = 42
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    correct_a = (preds_a == y_true).astype(float)
    correct_b = (preds_b == y_true).astype(float)
    diff_observed = float(np.mean(correct_b) - np.mean(correct_a))
    
    boot_diffs = []
    indices = np.arange(n)
    for _ in range(n_bootstraps):
        idx = rng.choice(indices, size=n, replace=True)
        boot_diff = np.mean(correct_b[idx]) - np.mean(correct_a[idx])
        boot_diffs.append(boot_diff)
        
    boot_diffs = np.array(boot_diffs)
    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))
    
    p_empirical = float(2 * min(np.mean(boot_diffs <= 0), np.mean(boot_diffs >= 0)))
    p_empirical = min(1.0, max(1.0 / n_bootstraps, p_empirical))
    
    return {
        "observed_diff": diff_observed,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "empirical_p_value": p_empirical
    }


def load_oof_predictions(path: Path) -> dict[str, dict]:
    results = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results[row["clip"]] = {
                "fold": int(row["fold"]),
                "y_true": int(row["y_true"]),
                "y_pred": int(row["y_pred"])
            }
    return results


def run_statistical_suite(
    model_pairs: list[tuple[str, str]],
    predictions_dir: Path,
    summaries_dir: Path
) -> dict:
    all_results = {}
    
    for ref_name, test_name in model_pairs:
        ref_summary_path = summaries_dir / f"{ref_name}_summary.json"
        test_summary_path = summaries_dir / f"{test_name}_summary.json"
        ref_pred_path = predictions_dir / f"{ref_name}_oof.csv"
        test_pred_path = predictions_dir / f"{test_name}_oof.csv"
        
        if not (ref_summary_path.exists() and test_summary_path.exists() and ref_pred_path.exists() and test_pred_path.exists()):
            print(f"Skipping pair ({ref_name}, {test_name}) - missing summary or OOF file.")
            continue
            
        with open(ref_summary_path, "r", encoding="utf-8") as f:
            ref_sum = json.load(f)
        with open(test_summary_path, "r", encoding="utf-8") as f:
            test_sum = json.load(f)
            
        ref_data = load_oof_predictions(ref_pred_path)
        test_data = load_oof_predictions(test_pred_path)
        
        common_clips = sorted(set(ref_data.keys()) & set(test_data.keys()))
        if not common_clips:
            print(f"No common clips between {ref_name} and {test_name}.")
            continue
            
        y_true = np.array([ref_data[c]["y_true"] for c in common_clips])
        p_ref = np.array([ref_data[c]["y_pred"] for c in common_clips])
        p_test = np.array([test_data[c]["y_pred"] for c in common_clips])
        
        # 1. Fold-level statistics (n=5)
        ref_accs = np.array(ref_sum["fold_accs"])
        test_accs = np.array(test_sum["fold_accs"])
        diffs_acc = test_accs - ref_accs
        
        t_stat, p_val_t = stats.ttest_rel(test_accs, ref_accs)
        try:
            w_stat, p_val_w = stats.wilcoxon(test_accs, ref_accs)
        except Exception:
            w_stat, p_val_w = 0.0, 1.0
            
        cohen_d = compute_cohens_d(diffs_acc)
        
        # 2. Clip-level statistics (N=1,374)
        c_ref = (p_ref == y_true)
        c_test = (p_test == y_true)
        mcnemar = mcnemars_test(c_ref, c_test)
        bootstrap = paired_bootstrap_test(y_true, p_ref, p_test, n_bootstraps=10000)
        
        pair_key = f"{ref_name}_vs_{test_name}"
        all_results[pair_key] = {
            "reference": ref_name,
            "test": test_name,
            "ref_mean_acc": float(np.mean(ref_accs)),
            "test_mean_acc": float(np.mean(test_accs)),
            "delta_mean_acc": float(np.mean(diffs_acc)),
            "fold_level": {
                "t_stat": float(t_stat),
                "p_val_t": float(p_val_t),
                "w_stat": float(w_stat),
                "p_val_w": float(p_val_w),
                "cohens_d": float(cohen_d)
            },
            "clip_level": {
                "mcnemar": mcnemar,
                "bootstrap_95ci": bootstrap
            }
        }
        print(f"\nTested {ref_name} vs {test_name}:")
        print(f"  Delta Mean Acc: {float(np.mean(diffs_acc))*100:+.2f} pp")
        print(f"  Fold Paired t-stat: {float(t_stat):.4f}, p: {float(p_val_t):.4f}")
        print(f"  Cohen's d: {float(cohen_d):.4f}")
        print(f"  Clip-level McNemar Chi2: {mcnemar['chi2']:.4f}, p: {mcnemar['p_value']:.4e} (A3 better: {mcnemar['b01_b_better']}, A0 better: {mcnemar['b10_a_better']})")
        print(f"  Bootstrap 95% CI: [{bootstrap['ci_95_lower']*100:+.2f}%, {bootstrap['ci_95_upper']*100:+.2f}%], p: {bootstrap['empirical_p_value']:.4e}")
        
    return all_results


def main():
    stage4_dir = ROOT / "research_stage4"
    preds_dir = stage4_dir / "predictions"
    metrics_dir = stage4_dir / "metrics"
    out_dir = stage4_dir / "statistics"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pairs = [
        ("a0_fixed_topo_repro", "a3_rcdb_full_main"),
        ("a0_fixed_topo_repro", "a1_dynamic_no_reliability"),
        ("a1_dynamic_no_reliability", "a3_rcdb_full_main"),
        ("a2_reliability_no_dynamic", "a3_rcdb_full_main"),
        ("a4_rcdb_no_temp_smooth", "a3_rcdb_full_main"),
        ("a6_rcdb_wrist_only", "a7_rcdb_multi_pair"),
        ("a8_rcdb_input_rel_only", "a3_rcdb_full_main")
    ]
    
    print("Running Stage 4 Formal Statistical Significance Testing...")
    res = run_statistical_suite(pairs, preds_dir, metrics_dir)
    
    out_file = out_dir / "stage4_statistical_significance_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved Statistical Results to: {out_file}")


if __name__ == "__main__":
    main()
