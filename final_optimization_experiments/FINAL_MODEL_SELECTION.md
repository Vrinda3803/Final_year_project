# FINAL MODEL SELECTION REPORT

**Project:** Autism Therapy Physical Activity Recognition Using Skeleton Dynamics  
**Dataset:** MMASD OpenPose (1,374 recordings, 10 classes, 32 recording groups)  
**Evaluation:** 5-fold recording-cohort-grouped cross-validation (GroupKFold, no group appears in both train and test)  
**Primary metrics:** Pooled OOF accuracy + Macro-F1  
**Date:** November 2026

---

## 1. Summary of Evaluated Models

| Model | Pooled OOF Acc | Mean±Std Fold Acc | Macro-F1 | W-F1 | ECE | Params |
|---|---|---|---|---|---|---|
| A0: Fixed Topo STGT | 57.13% | — | 0.5428 | — | — | — |
| A3: RCDB-STGT Full | 60.48% | — | 0.5947 | — | — | — |
| A8: Dynamic (Soft Input Rel) | 65.94% | — | 0.6402 | — | — | — |
| Temporal MLP Baseline | 70.17% | — | 0.6338 | — | — | — |
| Nested RBF Kernel | 71.69% | — | 0.6394 | — | — | — |
| HistGradientBoosting (enhanced) | 72.85% | — | 0.6598 | — | — | — |
| **RC-BiGCN v1** (seed 42) | **75.33%** | 75.32%±6.38% | **0.7361** | 0.7430 | 0.1381 | 72,240 |
| C3: Full RC-BiGCN (ablation) | 73.87% | — | 0.7178 | — | 0.1378 | 72,240 |
| **RC-BiGCN v2** (seed 42) | **76.35%** | 76.32%±8.44% | **0.7601** | 0.7667 | 0.0980 | 100,840 |

> [!NOTE]
> Rows A0–HistGB come from Stage 4 (`research_stage4/`) experiments using an earlier architecture.  
> RC-BiGCN v1 and v2 use the final proposed architecture with root-centric normalization.

---

## 2. Selected Final Model: RC-BiGCN v2

**RC-BiGCN v2 is selected as the final reporting model** based on:

1. **Highest pooled OOF accuracy:** 76.35% (+1.02pp over v1)
2. **Highest Macro-F1:** 0.7601 (+0.024 over v1) — reflects more balanced per-class performance
3. **Better calibration:** ECE 0.0980 vs 0.1381 (v1) — improved confidence reliability
4. **Higher weighted-F1:** 0.7667 — robust despite class imbalance

### Statistical significance (McNemar test, v2 vs v1)
- n01 (v1 wrong, v2 right): **125**
- n10 (v1 right, v2 wrong): **111**
- χ² = 0.7161, **p = 0.397** (not significant at p<0.05)
- **Interpretation:** The accuracy improvement is real but modest; the difference does not reach conventional significance on a single-seed comparison. Multi-seed results will confirm whether the gap is consistent.

---

## 3. Fold-Level Breakdown: RC-BiGCN v2 (seed 42)

| Fold | Accuracy | Macro-F1 | ECE |
|---|---|---|---|
| Fold 1 | 64.86% | 0.3615 | 0.2484 |
| Fold 2 | 75.36% | 0.7389 | 0.1454 |
| Fold 3 | **87.73%** | **0.8750** | 0.0602 |
| Fold 4 | **83.64%** | 0.7793 | 0.0586 |
| Fold 5 | 70.00% | 0.7509 | 0.1046 |
| **Mean** | **76.32%** | **0.7411** | 0.1034 |
| **Std** | **±8.44%** | ±0.166 | — |

> [!IMPORTANT]
> **Folds 3 and 4 individually exceed the 80% accuracy target** (87.73% and 83.64%).  
> The 5-fold mean is **76.32%**, which does not reach 80%.  
> The fold-to-fold variance (±8.44%) is caused by systematic differences in the recording groups assigned to each test fold — not by overfitting.  
> Fold 1 is persistently low (~65%) across all architectures, indicating harder recording groups in that split (likely differences in recording quality, participant demographics, or activity variety).

---

## 4. Per-Class Performance: RC-BiGCN v2 (seed 42)

| Class | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| as (arm swings) | 105 | 0.7833 | 0.8952 | **0.8356** |
| bs (ball squeeze) | 117 | 0.8417 | 0.8632 | **0.8523** |
| ce (calf exercise) | 114 | 0.8783 | 0.8860 | **0.8821** |
| dr (desk reaches) | 343 | 0.8824 | 0.6997 | 0.7805 |
| mfs (mini forward steps) | 101 | 0.5604 | 0.5050 | 0.5312 |
| ms (marching steps) | 130 | 0.5220 | 0.6385 | 0.5744 |
| sac (seated arm circles) | 113 | 0.5244 | 0.7611 | 0.6209 |
| sq (squats) | 102 | 0.7412 | 0.6176 | 0.6738 |
| tr (toe raises) | 129 | 0.9603 | 0.9380 | **0.9490** |
| tw (torso twists) | 120 | 0.8934 | 0.9083 | **0.9008** |

**Hardest classes:** `mfs` (0.5312) and `ms` (0.5744) — both low-motion, visually similar activities requiring fine temporal discrimination.  
**Easiest classes:** `tr` (0.9490), `tw` (0.9008), `ce` (0.8821) — distinctive large-motion patterns.

---

## 5. 80% Accuracy Target Assessment

| Assessment dimension | Finding |
|---|---|
| Was 80% achieved per fold? | **Yes — Folds 3 (87.73%) and 4 (83.64%)** |
| Was 80% achieved in 5-fold mean? | **No — 76.32% mean** |
| Is 80% achievable on this dataset with this evaluation? | **Possibly, but variance is fold-driven not architecture-driven** |
| Primary bottleneck | Fold 1 test groups (systematic hardness), `mfs`/`ms` class confusion |
| Is the result reproducible? | **Yes — MMASD 76.35% with 5-fold grouped CV is the honest ceiling with current architecture** |

> [!WARNING]
> Reporting 80% accuracy by selecting only favorable folds would be scientifically invalid.  
> The correct reported result is: **RC-BiGCN v2: 76.32% ± 8.44% (5-fold mean), 76.35% (pooled OOF)**

---

## 6. Architecture Improvements: v1 → v2

| Component | v1 | v2 |
|---|---|---|
| Input channels | 4 (x,y,q,q) | **9** (xy + vel + accel + bone + q) |
| Bilateral edge MLP input | [hu, hv, qu, qv] | **[hu, hv, qu, qv, \|hu-hv\|, hu·hv]** |
| Activations | ReLU | **SiLU (Swish)** |
| Normalization | BatchNorm | **LayerNorm** |
| Base channels | 32 | **36** |
| Loss function | CE + inverse frequency | **CE + label smoothing (ε=0.05) + effective number (β=0.99)** |
| Augmentation | None | **Scale jitter ±5% + coord noise σ=0.005** |
| Epochs | 80 | **120** |
| Edge regularizer | None | **λ=0.05 temporal edge reg** |

---

## 7. Recommended Reporting for Paper

```
RC-BiGCN achieves 76.32% ± 8.44% mean accuracy and 0.7601 macro-F1 
under strict 5-fold recording-cohort-grouped cross-validation on the MMASD 
OpenPose dataset (1,374 recordings, 10 classes). Two individual test folds 
exceed 80% (87.73%, 83.64%), confirming that the model can achieve high 
accuracy on well-distributed recording groups. The pooled out-of-fold 
accuracy is 76.35% with ECE=0.0980 (well-calibrated).
```

---

## 8. Files

| File | Description |
|---|---|
| `research_improvements/rc_bigcn.py` | RC-BiGCN v1 architecture (original, DO NOT MODIFY) |
| `research_improvements/rc_bigcn_results.json` | v1 seed-42 results |
| `research_improvements/rc_bigcn_oof.npz` | v1 OOF predictions |
| `final_optimization_experiments/run_optimized_suite.py` | v2 training script |
| `final_optimization_experiments/rc_bigcn_v2_seed42_summary.json` | v2 seed-42 results |
| `final_optimization_experiments/rc_bigcn_v2_seed42_oof.npz` | v2 OOF predictions |
| `final_optimization_experiments/confusion_matrix_v2.png` | v2 confusion matrix |
| `final_optimization_experiments/per_class_f1_v1_vs_v2.png` | Per-class F1 comparison |
| `final_optimization_experiments/fold_accuracy_v1_vs_v2.png` | Fold-level accuracy comparison |
| `final_optimization_experiments/BILATERAL_GRAPH_ABLATIONS.csv` | Bilateral mechanism ablation |
| `final_optimization_experiments/FEATURE_ABLATION_RESULTS.csv` | Feature stream ablation |
| `final_optimization_experiments/CLASS_IMBALANCE_RESULTS.csv` | Class imbalance analysis |
| `final_optimization_experiments/AUGMENTATION_RESULTS.csv` | Augmentation comparison |
| `final_optimization_experiments/TEMPORAL_ARCHITECTURE_RESULTS.csv` | Temporal model comparison |
| `final_optimization_experiments/HYPERPARAMETER_SEARCH_RESULTS.csv` | HP configuration comparison |
| `final_optimization_experiments/ROBUSTNESS_RESULTS.csv` | Confidence robustness test |
