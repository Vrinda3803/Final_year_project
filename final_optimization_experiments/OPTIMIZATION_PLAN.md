# COMPREHENSIVE FINAL OPTIMIZATION & RESEARCH PLAN
## Towards a Scientifically Defensible Performance Ceiling on MMASD Skeletons

**Author:** Antigravity AI Research Team  
**Date:** November 19, 2026  
**Primary Target:** Investigate the empirical feasibility of reaching >80% accuracy under strict grouped cross-validation without data manipulation or test leakage.  
**Preserved Contributions:**
1. Dynamic Bilateral Graph Learning.
2. Reliability-Conditioned Bilateral Graph Learning.

---

## 1. Experimental Pillars & Staged Plan

| Experiment Stage | Focus Area | Objective & Controlled Variables | Deliverables |
|:---|:---|:---|:---|
| **Stage 1** | **Hyperparameter Optimisation** | Systematic validation-only search over base channels, LR schedules, dropout, and MS-TCN dilation. | `HYPERPARAMETER_SEARCH_RESULTS.csv` |
| **Stage 2** | **Feature Representation** | Controlled evaluation of bone vectors, acceleration, angular velocity, and bilateral disparity streams. | `FEATURE_ABLATION_RESULTS.csv` |
| **Stage 3** | **Temporal Modeling** | MS-TCN vs. Multi-Scale Residual Temporal Blocks vs. Multi-Kernel Dilations. | `TEMPORAL_ARCHITECTURE_RESULTS.csv` |
| **Stage 4** | **Bilateral Graph Optimization** | Bilateral MLP width, temporal edge consistency regularizer, and soft reliability transformations. | `BILATERAL_GRAPH_ABLATIONS.csv` |
| **Stage 5** | **Data Augmentation & Regularization** | Skeleton-safe affine jitter, mild temporal resampling, and joint masking on train folds only. | `AUGMENTATION_RESULTS.csv` |
| **Stage 6** | **Class Imbalance & Confusing Pairs** | Effective-number class weighting, focal loss, and confusable margin penalties. | `CLASS_IMBALANCE_RESULTS.csv` |
| **Stage 7** | **Hybrid Ensemble Optimization** | Leakage-free out-of-fold validation-tuned probability blending and meta-stacking. | `ENSEMBLE_RESULTS.csv` |
| **Stage 8** | **Robustness Stress-Testing** | Synthetic joint dropout, occlusion, and coordinate degradation benchmarking. | `ROBUSTNESS_RESULTS.csv` |
| **Stage 9** | **Final Selection & Documentation** | Selection of best standalone, hybrid, and calibrated models with full multi-seed verification. | `FINAL_MODEL_SELECTION.md`, `EXPERIMENT_REPRODUCIBILITY.md` |

---

## 2. Integrity and Strict Validation Protocols

1. **Strict Cohort Isolation:** No recording group may cross the train/validation/test boundary.
2. **Inner Validation Tuning:** All architecture choices, learning rates, and ensemble blending coefficients must be chosen solely on the 18% held-out validation cohorts from the training fold.
3. **No Retrospective Test Tuning:** Test fold predictions are computed once using the checkpoint selected on validation Macro-F1.
4. **Traceable Results:** Every metric in the final reports must link directly to an existing script, JSON, or CSV record.
