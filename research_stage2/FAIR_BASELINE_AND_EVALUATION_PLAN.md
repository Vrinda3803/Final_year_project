# Stage 2: Fair Baseline and Evaluation Plan

**Project:** Autism Therapy Physical Activity Recognition using 2D Skeleton Dynamics  
**Date:** March 2026  
**Scope:** Experimental protocol, leak-proof validation integrity, baseline harmonization, statistical significance testing, and evaluation metrics for comparing Stage 2 models against prior work.

---

## 1. Experimental Integrity and Leak-Proof Protocol

To guarantee that performance comparisons between our proposed Stage 2 contribution and existing baselines are experimentally defensible, the evaluation protocol adheres to five non-negotiable rules:

1. **Strictly Identical Fold Partitions:** All models must be evaluated on the exact same 5-fold `GroupKFold` splits generated in Stage 1 (`novel_contribution/data_processing/dataset_loader.py`). The 32 stored group/subject IDs are partitioned into:
   - Fold 0: 279 samples (6 test groups)
   - Fold 1: 276 samples (7 test groups)
   - Fold 2: 275 samples (6 test groups)
   - Fold 3: 271 samples (6 test groups)
   - Fold 4: 273 samples (7 test groups)
   Total: 1,374 recordings across 10 activity classes. No subject group may appear in both train and validation sets for any fold.
2. **Fixed Random Seeding:** Global seed is locked to `seed = 42` across Python `random`, `numpy`, PyTorch CPU, and scikit-learn.
3. **No Target Leakage in Preprocessing:** Normalization statistics (skeletal bounding box, scale center, coordinate standardization) must be computed either sample-locally or strictly from the training fold without accessing validation fold data.
4. **Identical Hardware and Compute Environment:** All training and evaluation run on standard CPU (PyTorch 2.15.0+cpu, Windows 11) to maintain exact runtime latency comparability.
5. **Preservation of Existing Checkpoints and Logs:** All outputs from Stage 1 (`novel_contribution/`) remain read-only. All Stage 2 models, logs, predictions, and ablation checkpoints are stored strictly in `research_stage2/experiments/`.

---

## 2. Comprehensive Suite of Benchmark Baselines

To ensure rigorous context, the proposed method must be compared against all historical and current local baselines across the 10-class MMASD 2D OpenPose task:

| Baseline Model | Input Modality & Representation | Architecture Type | Reported Fold Acc | Reported Fold Macro-F1 | Pooled OOF Macro-F1 | Status |
|---|---|---|:---:|:---:|:---:|:---:|
| **1. Feature Ensemble** | 1,754 engineered kinematic features | RF / ExtraTrees / XGBoost | — | — | 0.7912* | Historical (11-class; not directly comparable) |
| **2. RBF Kernel Baseline** | 2D OpenPose trajectory statistics | SVM with RBF Kernel | 71.69% | 0.6394 | 0.6920 | Completed local baseline (10-class) |
| **3. Confidence-Blind Graph+BiGRU** | 2D Skeletons without confidence ($c=1$) | 2-layer GCN + BiGRU | 39.84% | 0.3219 | 0.3219 | Historical local benchmark |
| **4. Confidence-Aware Graph+BiGRU** | 2D Skeletons with confidence masking | 2-layer GCN + BiGRU | 34.88% | 0.2584 | 0.2584 | Historical local benchmark |
| **5. Baseline STGT** (Stage 1 Ablation) | 2D Skeletons without reliability gating | Standard GAT + Temporal Transformer | 37.86% ± 10.81% | 0.2911 ± 0.0791 | 0.3644 | Completed Stage 1 benchmark |
| **6. RA-STGT v1** (Stage 1 Model) | 2D Skeletons + Joint Velocity Prior | SR-GAT + RATA + ECE Calibration | 46.47% ± 7.84% | 0.4137 ± 0.1328 | 0.4510 | Completed Stage 1 benchmark |
| **7. PR-STGT / RA-STGT v2** (Proposed) | Augmented Bi-Relational Graph + Phase | PR-GAT + Phase-RATA + $\mathcal{L}_{\text{CPM}}$ | **Target: >52.0%** | **Target: >0.480** | **Target: >0.510** | **Stage 2 Target** |

*\*Note on Feature Ensemble:* The 11-class feature ensemble includes an additional `rest` class and uses pre-extracted tabular features with ANOVA selection. It serves as an upper bound on feature engineering, whereas our research focuses on end-to-end spatio-temporal deep graph learning from raw noisy coordinate sequences.

---

## 3. Evaluation Metrics Suite

To provide a complete evaluation of accuracy, calibration, fine-grained class balance, and clinical utility, all models must be benchmarked on:

### A. Classification Performance
- **Fold-wise Mean Accuracy & Standard Deviation:** $\bar{\text{Acc}} = \frac{1}{K} \sum_{k=1}^K \text{Acc}_k \pm \sigma_{\text{Acc}}$
- **Fold-wise Mean Macro-F1 & Standard Deviation:** $\bar{\text{F1}} = \frac{1}{K} \sum_{k=1}^K \text{F1}_{\text{macro}, k} \pm \sigma_{\text{F1}}$
- **Pooled Out-of-Fold (OOF) Macro-F1:** Macro-averaged F1 score computed across all $N=1,374$ concatenated out-of-fold validation predictions.
- **Pooled Out-of-Fold Weighted F1:** Sample-weighted F1 accounting for class frequency.
- **Per-Class F1 Scores:** Tracking all 10 therapy exercise classes, especially target confusable pairs (`dr`, `sac`, `ms`, `mfs`).

### B. Probability Calibration & Uncertainty
- **Expected Calibration Error (ECE):**
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)| \quad (M=10 \text{ bins})$$
- **Maximum Calibration Error (MCE):** $\max_{m=1,\dots,M} |\text{acc}(B_m) - \text{conf}(B_m)|$
- **Negative Log-Likelihood (NLL):** Multi-class cross-entropy on validation predictions.

### C. Selective Prediction & Clinical Risk-Coverage
- **Risk-Coverage Curve:** Evaluated at confidence thresholds $\tau \in [0.0, 0.9]$ in steps of 0.05.
- **Selective Accuracy at Key Coverages:** Recorded at 100%, 80%, 50%, 25%, and 10% coverage.
- **Reliability-Conditioned Rejection:** Assessing whether abstention flags correlate with noisy OpenPose frames.

### D. Missingness Stress Testing
- **Degradation under 0%, 20%, and 40% Synthetic Occlusion:** Simulating OpenPose joint tracking dropouts to measure graceful degradation:
  $$\Delta_{\text{robustness}} = \text{F1}_{0\%} - \text{F1}_{40\%}$$

---

## 4. Statistical Significance Testing Protocol

To prove that improvements achieved by PR-STGT over RA-STGT v1 and Baseline STGT are statistically meaningful and not artifacts of random fold variance:
1. **Paired Student's t-test:** Two-tailed paired $t$-test on fold-wise macro-F1 values ($N_{\text{folds}} = 5$).
2. **Wilcoxon Signed-Rank Test:** Non-parametric paired test on fold-wise performance.
3. **Significance Threshold:** Results are considered statistically significant if $p < 0.05$.
4. **Reporting Format:** Results will report $(t, p)$ and $(W, p)$ explicitly alongside effect sizes (Cohen's $d$).
