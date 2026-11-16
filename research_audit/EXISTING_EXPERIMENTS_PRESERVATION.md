# EXISTING EXPERIMENTS PRESERVATION RECORD

**Author/Role:** Senior AI/ML Research Engineer & Computer Vision Researcher  
**Project:** Autism Therapy Activity Recognition (Final-Year BTech CSE-AI Project)  
**Date of Record:** November 13, 2026  
**Preservation Mandate:** Non-negotiable historical record. No experiment, code file, or result output shall be modified, deleted, overwritten, or re-scored.

---

## 1. Preservation Principles & Objective

In scientific research, negative, exploratory, and interim results are vital evidence. They prevent circular research paths, expose methodological pitfalls, and anchor future claims in rigorous reality.

This document serves as an immutable register of all experimental work completed prior to the implementation of the novel contribution. Every metric below is preserved exactly as recorded in the local result JSONs and CSVs.

---

## 2. Historical Experiment Registry & Metrics

### A. MMASD Feature-Based Activity Recognition (Supervised 11-Class)

* **Script:** `experiments/mmasd_advanced_90.py`
* **Result Artifacts:** `experiments/mmasd_advanced_90_results_2026_11_12.json`, `results/benchmarks_2026_11_12.json`
* **Dataset & Representation:**
  * 1,693 activity recordings across 11 activity classes.
  * 32 stored group IDs derived from recording filenames.
  * 1,754 initial features: 1,248 multi-bank temporal features (moments, temporal quantiles, ordered segment bins) + 506 rich kinematic features.
  * Fold-local `RobustScaler` and ANOVA `SelectKBest` retaining top 800 features.
  * 11 training-fold class-prototype Euclidean distance features appended.
* **Classifier:** Weighted voting ensemble of Random Forest (weight: 0.40), ExtraTrees (weight: 0.35), and XGBoost (weight: 0.25).
* **Validation Protocol:** 5-fold cross-validation grouped by stored ID (`GroupKFold`), Seed 42.
* **Preserved Metrics:**
  * **Accuracy:** **82.87%** (1,403 / 1,693 correct)
  * **Macro-F1:** **0.7912**
  * Weighted-F1: 0.8256
* **Key Observations & Caveats:**
  * This is the highest supervised activity classification score achieved locally on MMASD.
  * It represents an 11-class problem on tabular summary features, NOT a 10-class graph convolution model.
  * It cannot be directly compared to published 10-class skeleton papers without reconciling classes (e.g. inclusion of frog pose `fg`) and verifying participant IDs.

---

### B. MMASD Unsupervised Temporal Autoencoder & Dynamics

* **Scripts:** `experiments/mmasd_dynamics_replication.py`, `experiments/mmasd_temporal_autoencoder_replication.py`
* **Result Artifacts:** `results/mmasd_dynamics_replication_2026_11_12.json`, `results/mmasd_temporal_autoencoder_2026_11_12.json`
* **Motivation:** Exploration of Akinwumi, Qian, & Ojo (Frontiers in Neuroscience 2026) regarding unsupervised behavioural primitive discovery in autism therapy.
* **Dataset & Setup:**
  * 7,049 temporal windows extracted from 1,661 recordings.
  * 8 time steps per window (stride 4, 50% temporal overlap).
  * 64 kinematic channels per frame.
  * 6 cluster primitives evaluated across 5 stored-ID folds.
* **Preserved Metrics:**

| Method | Silhouette Score | Activity ARI | Activity NMI | Seed Stability ARI |
|---|---:|---:|---:|---:|
| PCA + k-means Baseline | 0.0343 | 0.0928 | 0.2198 | 0.7422 |
| Denoising Transformer Autoencoder | **0.1474** | **0.2886** | **0.4406** | **0.6203** |

* **Temporal Transition Analysis:**
  * Observed self-transition rate: **0.9213**
  * Shuffled permutation comparison (199 permutations): **0.8936** ($p = 0.005$)
* **Critical Methodological Caveat:**
  * Because windows overlap by 50% of their frames by construction, temporal persistence is inherently inflated by structural frame sharing.
  * This result must NOT be interpreted as conclusive clinical proof of behavioral perseveration.

---

### C. Quality-Aware Spatiotemporal Graph & BiGRU Experiments

* **Scripts:** `models/quality_aware_stgcn.py`, `experiments/mmasd_quality_aware_label_efficiency.py`
* **Result Artifacts:** `results/mmasd_quality_aware_label_efficiency_2026_11_12.json`, `results/mmasd_quality_aware_label_efficiency_summary_2026_11_12.csv`
* **Architecture:**
  * Anatomical graph over 8 aggregated body groups.
  * Spatiotemporal graph convolution coupled with a Bidirectional GRU (BiGRU).
  * Quality mask input indicating observed body groups.
  * Masked-feature reconstruction auxiliary pretraining objective.
* **Controlled Conditions:**
  * Evaluated at 25%, 50%, and 100% labeled training budgets.
  * Evaluated under clean input and synthetic 20% and 40% body-group occlusion.
* **Preserved Metrics (Macro-F1):**

| Training Label Budget | Model Architecture / Variant | Clean Input Macro-F1 | 40% Synthetic Masking Macro-F1 |
|---:|---|---:|---:|
| 25% Labels | Supervised (without quality signal) | 0.3304 | 0.2482 |
| 25% Labels | Supervised (with quality mask) | 0.3372 | 0.3174 |
| 25% Labels | Self-Supervised Pretrained + Quality Mask | 0.3318 | 0.3164 |
| 50% Labels | Supervised (without quality signal) | 0.4197 | 0.2845 |
| 50% Labels | Supervised (with quality mask) | 0.4395 | 0.3528 |
| 50% Labels | Self-Supervised Pretrained + Quality Mask | 0.4058 | 0.3667 |
| 100% Labels | Supervised (without quality signal) | 0.4806 | 0.3228 |
| 100% Labels | Supervised (with quality mask) | **0.5079** | 0.4174 |
| 100% Labels | Self-Supervised Pretrained + Quality Mask | 0.4750 | **0.4407** |

* **Critical Methodological Caveat:**
  * Self-supervised pretraining improved resilience under heavy 40% synthetic masking (0.4407 vs 0.4174), but **degraded clean-input performance** (0.4750 vs 0.5079).
  * Synthetic block masking does not reflect the natural distribution of keypoint tracking errors or joint-level detection failures.

---

### D. Local 2D OpenPose Archive Experiments (10-Class)

* **Data Archive:** `data/raw/mmasd/Dataset_FINAL/2D skeleton/2D_openpose.zip`
* **Processed Cache:** `data/processed/mmasd_openpose/mmasd_openpose_64frames.npz` (1,374 clips, 10 classes, 32 IDs)
* **Scripts:** `experiments/mmasd_openpose_*.py`
* **Result Artifacts:** `results/mmasd_openpose_*_2026_11_12.json`
* **Preserved Metrics:**

| Experiment Script | Method / Architecture | Mean Fold Accuracy (± SD) | Mean Fold Macro-F1 (± SD) | Pooled OOF Metrics |
|---|---|---:|---:|---|
| `mmasd_openpose_quality_graph.py` | Confidence-blind Graph + BiGRU | 39.84% ± 8.94 | 0.3219 ± 0.1068 | Acc: 39.81%, F1: 0.3204 |
| `mmasd_openpose_quality_graph.py` | Confidence-aware Graph + BiGRU | 34.88% ± 9.37 | 0.2584 ± 0.1161 | Acc: 34.86%, F1: 0.2571 |
| `mmasd_openpose_temporal_features.py` | Temporal-summary MLP (2,000 dims) | 70.17% ± 10.77 | 0.6338 ± 0.1903 | Acc: 70.16%, F1: 0.6782 |
| `mmasd_openpose_reliability_features.py`| Confidence-weighted MLP concat | 70.24% ± 8.44 | 0.6255 ± 0.1929 | Acc: 70.23%, F1: 0.6710 |
| `mmasd_openpose_mlp_seed_ensemble.py` | 3-Seed MLP Probability Ensemble | 70.83% ± 8.98 | 0.6226 ± 0.1903 | Acc: 70.82%, F1: 0.6698 |
| `mmasd_openpose_hierarchical_mlp.py` | Auxiliary Theme-Head Multi-Task | 68.57% ± 11.63 | 0.6080 ± 0.2073 | Acc: 68.56%, F1: 0.6482 |
| `mmasd_openpose_multistream_features.py`| Joint + Bone + Motion Multi-Stream | 70.74% ± 11.19 | 0.6265 ± 0.2082 | Acc: 70.74%, F1: 0.6725 |
| `mmasd_openpose_rbf_kernel.py` | **Nested RBF Kernel Ridge Regression** | **71.69% ± 9.10** | **0.6394 ± 0.1918** | **Acc: 71.69%, Macro-F1: 0.6920** |
| `mmasd_openpose_rbf_prior_adjust.py` | RBF Ridge with Class-Prior Bias | 69.29% ± 11.85 | 0.6207 ± 0.2265 | Acc: 69.29%, F1: 0.6734 |
| `mmasd_openpose_nested_trees.py` | Nested Random Forest / ExtraTrees | 68.42% ± 8.34 | 0.6242 ± 0.1845 | Acc: 68.41%, Macro-F1: 0.6517 |

* **Key Takeaway:**
  * Naive Graph+BiGRU performed very poorly (<40% accuracy) on this dataset.
  * Merely weighting features with confidence or concatenating confidence vectors failed to improve macro-F1.
  * Dense kernel ridge regression on temporal bin moments remains the strongest local 2D baseline (71.69% accuracy / 0.6920 pooled macro-F1).

---

### E. Adaptive Sampling via Reinforcement Learning (PPO)

* **Scripts:** `experiments/mmasd_adaptive_sampling_rl.py`, `experiments/avasd_adaptive_sampling_rl.py`
* **Result Artifacts:** `results/mmasd_adaptive_sampling_rl_2026_11_12.json`, `results/avasd_adaptive_sampling_rl_2026_11_12.json`
* **Setup:** Evaluated PPO policy to select an 8-window budget per clip using training-set descriptor novelty as a proxy reward.
* **Preserved Metrics:**
  * **MMASD 5-Fold Grouped CV:**
    * Uniform Sampler: 67.29% ± 6.32 accuracy, **0.5762 ± 0.0823 macro-F1**
    * Random Sampler: 67.80% ± 4.98 accuracy, **0.5939 ± 0.0868 macro-F1**
    * Novelty Heuristic: 67.65% ± 5.47 accuracy, **0.5843 ± 0.0760 macro-F1**
    * PPO Policy: 66.62% ± 6.03 accuracy, **0.5701 ± 0.0883 macro-F1**
  * **AV-ASD 5-Fold Grouped CV:**
    * Uniform Sampler: 62.22% accuracy, **0.5067 macro-F1**
    * Random Sampler: 59.87% accuracy, **0.4992 macro-F1**
    * Novelty Heuristic: 60.98% accuracy, **0.5033 macro-F1**
    * PPO Policy: 60.98% accuracy, **0.4941 macro-F1**
* **Definitive Conclusion:**
  * PPO sampling was an exploratory negative result.
  * Novelty-based RL policies did not outperform uniform or random temporal sampling.
  * The policy learned descriptor novelty, NOT a clinical therapy policy.

---

### F. AV-ASD Local Experiments (Acoustic & Video Audits)

* **Scripts:** `experiments/avasd_multilabel_audio_grouped.py`, `experiments/avasd_video_level_nested.py`, `experiments/avasd_video_context_nested.py`
* **Result Artifacts:** `results/avasd_*_2026_11_12.json`
* **Preserved Metrics:**
  1. **10-Indicator Multi-Label Audio Task (`avasd_multilabel_audio_grouped.py`):**
     * 227 clips with usable aligned audio across 138 source-video groups.
     * Behavior Macro-F1 (fixed 0.5 threshold): **0.2811**
     * Behavior Micro-F1: **0.3145**
     * Macro Average Precision: **0.2202**
     * Exact Match Ratio: **0.0176**
  2. **Earlier 3-Class Segment Task (`avasd_3class_hyper.py`):**
     * 961 segments from 87 video IDs.
     * Pooled Accuracy: **76.59%**, Macro-F1: **0.5370**
     * **Audit Warning:** Extreme class and sample imbalance. Class support: 206 aggressive, 55 reactivity, and 700 upper-limb stereotypies. A single 29-minute video contributed **506 of the 961 segments** (all labeled upper-limb stereotypies). This 76.59% accuracy does NOT reflect generalizable performance.
  3. **Video-Level Aggregated Classifier (`avasd_video_level_nested.py`):**
     * 87 video units evaluated with 5-fold stratified CV.
     * Video-Level Accuracy: **60.92%**
     * Video-Level Macro-F1: **0.5492**
     * Majority-Class Video Baseline: **52.87%**

---

## 3. Preservation Verification Matrix

```
+---------------------------------------------------------------------------------------------------------+
|                                    HISTORICAL RESULTS INTEGRITY LOCK                                    |
+------------------------------+---------------------------+-----------------------+----------------------+
| Experiment                   | Primary Script            | Saved Result Metric   | Integrity Status     |
+------------------------------+---------------------------+-----------------------+----------------------+
| MMASD 11-Class Feature Ens.  | mmasd_advanced_90.py      | 82.87% Acc / 0.7912 F1| VERIFIED & LOCKED    |
| MMASD Autoencoder Dynamics   | mmasd_dynamics_*.py       | ARI 0.2886 / NMI 0.440| VERIFIED & LOCKED    |
| Quality Graph (Synthetic)    | quality_aware_stgcn.py    | 40% Mask F1: 0.4407   | VERIFIED & LOCKED    |
| MMASD 2D OpenPose RBF Pilot  | mmasd_openpose_rbf_*.py   | 71.69% Acc / 0.6920 F1| VERIFIED & LOCKED    |
| MMASD PPO Sampler            | mmasd_adaptive_sampling.py| 0.5701 Macro-F1       | VERIFIED & LOCKED    |
| AV-ASD 10-Label Audio        | avasd_multilabel_audio.py | 0.2811 Macro-F1       | VERIFIED & LOCKED    |
| AV-ASD 3-Class Segment       | avasd_3class_hyper.py     | 76.59% Acc (506 imbal)| VERIFIED & LOCKED    |
| AV-ASD Video-Level           | avasd_video_level_nested  | 60.92% Acc / 0.5492 F1| VERIFIED & LOCKED    |
+------------------------------+---------------------------+-----------------------+----------------------+
```
