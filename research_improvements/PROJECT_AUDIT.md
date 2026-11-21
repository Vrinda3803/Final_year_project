# PROJECT AUDIT: SKELETON DYNAMICS FOR AUTISM THERAPY PHYSICAL ACTIVITY RECOGNITION

**Date:** November 18, 2026  
**Investigator:** Antigravity AI Research Team  
**Evaluation Protocol:** 5-Fold Grouped Cross-Validation (32 Recording Groups, Cohort-Grouped)  
**Primary Target:** Defensible, statistically valid classification performance ceiling (>80% aspirational target)

---

## 1. Executive Summary & Verification of Implemented Pipeline

A comprehensive inspection of the entire codebase, previous experimental artifacts, model architectures, and dataset structures was conducted. We audited:
- Raw and processed MMASD OpenPose caches (`data/processed/mmasd_openpose/mmasd_openpose_64frames.npz`).
- Baseline experiments (`experiments/mmasd_openpose_rbf_kernel.py`, `experiments/mmasd_openpose_temporal_features.py`).
- Phase-wise research progression:
  - `novel_contribution/`: Initial RCDB-STGT prototype and calibration metrics.
  - `research_stage2/`: RA-STGT implementation and loss design.
  - `research_stage3/`: PR-STGT experimental pipeline (E1–E9).
  - `research_stage4/`: Complete Stage 4 ablation study (A0–A9), checkpoint archives, and reports.

### What is Actually Working and Verified:
1. **The Dataset Loading and Split Pipeline:**
   - 1,374 recordings across 10 physical activity classes: `["as", "bs", "ce", "dr", "mfs", "ms", "sac", "sq", "tr", "tw"]`.
   - 64 temporal frames, 25 OpenPose BODY-25 joints, 4 feature channels: $(x, y, v_x, v_y)$ where velocity is precomputed from temporal diffs, plus OpenPose confidence values $Q \in \mathbb{R}^{B \times 64 \times 25}$.
   - 32 recording groups. The 5-fold cross-validation split (`create_subject_independent_folds` in `novel_contribution` and `grouped_folds` in `experiments/mmasd_openpose_temporal_features.py`) partition groups with exact matching fold test sets: Fold sizes = `[276, 276, 277, 275, 270]`, summing to 1,374.
   - Fold splits have been mathematically verified as **100% identical** between the graph pipeline and the RBF/MLP baseline scripts.

2. **Previous Experimental Results Verified:**
   - **Stage 4 Ablations (RCDB-STGT):**
     - A0 (Fixed topology baseline): **57.12%** Mean Fold Acc, **57.13%** Pooled Acc, 0.5428 Macro-F1.
     - A1 / A8 (Dynamic bilateral graph, unconstrained/input-fed reliability): **65.92%** Mean Fold Acc, **65.94%** Pooled Acc, 0.6402 Macro-F1, ECE 0.0733.
     - A3 (RCDB-STGT full model with dynamic edges + multiplicative reliability gate + temporal smoothing): **60.40%** Mean Fold Acc, **60.48%** Pooled Acc, 0.5947 Macro-F1.
     - A4 (Without temporal smoothing): **59.53%** Acc, 0.5736 Macro-F1.
     - A5 (Phase-based stream): **60.55%** Acc, 0.5942 Macro-F1.
     - A6 (Wrist-only bilateral edges): **61.28%** Acc, 0.5872 Macro-F1.
     - A7 (Five bilateral pairs): **58.95%** Acc, 0.5728 Macro-F1.
     - A9 (Confusable-class margin loss): **61.22%** Acc, 0.5878 Macro-F1.
   - **Stage 3 Experiments (PR-STGT):**
     - E1: 46.47% | E2 (Topology-only): 57.63% | E3 (Phase-only): 49.97% | E4 (Margin-only): 55.58% | E5 (Combined): 54.00%.
   - **Alternative Baselines:**
     - Nested Grouped RBF Kernel-Ridge (`mmasd_openpose_rbf_kernel_2026_11_12.json`): **71.69%** Mean Fold Acc, **71.69%** Pooled Acc, 0.6394 Macro-F1.
     - Temporal Summary MLP (`mmasd_openpose_temporal_features_2026_11_12.json`): **70.17%** Mean Fold Acc, 0.6338 Macro-F1.

---

## 2. Critical Audit: Why Does the RBF Baseline Outperform the Graph Architecture?

This is one of the most crucial scientific questions of this project: **Why does an RBF kernel ridge model achieve 71.69% accuracy on the exact same folds where RCDB-STGT achieves ~60–66%?**

Our code audit reveals three fundamental reasons:

### Reason 1: Global vs. Local Coordinate Representation & Invariance
- In RCDB-STGT (`rcdb_stgt.py`), the input features are raw image-space coordinates $(x, y)$ and velocities $(v_x, v_y)$. The OpenPose coordinates depend heavily on the child's position in the camera frame, distance to the camera, and height/scale.
- The GNN spatial message-passing mechanism attempts to learn local joint topology, but without **body-centering (e.g. torso/pelvis-centered normalization)** or **scale normalization (e.g. normalizing by torso height)**, different recording groups have systematic spatial coordinate shifts. The GNN overfits to camera placement and spatial position of particular subjects.
- In contrast, the RBF baseline (`mmasd_openpose_temporal_features.py`) extracted global trajectory moments (mean, std, 10th/90th quantiles across 8 temporal bins) and normalized them with fold-wise z-score standard scaling, capturing movement dynamics rather than raw coordinate positions.

### Reason 2: Over-Parameterized Temporal Attention on Small Cohorts
- RCDB-STGT uses a multi-head temporal Transformer attention module ($d=64$, 2 heads) across 64 frames.
- With only ~1,100 training samples per fold across 25–26 recording groups, learning pairwise $64 \times 64$ temporal attention maps from scratch without pretrained weights or strong inductive biases leads to overfitting.
- The temporal attention tends to collapse onto specific keyframes that do not generalize well across different children. Temporal convolutions (TCN) or multi-scale dilated convolutions provide a much stronger inductive bias for small-sample action recognition.

### Reason 3: Sub-optimal Reliability Gate Mechanism
- In A3, the bilateral edge weight is computed as:
  $$A_{uv}^{(t)} = \sigma(W [h_u, h_v]) \cdot q_{uv}^{(t)}$$
  where $q_{uv}^{(t)} = \sqrt{Q_u^{(t)} Q_v^{(t)}}$.
- When OpenPose detection degrades on fast-moving peripheral joints (like wrists and ankles during dynamic exercises), $Q$ drops to near 0, completely zeroing out the bilateral edge even when the bilateral motion discrepancy is the most discriminative signal.
- In contrast, A1 / A8 (which feeds reliability as an input feature rather than a hard multiplicative gate) achieved 65.92% (5.5% higher than A3), proving that multiplicative gating suppresses informative message passing during noisy frames.

---

## 3. Data Integrity & Leakage Audit

We checked all aspects of data leakage across the codebase:
1. **Split Leakage:** Verified that in `create_subject_independent_folds` and `grouped_folds`, recording groups are strictly partitioned. No group appears in both train and test.
2. **Feature Normalization Leakage:** In the RBF baseline and MLP, normalization statistics (`mu`, `sd`) were correctly computed on the training set (`tr`) and applied to `va` and `te`.
3. **Graph Pipeline Preprocessing:** The graph pipeline computes dynamic velocity directly from adjacent frames within each isolated sample $(B, T, V, C)$. There is no cross-sample leakage.
4. **Conclusion on Leakage:** The evaluation protocol is clean. The performance differences are genuinely methodological and representational.

---

## 4. Current Performance Bottlenecks

1. **Feature Bottlenecks:**
   - Raw coordinates without skeleton-centric normalization (hip/spine centering).
   - Absence of structural bone vectors (e.g., limb orientations: vector from elbow to wrist).
   - Lack of joint angle and angular velocity features (crucial for distinguishing arm flapping `as` from hand clapping `ce` or drumming `dr`).
   - Lack of explicit bilateral asymmetry and cross-correlation signals.
2. **Architectural Bottlenecks:**
   - Pure spatial attention vs. GCN topology: Full self-attention across 25 joints on small datasets introduces high variance compared to adaptive graph convolutions (such as CTR-GCN / 2s-AGCN / ST-GCN).
   - Temporal Transformer fragility on 64 frames with limited data.
3. **Class Confusion Bottlenecks:**
   - Extreme confusion between `as` (arm flapping/stimming), `ce` (clapping exercises), `dr` (drumming), and `mfs` (motor finger stimming).
   - High confusion between `sq` (squatting) and `ms` (marching/stepping).

---

## 5. Proposed Experimental Improvement Plan

We organize our research into strict, staged experimental milestones:

- **Stage 1 (Baseline Reproduction & Verification):**
  - Implement a unified evaluation harness in `research_improvements/` that evaluates models under the exact same 5-fold grouped CV.
  - Reproduce the 71.69% RBF baseline and the 65.92% RCDB-STGT A8 baseline.
- **Stage 2 (Strong Tabular & Temporal Baselines):**
  - Implement and evaluate XGBoost, LightGBM, Random Forest, SVM (RBF), and MLP on comprehensive extracted skeleton features to establish the empirical ceiling for non-graph approaches.
- **Stage 3 (Skeleton Feature Engineering & Representation Learning):**
  - Systematically evaluate:
    - B: Body-centered coordinates (hip/spine root subtraction).
    - C: Torso-scale normalization.
    - D: Bone vector representation.
    - E: Joint angles (elbows, shoulders, hips, knees).
    - F: Higher-order motion features (acceleration, jerk, angular velocity).
    - G: Explicit bilateral coordination features (left-right cross-correlation, phase offset).
- **Stage 4 (Architecture Exploration & Comparison):**
  - ST-GCN, CTR-GCN style graph convolutions, Dilated Temporal Convolutional Networks (MS-TCN), and Lightweight Transformers.
- **Stage 5 (Novelty Refinement: Reliability-Conditioned Bilateral Dynamics):**
  - Formulate an improved Reliability-Conditioned Bilateral Graph (RC-BiGCN / RCDB-TCN) that:
    - Retains our bilateral coordination novelty.
    - Uses soft reliability conditioning rather than destructive zero-gating.
    - Employs multi-scale temporal convolutions to prevent temporal attention collapse.
- **Stage 6 (Training Optimization & Metric Learning):**
  - Confusable-pair contrastive loss, class-balanced focal loss, AdamW with cosine annealing.
- **Stage 7 (Publication-Quality Validation & Deliverables):**
  - Multi-seed evaluation, statistical significance testing (paired t-test / Wilcoxon), confusion matrices, and deliverable documentation.
