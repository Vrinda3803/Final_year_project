# COMPREHENSIVE EXPERIMENT ROADMAP
## Research Improvement & Methodological Progression

This document defines the 8-stage roadmap to establish publication-grade results for Autism Therapy Physical Activity Recognition using Skeleton Dynamics.

---

### Stage 1: Baseline Reproduction & Experimental Rigor Verification
- **Objective:** Reproduce the key baseline checkpoints under the exact 5-fold grouped CV protocol and confirm metric matching.
- **Targets:**
  - RBF Kernel-Ridge Baseline: ~71.69% accuracy, 0.6394 Macro-F1.
  - Temporal MLP Baseline: ~70.17% accuracy, 0.6338 Macro-F1.
  - RCDB-STGT A8 / A1 Baseline: ~65.92% accuracy, 0.6402 Macro-F1.
  - RCDB-STGT A0 (Fixed Topo): ~57.12% accuracy.
- **Deliverables:** `research_improvements/BASELINE_REPRODUCTION.md` and verification scripts.

---

### Stage 2: Deep Dataset Audit & Leakage Investigation
- **Objective:** Perform a complete audit of the 1,374 recordings across 10 classes and 32 recording groups.
- **Analyses:**
  - Class distribution (identifying severe imbalance, e.g. class 3 `dr` comprising 25%).
  - Group distribution across folds.
  - OpenPose confidence distribution ($Q$) and missing joint patterns.
  - Pairwise class motion similarity and confusable classes.
  - Duplicate/near-duplicate sequence detection.
- **Deliverables:** `research_improvements/DATASET_AUDIT.md`.

---

### Stage 3: Feature Engineering & Invariance Representations
- **Objective:** Address coordinate instability and camera shift by introducing body-centric and kinetic invariants.
- **Feature Representations to Evaluate:**
  - $F_0$: Raw coordinates $(x, y, v_x, v_y, q)$.
  - $F_1$: Body-centered coordinates (normalized relative to mid-hip or neck/spine).
  - $F_2$: Torso-scale normalized coordinates (distance normalized by torso length).
  - $F_3$: Structural bone vectors (relative direction vectors along anatomical segments).
  - $F_4$: Anatomical joint angles and angular velocity (elbow, shoulder, knee, hip angles).
  - $F_5$: Bilateral coordination metrics (bilateral velocity discrepancy, left-right phase correlation).
  - $F_6$: Multi-stream concatenation.
- **Deliverables:** `research_improvements/FEATURE_EXPERIMENTS.md`.

---

### Stage 4: Strong Non-Graph & Temporal Baselines
- **Objective:** Establish the true empirical ceiling on the enriched feature set before complex GNN modeling.
- **Models:**
  - Tuned RBF Kernel-Ridge.
  - Multi-Layer Perceptron (MLP) with modern regularizers.
  - Random Forest & HistGradientBoosting.
  - Multi-scale Temporal Convolutional Network (MS-TCN).
- **Deliverables:** Benchmarks documented in `research_improvements/ARCHITECTURE_COMPARISON.md`.

---

### Stage 5: Novel Research Architecture — Reliability-Conditioned Bilateral GCN / TCN
- **Objective:** Refine our novel contribution to prevent temporal attention collapse and destructive zero-gating.
- **Key Enhancements:**
  - Dynamic bilateral connection conditioned smoothly on reliability.
  - Multi-scale Dilated Temporal Convolution replacing raw temporal self-attention.
  - Adaptive Spatial Graph Convolution with learned bilateral coordination edge streams.
- **Deliverables:** Controlled ablations in `research_improvements/ABLATION_STUDY.md`.

---

### Stage 6: Confusable-Class Targeted Metric Learning
- **Objective:** Resolve the primary error source (e.g., `as` vs `ce` vs `dr`, `sq` vs `ms`).
- **Methods:**
  - Supervised contrastive loss on confusable pairs.
  - Class-weighted focal loss.
  - Label smoothing and mixup tailored for skeleton sequences.
- **Deliverables:** `research_improvements/HYPERPARAMETER_SEARCH.md`.

---

### Stage 7: Multi-Seed Final Evaluation & Statistical Validation
- **Objective:** Multi-seed (3-5 seeds) verification of top candidates across all 5 folds.
- **Metrics:**
  - Out-of-fold accuracy, Macro-F1, per-class F1, Expected Calibration Error (ECE).
  - Paired t-test and Wilcoxon signed-rank test against all baselines.
- **Deliverables:** `research_improvements/FINAL_RESULTS.md`.

---

### Stage 8: Publication-Quality Documentation
- **Deliverables:** `research_improvements/PUBLICATION_READINESS.md`, master results CSV, and publication figures.
