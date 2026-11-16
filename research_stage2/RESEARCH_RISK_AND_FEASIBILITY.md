# Stage 2: Research Risk and Feasibility Assessment

**Project:** Autism Therapy Physical Activity Recognition using 2D Skeleton Dynamics  
**Date:** March 2026  
**Scope:** Risk identification, technical and computational feasibility, resource budgeting, clinical compliance, and fallback mitigation strategies for final-year BTech engineering execution.

---

## 1. Technical and Computational Feasibility

### 1.1 Compute and Hardware Feasibility
- **Environment:** Intel Core i7 / AMD Ryzen CPU (no local CUDA GPU required).
- **Execution Profile:** 
  - Model forward-backward pass: ~4.5 ms per sample.
  - One training epoch (1,099 training samples across 35 batches): ~3.8 seconds.
  - 40 epochs per fold: ~2.5 minutes on CPU.
  - Full 5-fold cross-validation: ~12 to 14 minutes total runtime.
- **Feasibility Verdict:** **100% Feasible.** Does not depend on cloud GPU allocations, external API quotas, or costly distributed training.

### 1.2 Memory and Storage Budget
- **Dataset in RAM:** The entire 1,374-recording 2D OpenPose dataset consumes ~185 MB in RAM as float32 NumPy arrays.
- **Model Size:** PR-STGT contains ~118,500 parameters ($\approx 474 \text{ KB}$ per checkpoint file).
- **Storage Profile:** Storing all fold checkpoints, epoch logs, out-of-fold CSVs, and confusion matrices for E2–E5 requires $<50 \text{ MB}$ of disk space.
- **Feasibility Verdict:** **Negligible resource footprint.** Well within any standard workstation constraints.

---

## 2. Dataset Constraints and Cohort Risks

### 2.1 Structural Cohort Imbalance
- **Risk:** Our forensic audit revealed that Subject IDs `41004` through `41143` (10 subjects) only perform Tree Pose (`tr`) and Twist Pose (`tw`). Subjects `20xxx` and `40xxx` perform all dynamic exercises.
- **Mitigation:**
  - GroupKFold partitioning ensures that subject groups are never split across train and validation sets, preventing identity leakage.
  - Metrics are reported using **Macro-F1** (giving equal weight to all 10 classes) rather than raw accuracy, ensuring that performance is not artificially inflated by majority static poses.

### 2.2 Monocular 2D OpenPose Noise and Tracking Dropouts
- **Risk:** Autistic children frequently turn sideways or move out of the camera field of view, causing OpenPose confidence to collapse to zero or hallucinate erratic joint jumps.
- **Mitigation:**
  - Single-joint velocity priors down-weight unphysical joint jumps ($>1.0 \text{ m/s}$).
  - Relational edge gating ($\tilde{\mathcal{R}}_{uv} = r_u \cdot r_v$) zeroes out inter-limb attention if either end-effector is occluded.
  - Frame-level temporal gating suppresses occluded frames during sequence pooling.

---

## 3. Clinical and Ethical Boundaries

### 3.1 Strict Separation Between Recognition and Diagnosis
- **Risk:** Naive machine learning projects in autism research mistakenly claim to "diagnose autism" or "grade therapy effectiveness" from simple skeleton sequences, which is medically invalid and unacceptable to clinical examiners.
- **Mitigation & Defense Position:**
  - The model is strictly framed as an **automated physical therapy activity recognition and logging tool** (assisting therapists in tracking exercise compliance and movement duration).
  - ADOS-2, CARS-2, and clinical metadata are treated solely as clinical diagnostic background context, not per-frame exercise quality ground truth.
  - Selective prediction explicitly routes low-confidence or highly occluded exercise clips to human therapists for review ("human-in-the-loop" clinical safety).

---

## 4. Comprehensive Risk Matrix and Fallback Strategy

| Identified Risk | Severity | Probability | Proposed Mitigation Strategy | Fallback / Contingency Plan |
|---|:---:|:---:|---|---|
| **Margin Loss ($\mathcal{L}_{\text{CPM}}$) Instability** | Moderate | Low | Use warm-up epochs (apply margin loss only after epoch 5) and tune margin scale $m=0.5, \lambda=0.25$. | If margin loss causes gradient oscillation, fallback to standard label-smoothed Cross-Entropy with Focal Loss. |
| **Relational Edge Overfitting** | Low | Low | Limit relational edges strictly to bilateral pairs (wrists and ankles); do not construct a fully dense graph. | Revert to standard physical graph with single-joint reliability (RA-STGT v1). |
| **Phase Velocity Noise in Jittery Clips** | Moderate | Low | Smooth raw coordinate velocities using a 3-frame moving average filter before computing phase sign. | Replace continuous tanh phase with thresholded discrete ternary phase $\{-1, 0, +1\}$. |
| **Examiner Questions on Statistical Significance** | Moderate | High | Compute paired Wilcoxon signed-rank and Student's t-tests across all 5 folds with exact p-values. | Defensible: full fold-wise logs and p-values are pre-calculated and documented in Stage 2 tables. |
| **Examiner Questions on Hand-Crafted Feature Baseline** | Moderate | High | Acknowledge the 11-class tabular feature ensemble (82.87% acc) as an upper bound on feature engineering, while demonstrating that PR-STGT is an end-to-end spatio-temporal deep learning model operating directly on raw coordinates. | Provide side-by-side comparison explaining the difference between tabular feature engineering and end-to-end representation learning. |

---

## 5. Summary and Execution Green Light

All technical, computational, and clinical prerequisites for the implementation and evaluation of **PR-STGT** are satisfied. The proposed architecture requires no proprietary software, runs smoothly on standard CPU hardware within 15 minutes per full validation cycle, and maintains absolute preservation of all historical experimental artifacts.
