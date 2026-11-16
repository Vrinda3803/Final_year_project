# LIMITATIONS, FAILURE MODES & CLINICAL BOUNDARY ANALYSIS: STAGE 4

**Project:** Autism Therapy Physical Activity Recognition Using 2D Skeleton Dynamics  
**Framework:** Reliability-Conditioned Dynamic Bilateral Graph Learning (RCDB-STGT)  
**Date:** November 2026  
**Status:** Completed & Validated  

---

## 1. Executive Summary

A scientifically rigorous research paper must provide an honest, unvarnished appraisal of its limitations and failure modes. Rather than obscuring performance plateaus or discrepancies between model paradigms, this document details the mathematical, kinematic, and clinical boundaries of the proposed RCDB-STGT architecture on the MMASD 2D skeleton benchmark.

---

## 2. Failure Mode Analysis & Error Breakdown

### 2.1 The Severe Static Pose Skew in Fold 4
Across the 5-fold cross-validation split, the cohort assignments create an acute distributional anomaly in Fold 4:
- In Folds 0 through 3, rhythmic dynamic exercises (Drumming, Arm Spring, Marching) constitute 78% to 85% of samples.
- In **Fold 4**, static pose exercises (specifically Tree Pose and Standing without arm motion) account for **45.9%** of the held-out test cohort samples.

**Observed Empirical Impact:**
- In E2 (Fixed Bilateral Topology), Fold 0–3 accuracies range from 60.00% to 66.06%, while Fold 4 collapses to **38.89%**.
- In E5 (Full PR-STGT), Fold 4 drops to 54.44% while Fold 0 drops to 30.80%.

**Mechanistic Cause:**
Deep spatio-temporal graph models rely heavily on temporal coordinate velocity ($\Delta \mathbf{x}$) and rhythmic frequency modulations to distinguish exercise classes. During static poses (such as Tree Pose), coordinate velocities approach zero ($\|\mathbf{v}\| \to 0$). Under static equilibrium, OpenPose keypoint jitter introduces spurious high-frequency noise that mimics micro-tremors, misleading temporal attention heads.

### 2.2 Confusable Kinematic Exercise Twins
Forensic analysis of the out-of-fold error distributions reveals that misclassifications are concentrated in three specific confusable pairs:

```
+---------------------------------------------------------------------------------------------------+
|                                  CONFUSABLE EXERCISE TWIN MATRIX                                  |
+--------------------------+--------------------------+---------------------------------------------+
| Confusable Pair          | Primary Shared Kinematic | Discriminative Coordinate Feature           |
+--------------------------+--------------------------+---------------------------------------------+
| Drumming (dr) vs.        | Rhythmic upper-limb      | Torso posture: Standing upright in Clap vs. |
| Stand and Clap (sac)     | bilateral wrist motion   | slight forward torso tilt in Drumming       |
+--------------------------+--------------------------+---------------------------------------------+
| Marching in Place (ms)   | Alternating knee flexion | Global horizontal displacement (delta x, y) |
| vs. Marching Fwd/Side    | and ankle lift           | across the temporal sequence                |
+--------------------------+--------------------------+---------------------------------------------+
| Squats (sq) vs.          | Symmetrical hip and knee | Upper limb elevation: Wrists overhead in   |
| Arm Spring (as)          | flexion                  | Arm Spring vs. chest height in Squat        |
+--------------------------+--------------------------+---------------------------------------------+
```

When 2D joint detection confidence is low, the model defaults to the majority class (Drumming, $N=343$), causing asymmetric recall trade-offs.

### 2.3 2D Monocular Midline Occlusion & Keypoint Swapping
OpenPose BODY_25 operates on single-camera RGB video without depth ($z$-coordinate). When a child performs Drumming or Clapping:
1. Left and right wrists intersect near the anatomical sagittal plane (midline).
2. The bottom-up part affinity fields (PAFs) frequently swap joint 4 (RWrist) and joint 7 (LWrist) or merge them into a single coordinate.
3. This creates artificial velocity spikes ($\|\mathbf{v}\| > 15.0$ pixels/frame).
4. *Mitigation in RCDB-STGT:* The kinematic velocity plausibility prior successfully down-weights these spikes ($r_i(t) \to 0$), but information about the true hand position is lost during the occlusion period.

---

## 3. The Deep Graph vs. Hand-Crafted Feature Gap

A critical finding established across Stage 1, 2, and 3 is that the **RBF Kernel SVM baseline achieves 71.69% mean fold accuracy**, whereas our best deep graph architectures achieve 57% to 60%.

This performance difference is governed by three fundamental machine learning principles:

1. **Sample Efficiency and Inductive Bias on Small Clinical Datasets:**
   The MMASD dataset contains only 1,374 recordings. In each cross-validation fold, the training partition comprises approximately 900 clips.
   - The RBF Kernel SVM operates on **1,754 pre-engineered domain features** (robustly selected down to 800 via fold-local ANOVA). These features explicitly encode global sequence statistics: mean velocities, spectral power, joint distance percentiles, and prototype distances. With $N \approx 900$, an RBF kernel in a well-conditioned 800-dimensional space exhibits strong regularization and cannot easily overfit.
   - Deep graph models (RCDB-STGT, PR-STGT) must learn spatial projection weights, bilateral attention, temporal self-attention, and classification heads from raw coordinate tensors $(B, 64, 25, 4)$. Learning end-to-end representations from 900 samples is severely data-constrained.

2. **Sequence Length Downsampling:**
   Deep graph architectures truncate or resample raw video sequences to fixed $T=64$ frames to maintain tractable tensor operations and prevent memory overflow. In contrast, the engineered feature pipeline extracts summary statistics across the entire variable-length recording ($T$ up to 300+ frames), capturing long-term exercise repetitions that are truncated in deep models.

3. **Inference Latency Trade-Off:**
   While the RBF SVM achieves higher accuracy, its feature extraction pipeline requires computing 1,754 statistics across full-length videos, taking ~120 ms per sample. In contrast, RCDB-STGT processes fixed skeleton tensors in **3.9 ms on CPU**, making it suitable for real-time edge deployment.

---

## 4. Methodological & Cohort Boundaries

### 4.1 Cohort Group IDs vs. Biological Participant Identity
As established in the Stage 4 Dataset Audit:
- The 32 group IDs present in the MMASD distribution represent recording session prefixes (e.g., `40143`, `40023`), not biological participant identifiers.
- A single participant may have returned for multiple recording sessions across different days under different directory prefixes.
- **Scientific Claim Boundary:** Evaluations must strictly be reported as **group-independent cross-validation across 32 cohorts**, never as unverified "subject-independent" evaluation.

### 4.2 Demarcation from AV-ASD Dataset
- The project repository includes a separate dataset named AV-ASD comprising 666 video clips.
- AV-ASD focuses on **behavioral stereotypy detection** (hand flapping, head banging, body rocking), with no joint coordinate annotations.
- The two datasets have completely disjoint taxonomies, different sensor modalities, and cannot be combined or joint-trained.

### 4.3 Task Boundary: Activity Recognition vs. Autism Diagnosis
- **This research performs physical activity recognition during therapy sessions.**
- It classifies which exercise the child is performing (e.g., Drumming vs. Squatting) and monitors motor execution fidelity.
- **It does NOT perform automated autism diagnosis, clinical severity scoring, or behavioral screening.** Any clinical deployment must be situated as an assistive tool for physical therapists.

---

## 5. Statistical Power Boundaries (n = 5 Folds)

Standard 5-fold cross-validation yields $n = 5$ paired observations. As highlighted in statistical literature:
- A paired $t$-test with $n=5$ has approximately 20% to 30% statistical power to detect medium effect sizes ($d \approx 0.5$) at $\alpha = 0.05$.
- The non-parametric Wilcoxon signed-rank test with $n=5$ can only yield discrete $p$-values ($\{0.0625, 0.1250, 0.1875, \dots\}$) and mathematically **cannot achieve $p < 0.05$**.
- Therefore, all fold-level tests must be transparently reported as exploratory. Definitive statistical significance is established only through **sample-level McNemar testing ($N=1,374$)** and **10,000-iteration paired bootstrap confidence intervals**.
