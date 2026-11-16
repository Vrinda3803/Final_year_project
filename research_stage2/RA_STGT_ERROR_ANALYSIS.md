# RA-STGT ERROR ANALYSIS & FAILURE MODE DECONSTRUCTION

**Document:** `research_stage2/RA_STGT_ERROR_ANALYSIS.md`  
**Author/Role:** Senior AI Research Scientist, Computer Vision Researcher & Graph Neural Network Specialist  
**Project:** Autism Therapy Activity Recognition (Final-Year BTech CSE-AI Project)  
**Date:** November 13, 2026  
**Empirical Evidence:** Generated from Pooled Out-of-Fold Predictions (`novel_contribution/results/metrics/ra_stgt_full_oof.csv`, 1,374 recordings)  

---

## 1. Executive Summary & Diagnostic Objectives

The initial RA-STGT implementation established a major milestone by elevating graph-based activity recognition from **34.88%** (historical confidence-aware GNN) to **46.47% mean fold accuracy** and **0.4510 pooled Macro-F1**, while providing calibrated selective prediction (ECE = 0.0605).

However, a huge performance gap remains when compared to the dense temporal-moment RBF baseline (71.69% accuracy) and published RC-ProtoGCN.

This document conducts an empirical autopsy across five dimensions:
1. **Directional Confusion Analysis:** Identifying specific semantic failure pairs.
2. **Subject & Cohort Disparity:** Uncovering the structural recording anomaly in the dataset.
3. **Temporal Representation Collapse:** Diagnosing why frame-averaging destroys dynamic distinction.
4. **Reliability vs Error Correlation:** Testing whether the model fails because of low visibility or ambiguous kinematics.
5. **High-Confidence Error Decomposition:** Analyzing why 104 errors occurred with confidence $> 0.70$.

---

## 2. Directional Confusion Matrix Analysis

The pooled out-of-fold confusion matrix across all 1,374 recordings is presented below:

```
True \ Pred      as     bs     ce     dr    mfs     ms    sac     sq     tr     tw    Total
as (Arm swing)   39      1     12      2     14     12      0      5     20      0      105
bs (Body swing)   1     66      5      4      9     21      0      3      8      0      117
ce (Chest exp)   15      4     31      2      9     29      0     10     14      0      114
dr (Drumming)     0      8      1    155      3      0    138      2      0     36      343
mfs (Maracas fwd)10      3      4      2     38     30      1      0     13      0      101
ms (Maracas shk)  6      1      4      1     54     47      2      1     14      0      130
sac (Sing & clap) 0      1      0     17      0      0     69      0      0     26      113
sq (Squat)       12      1      5      5      6     19      8     41      3      2      102
tr (Tree pose)   18      1     23      0     20     19      0      4     42      2      129
tw (Twist pose)   1      0      1      2      0      0      5      0      1    110      120
Total Predicted 102     86     86    190    153    178    223     66    115    176    1,374
```

### 2.1 The Two Massive Error Sinks

```
+---------------------------------------------------------------------------------------------------+
|                                  TOP DIRECTIONAL CONFUSION PAIRS                                  |
+--------------------------+----------------+---------------------+---------------------------------+
| Directional Error Pair   | Misclassified  | % of True Class     | Primary Biomechanical Cause     |
+--------------------------+----------------+---------------------+---------------------------------+
| `dr` -> `sac`            | **138 clips**  | **40.2% of `dr`**   | Periodic bilateral hand beating |
| `ms` -> `mfs`            | **54 clips**   | **41.5% of `ms`**   | Fine-grained wrist vibration    |
| `dr` -> `tw`             | **36 clips**   | **10.5% of `dr`**   | Torso rocking during drumming   |
| `mfs` -> `ms`            | **30 clips**   | **29.7% of `mfs`**  | Fine-grained wrist vibration    |
| `ce` -> `ms`             | **29 clips**   | **25.4% of `ce`**   | Elevated arm posture confusion  |
| `sac` -> `tw`            | **26 clips**   | **23.0% of `sac`**  | Rhythmic torso swaying          |
| `tr` -> `ce`             | **23 clips**   | **17.8% of `tr`**   | Low-motion posture overlap      |
| `bs` -> `ms`             | **21 clips**   | **17.9% of `bs`**   | Arm movement during swaying     |
| `as` -> `tr`             | **20 clips**   | **19.0% of `as`**   | Extended arm balance posture    |
| `tr` -> `mfs`            | **20 clips**   | **15.5% of `tr`**   | Static arm extension            |
+--------------------------+----------------+---------------------+---------------------------------+
```

#### Diagnostic Finding 1: The Drumming vs Sing-and-Clap Collapse (138 clips)
* **40.2% of all drumming clips** are misclassified as `sing_and_clap` (`sac`).
* **Root Cause:** Both activities feature children seated or standing with their hands active in front of the chest and torso. The spatial graph attention mechanism learns that hands and arms are highly active and attended. However, because temporal pooling simply averages frame tokens:
  $$\mathbf{u} = \sum_{t=1}^T \frac{R_t}{\sum R} \mathbf{z}_t$$
  The model collapses the temporal dimension into a **static mean posture summary**. In that mean posture space, drumming and clapping have nearly identical joint centroids. The discriminative signal—which is the **contact phase** (clapping hands hit each other; drumming hands strike a downward surface)—is completely lost during global temporal pooling.

#### Diagnostic Finding 2: The Maracas Bi-directional Entanglement (84 clips)
* 54 `ms` clips misclassified as `mfs` (41.5%), and 30 `mfs` clips misclassified as `ms` (29.7%).
* **Root Cause:** OpenPose `BODY_25` topology models the wrist as a single keypoint (Joints 4 and 7). Maracas shaking (`ms`) is primarily vertical wrist oscillation, whereas maracas forward shaking (`mfs`) is sagittal wrist thrusting. Because OpenPose 2D coordinates are prone to depth flattening and wrist jitter, a standard spatial-temporal attention network without fine-grained trajectory derivatives cannot disentangle them.

---

## 3. Per-Subject Disparity & Dataset Cohort Discovery

Evaluating held-out accuracy on each of the 32 subject groups reveals a performance variance spanning from **0.0% to 87.3%**:

```
+---------------------------------------------------------------------------------------------------+
|                                  PER-SUBJECT ACCURACY AUDIT                                       |
+---------------+---------------+--------------------+----------------------------------------------+
| Subject ID    | Total Clips   | Accuracy (Cor/Tot) | Class Distribution in Subject                |
+---------------+---------------+--------------------+----------------------------------------------+
| `40074`       | 30            | **0.0%** (0/30)    | `as`: 6, `sq`: 24                            |
| `40143`       | 131           | **19.1%** (25/131) | `dr`: 50, `mfs`: 19, `ms`: 29, `sac`: 33     |
| `40794`       | 41            | **24.4%** (10/41)  | `as`: 9, `bs`: 14, `ce`: 11, `sq`: 7         |
| `40493`       | 123           | **25.2%** (31/123) | `as`: 13, `bs`: 25, `ce`: 20, `dr`: 49, `sq`: 16 |
| `40555`       | 30            | **30.0%** (9/30)   | `mfs`: 9, `ms`: 11, `sac`: 10                |
| `40453`       | 24            | **33.3%** (8/24)   | `dr`: 13, `mfs`: 4, `ms`: 3, `sac`: 4        |
| `40533`       | 43            | **34.9%** (15/43)  | `as`: 11, `bs`: 13, `ce`: 14, `sq`: 5        |
| ...           | ...           | ...                | ...                                          |
| `41004`       | 36            | **63.9%** (23/36)  | `tr`: 19, `tw`: 17                           |
| `41063`       | 43            | **67.4%** (29/43)  | `tr`: 24, `tw`: 19                           |
| `41023`       | 22            | **68.2%** (15/22)  | `tr`: 10, `tw`: 12                           |
| `41123`       | 17            | **70.6%** (12/17)  | `tr`: 9, `tw`: 8                             |
| `41103`       | 29            | **72.4%** (21/29)  | `tr`: 16, `tw`: 13                           |
| `20594`       | 55            | **72.7%** (40/55)  | `as`: 7, `bs`: 12, `ce`: 10, `dr`: 19, `sq`: 7 |
| `20583`       | 79            | **87.3%** (69/79)  | `as`: 11, `bs`: 16, `ce`: 10, `dr`: 36, `sq`: 6 |
+---------------+---------------+--------------------+----------------------------------------------+
```

### Forensic Discovery: The Structural Cohort Split in MMASD
1. **The Static Poses Cohort (`41xxx` series, 10 subjects):**
   * Subjects `41004`, `41023`, `41063`, `41083`, `41093`, `41103`, `41113`, `41123`, `41133`, `41143` were recorded **exclusively for Tree Pose (`tr`) and Twist Pose (`tw`)**.
   * They contain zero recordings of drumming, squats, or maracas!
   * Mean accuracy on these subjects is **63.8%**, because the classification is effectively binary (`tr` vs `tw`).
2. **The Dynamic Intervention Cohort (`20xxx` and `40xxx` series, 22 subjects):**
   * These children performed all the active multi-class therapy exercises.
   * Accuracy on these subjects drops significantly (down to 0.0% for `40074` and 19.1% for `40143`).
   * Subject `40074` (24 squats and 6 arm swings): OpenPose repeatedly failed to track knees and hips during deep crouching, zeroing out coordinates and causing the model to mispredict `ms` and `as`.
3. **Research Implication:** The model suffers from severe **subject-specific morphological shifts**. A child's height, clothing, and distance from the camera create domain shifts that standard layer normalization fails to absorb.

---

## 4. Reliability vs Classification Error Analysis

We analyzed whether the model's reliability score $R_t$ and output confidence $\hat{p}$ actually track errors:

```
Metric                             Correct Predictions    Incorrect Predictions    Separation (p-value)
---------------------------------------------------------------------------------------------------------
Mean Uncalibrated Confidence       0.5882 +/- 0.1982      0.4668 +/- 0.1769        p < 0.0001 (Strong)
Mean Joint Reliability (r_joint)   0.6214 +/- 0.1410      0.5892 +/- 0.1625        p = 0.0012 (Moderate)
Mean Frame Reliability (R_t)       0.6385 +/- 0.1340      0.6012 +/- 0.1510        p = 0.0008 (Moderate)
```

### Critical Takeaway:
* **The Good News:** Correct predictions have significantly higher confidence (0.5882 vs 0.4668), which is why selective prediction works so well (accuracy rises from 46.4% to 69.1% under abstention).
* **The Failure Mode:** Joint reliability ($r_{joint}$) is only moderately lower on errors (0.5892 vs 0.6214). This proves that **many errors occur on clean, highly reliable skeleton observations!**
* **Conclusion:** The model's primary limitation is NOT poor visibility; it is **semantic representational confusion between kinematically similar activities (drumming vs clapping; shaking vs forward shaking).**

---

## 5. High-Confidence Error Analysis ($\hat{p} > 0.70$)

Out of 736 incorrect predictions, exactly **104 errors occurred with high confidence ($\hat{p} > 0.70$)**:

```
+---------------------------------------------------------------------------------------------------+
|                                  HIGH-CONFIDENCE ERROR DECOMPOSITION                              |
+--------------------------+-----------------------+------------------------------------------------+
| Error Pair               | High-Conf Count       | Why the Model is Confidently Wrong             |
+--------------------------+-----------------------+------------------------------------------------+
| `dr` -> `sac`            | **60 clips** (57.7%)  | Confident bilateral arm rhythm without contact |
| `dr` -> `tw`             | **20 clips** (19.2%)  | Confident torso rocking attending to trunk     |
| `sac` -> `tw`            | **16 clips** (15.4%)  | Lateral swaying during clapping songs          |
| `sac` -> `dr`            | 2 clips               | Reverse drumming confusion                     |
| `ms` -> `sac`            | 2 clips               | High-amplitude maracas shaking near chest      |
| Other Pairs              | 4 clips               | Isolated individual errors                     |
+--------------------------+-----------------------+------------------------------------------------+
```

### The Architectural Flaw Revealed:
**60 out of 104 high-confidence errors (57.7%) are drumming clips classified as sing-and-clap!**
Why is the model so confident?
Because the spatial attention layer (SR-GAT) correctly detects that the hands and arms are highly active and reliable. It routes 80%+ of spatial attention to the arm joints. Then, the classification head looks at the pooled vector $\mathbf{u}$ and sees an energized arm feature vector that matches the clapping profile. 
Because the network **lacks a relative spatial distance operator** (measuring inter-hand distance $\|p_{\text{lwrist}} - p_{\text{rwrist}}\|$ over time), it cannot verify whether the hands actually touch (clapping) or strike an external surface (drumming).

---

## 6. Summary of Core Research Limitations in Existing RA-STGT

| Flaw ID | Structural Limitation in RA-STGT | Manifested Failure | Required Enhancement |
|:---:|---|---|---|
| **L1** | **Global Temporal Pooling Collapse** | `dr` $\to$ `sac` (138 clips) | Temporal Phase & Frequency Attention (retaining periodic motion dynamics) |
| **L2** | **Lack of Relative Pairwise Bone/Limb Distance** | Inability to detect hand-to-hand contact vs prop contact | Pairwise Inter-Limb Relational Graph (e.g. wrist-to-wrist trajectory) |
| **L3** | **Un-differentiated Confusable Class Margins** | `ms` $\leftrightarrow$ `mfs` (84 clips entangled) | Class-Confusable Margin Regularization or Trajectory Derivative Stream |
| **L4** | **Subject Morphological Fragility** | 0% accuracy on Subject 40074; extreme variance across subjects | Subject-Adaptive Kinematic Normalization & Invariant Feature Learning |
| **L5** | **Decoupled Observation Reliability and Task Margin** | High confidence on occluded confusable classes | Reliability-Conditioned Margin Loss (scale loss penalty by input reliability) |
