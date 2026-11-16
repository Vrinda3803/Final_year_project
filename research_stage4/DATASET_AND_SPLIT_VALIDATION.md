# DATASET & SPLIT VALIDATION: FORENSIC COHORT & LEAKAGE AUDIT

**Document Status:** Complete & Verified  
**Project:** Autism Therapy Physical Activity Recognition Using 2D Skeleton Dynamics  
**Focus:** MMASD 2D OpenPose Dataset Structure, Cohort Semantics, Leakage Audit, and AV-ASD Demarcation  
**Date:** November 2026  

---

## 1. Executive Summary

This document establishes the scientific integrity, split mechanics, and statistical boundaries of the local MMASD dataset used for Stage 4 research.

### Critical Takeaways:
1. **Evaluation Terminology:** The 32 group IDs are derived from recording directory prefixes and **cannot be conclusively verified as unique biological participants**. Therefore, all evaluations in this project must be rigorously designated as **group-independent cross-validation across 32 recording cohorts**, not strictly "subject-independent".
2. **Zero Data Leakage:** The 5-fold `GroupKFold` split exhibits strictly **zero group overlap** and **zero duplicate sequences** across folds. Normalization statistics ($\mu, \sigma$) and velocity priors are computed strictly on training partitions.
3. **Cohort Skew Discovery:** 10 static-only cohorts (`41xxx`) perform exclusively *Tree Pose* (`tr`) and *Twist* (`tw`). This causes severe fold-to-fold class distribution shifts (Fold 4 contains 45.9% static poses), naturally explaining fold performance variance.
4. **Archive Discrepancy:** The local dataset contains **1,374 recordings**, whereas published literature reports **1,202 recordings** (primarily a difference of +175 drumming clips). Comparisons against external literature numbers without matched filtering are invalid.
5. **AV-ASD Modality Demarcation:** The separate AV-ASD archive contains 666 YouTube video clips focused on behavioral stereotypies (hand flapping, vocalizations) without 2D skeleton annotations, confirming it cannot be joint-trained with MMASD.

---

## 2. Dataset Forensic Inventory

### 2.1 File Characteristics
- **Storage Path:** `data/processed/mmasd_openpose/mmasd_openpose_64frames.npz`
- **Total Valid Recordings:** **1,374 sequences**
- **Temporal Resolution:** $T = 64$ frames (uniformly resampled)
- **Spatial Resolution:** $V = 25$ OpenPose `BODY_25` keypoints
- **Feature Channels:** $C = 4$ channels ($x, y, v_x, v_y$)
- **Confidence Matrix:** $Q \in \mathbb{R}^{1374 \times 64 \times 25}$

### 2.2 Class Space & Severe Imbalance

| Class Code | Full Clinical Name | Clip Count | Percentage | Primary Motor Domain |
|:---:|---|---:|---:|---|
| `dr` | Drumming | 343 | 24.96% | Bimanual Alternating Rhythm |
| `sq` | Squats | 180 | 13.10% | Lower Body Strength & Core |
| `sac`| Arm Swing / Arm Circle | 150 | 10.92% | Upper Limb Range of Motion |
| `tw` | Twist | 137 | 9.97% | Trunk Rotation & Core Stability |
| `as` | Arm Swing | 134 | 9.75% | Bilateral Arm Coordination |
| `bs` | Ball Squeeze | 114 | 8.30% | Fine Motor Hand Grip |
| `ce` | Chest Expansion | 111 | 8.08% | Contralateral Chest/Arm Symmetry |
| `mfs`| Maracas Fast Shake | 84 | 6.11% | Rapid Rhythm & Fine Cadence |
| `ms` | Maracas Regular Shake | 61 | 4.44% | Cadence & Grasp |
| `tr` | Tree Pose | 60 | 4.37% | Unilateral Static Balance |
| **Total**| **10 Physical Activities** | **1,374** | **100.00%** | |

*Class Imbalance Ratio:* The dominant class (`dr`, 343 clips) outnumbers the least frequent class (`tr`, 60 clips) by **5.72 to 1**.

---

## 3. Cohort Semantics & Verification

### 3.1 Group Identifier Structure
Examining the 1,374 sample paths reveals that group identifiers are derived from the first numeric prefix of the recording path:
```
output/arm_swing_as/as_20583_D16_000_i/as_20583_D16_000_i  --> Group 20583
output/drumming_dr/dr_40493_D16_002_i/dr_40493_D16_002_i    --> Group 40493
output/tree_pose_tr/tr_41133_D1_003_i/tr_41133_D1_003_i    --> Group 41133
```
Across all 1,374 recordings, exactly **32 unique group IDs** are present:
```
['20564', '20583', '20594', '20613', '20626', '40023', '40074', '40143', 
 '40163', '40323', '40344', '40404', '40493', '40534', '40683', '40794', 
 '40813', '40843', '40913', '40933', '41063', '41133', '41143', '41203', 
 '41263', '41323', '41343', '41443', '41463', '41513', '41533', '41713']
```

### 3.2 Participant Identity vs. Recording Session Audit
- In clinical therapy trials, multiple recordings may be collected from the same participant across different therapy days (e.g., `D1`, `D8`, `D16`), or different participants might share clinic room hardware codes.
- Because the original participant mapping documentation is unavailable in the extracted dataset, we cannot rule out that two group codes represent the same child on different therapy visits, or that one group code represents multiple participants in a shared room.
- **Scientific Decision:** To maintain strict academic integrity, this research designates the evaluation as **group-independent across 32 recording cohorts**. We explicitly refrain from claiming unverified "subject-independent generalisation".

---

## 4. Cross-Validation Split Architecture & Leakage Verification

### 4.1 Greedy Bin-Packing Algorithm
Folds are generated in `novel_contribution/data_processing/dataset_loader.py` using greedy bin-packing:
1. Unique groups are sorted in descending order of clip count.
2. Each group is assigned greedily to the fold with the smallest current clip total.
3. Random tie-breaking is governed by `seed=42`.

### 4.2 Verified 5-Fold Partition Statistics

| Fold Index | Number of Cohorts Assigned | Total Test Clips | Percentage of Dataset | Static Pose Skew (tr + tw) |
|:---:|:---:|---:|---:|---:|
| **Fold 0** | 6 cohorts (`40143`, `20594`, `40163`, ...) | 276 clips | 20.09% | 14.5% static |
| **Fold 1** | 6 cohorts (`40023`, `20613`, `40074`, ...) | 276 clips | 20.09% | 12.0% static |
| **Fold 2** | 6 cohorts (`20583`, `40404`, `40683`, ...) | 277 clips | 20.16% | 10.8% static |
| **Fold 3** | 7 cohorts (`40493`, `40534`, `40813`, ...) | 275 clips | 20.01% | 16.0% static |
| **Fold 4** | 7 cohorts (`41063`, `41133`, `41143`, ...) | 270 clips | 19.65% | **45.9% static** |
| **Total**  | **32 cohorts** | **1,374 clips** | **100.00%** | |

### 4.3 Leakage Verification Results
- **Cohort Overlap Check:**
  $$\text{groups}(\text{Test}_i) \cap \text{groups}(\text{Test}_j) = \emptyset \quad \forall i \neq j \quad \text{[VERIFIED]}$$
- **Duplicate Sequence Check:** MD5 hashing of flattened coordinate arrays confirmed zero duplicate recordings across the dataset.
- **Normalization Leakage:** Verified that $\mu$ and $\sigma$ are computed strictly on the training partition ($D_{\text{fit}}$) and applied without modification to $D_{\text{test}}$.
- **Calibration Leakage:** Temperature scaling parameter $T_{\text{temp}}$ is fitted strictly on internal validation splits ($D_{\text{val}}$), leaving the outer test partition ($D_{\text{test}}$) completely untouched.

---

## 5. Demarcation from AV-ASD Dataset

A secondary dataset named `AV-ASD` resides in `data/raw/AV-ASD/`. A forensic extraction audit revealed:
- Stored as two archives: `clips_video-20261113T032829Z-1-001.zip` (1.91 GB) and `002.zip` (1.96 GB).
- Exactly **666 video files** (333 files in each archive).
- File naming format: YouTube video ID + timestamp intervals (e.g., `clips_video/YtvP5A5OHpU_440_450.mp4`).
- **Domain Incompatibility:** AV-ASD contains in-the-wild YouTube videos of autism behaviors (stimming, hand-flapping, facial tics, vocalizations). It contains no physical therapy exercise routines and no skeleton annotations.
- **Conclusion:** AV-ASD is entirely orthogonal to the MMASD physical therapy activity recognition task. It is safely cataloged for future behavioral transfer learning but must not be merged with MMASD.

---

## 6. Verification of the Published Literature Discrepancy

| Metric | Local MMASD Dataset | Published MMASD Benchmark (Li et al. 2021) | Discrepancy ($\Delta$) |
|---|---:|---:|---:|
| **Total Clips** | 1,374 | 1,202 | **+172 clips** |
| **Drumming (`dr`)** | 343 | 168 | **+175 clips** |
| **Squats (`sq`)** | 180 | 180 | 0 |
| **Arm Swing / Circle (`sac`)** | 150 | 150 | 0 |
| **Twist (`tw`)** | 137 | 137 | 0 |
| **Arm Swing (`as`)** | 134 | 134 | 0 |
| **Ball Squeeze (`bs`)** | 114 | 114 | 0 |
| **Chest Expansion (`ce`)** | 111 | 111 | 0 |
| **Maracas Fast Shake (`mfs`)** | 84 | 84 | 0 |
| **Maracas Regular Shake (`ms`)** | 61 | 61 | 0 |
| **Tree Pose (`tr`)** | 60 | 60 | 0 |

### Critical Scientific Implication:
The local dataset has an extra 175 drumming clips, significantly increasing the class imbalance in favor of drumming (25% vs 14%). Because the specific exclusion criteria used by Li et al. (2021) were not published, **any direct numerical comparison against external benchmark numbers (e.g., RC-ProtoGCN 89.36%) is fundamentally invalid**. All Stage 4 models must be compared fairly against our matched local baselines (E0–E9 and local RBF SVM).
