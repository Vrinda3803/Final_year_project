# DATASET AUDIT REPORT: MMASD OPENPOSE SKELETON COHORT

**Date:** November 18, 2026  
**Dataset Path:** `data/processed/mmasd_openpose/mmasd_openpose_64frames.npz`  
**Evaluation Standard:** 5-Fold Grouped Cross-Validation  

---

## 1. Dataset Dimensions & Volume
- **Total Samples ($N$):** 1,374 recordings.
- **Sequence Length ($T$):** 64 uniform frames per recording.
- **Joints ($V$):** 25 OpenPose BODY-25 joints.
- **Channels ($C$):** 4 $(x, y, v_x, v_y)$ with OpenPose confidence scores $Q \in \mathbb{R}^{1374 \times 64 \times 25}$.
- **Cohort Group Identifiers ($G$):** 32 distinct recording groups.

---

## 2. Class Distribution & Imbalance
The dataset contains 10 target therapy activities with significant class imbalance:
| Class Index | Class Code | Description | Sample Count | Percentage |
|:---:|:---:|:---|:---:|:---:|
| 0 | `as` | Arm Stimming / Flapping | 105 | 7.64% |
| 1 | `bs` | Body Swaying | 117 | 8.52% |
| 2 | `ce` | Clapping Exercises | 114 | 8.30% |
| 3 | `dr` | Drumming Exercises | **343** | **24.96%** |
| 4 | `mfs` | Motor Finger Stimming | 101 | 7.35% |
| 5 | `ms` | Marching / Stepping | 130 | 9.46% |
| 6 | `sac` | Standing Arm Circles | 113 | 8.22% |
| 7 | `sq` | Squats / Squatting | 102 | 7.42% |
| 8 | `tr` | Torso Rotation / Twisting | 129 | 9.39% |
| 9 | `tw` | Toe Walking | 120 | 8.73% |

**Key Imbalance Finding:** Drumming (`dr`) comprises nearly **25%** of the entire dataset, while classes like `mfs` (7.35%) and `sq` (7.42%) have fewer than 105 instances. Class-weighted cross-entropy or focal loss is required to prevent majority-class bias.

---

## 3. Motion Dynamics & Intensity Across Classes
Mean Euclidean joint displacement magnitude $\|\mathbf{v}\|$ across all joints and frames:
- **High-Intensity Gross Motor Activities:**
  - `sac` (Standing Arm Circles): **0.399**
  - `ms` (Marching / Stepping): **0.348**
  - `sq` (Squats): **0.326**
  - `tw` (Toe Walking): **0.292**
- **Low-Intensity / Fine Motor Activities:**
  - `ce` (Clapping): **0.219**
  - `mfs` (Motor Finger Stimming): **0.235**
  - `bs` (Body Swaying): **0.243**
  - `tr` (Torso Rotation): **0.247**
  - `as` (Arm Stimming): **0.253**
  - `dr` (Drumming): **0.261**

**Key Finding:** Several fine motor activities (`as`, `ce`, `dr`, `mfs`) share nearly identical aggregate velocity profiles (~0.22 - 0.26), meaning purely global temporal pooling without localized joint-level relative kinematics is insufficient to distinguish them.

---

## 4. Missing Joint and Quality Analysis ($Q$)
- **Overall Mean Joint Confidence:** $0.641$
- **Fraction of Completely Zeroed / Missing Joints ($Q = 0$):** $6.48\%$
- **Worst 5 Occluded / Degraded Joints:**
  1. Joint 17 (Right Ear): **34.43%** missing frames.
  2. Joint 18 (Left Ear): **23.57%** missing frames.
  3. Joint 20 (Left Small Toe): **15.25%** missing frames.
  4. Joint 23 (Right Small Toe): **12.71%** missing frames.
  5. Joint 21 (Left Heel): **11.34%** missing frames.

**Key Finding for Bilateral Graph Modeling:** Head and peripheral foot joints experience significant missingness. Crucially, the core bilateral coordination pairs (Wrists: 4 & 7, Elbows: 3 & 6, Knees: 10 & 13) have high reliability (>88% detection rate). A hard multiplicative zero-gate hurts wrists during fast movement, while soft reliability conditioning preserves signal.

---

## 5. Duplicate and Leakage Integrity
- No duplicate sequence was detected across recordings. Minimum pairwise feature distance between random clips is positive ($>0.08$).
- Grouped 5-fold cross-validation strictly partitions the 32 recording groups:
  - Fold 0: 276 test clips
  - Fold 1: 276 test clips
  - Fold 2: 277 test clips
  - Fold 3: 275 test clips
  - Fold 4: 270 test clips
- All group sets are disjoint: $\text{Test}_i \cap \text{Test}_j = \emptyset$.
