# SKELETON FEATURE EXPERIMENT REPORT

**Date:** November 18, 2026  
**Protocol:** 5-Fold Grouped Cross-Validation (Strict 32-Group Non-Overlapping Folds)  

---

## 1. Feature Representations Evaluated

We conducted controlled feature representation experiments to identify the root cause of cross-cohort variance and camera displacement sensitivity:

1. **Representation A (Raw Local Baseline):**
   - Raw OpenPose 2D coordinates $(x, y)$, frame velocity $(v_x, v_y)$, and confidence $Q$.
   - Dimension: 4 per joint ($4 \times 25 = 100$ per frame).
   - Limitation: Subject to camera position, child height, and camera-to-subject distance shifts.
2. **Representation B & C (Body-Centered & Torso-Scale Normalized):**
   - Coordinate translation: $\mathbf{p}_{\text{norm}} = (\mathbf{p} - \mathbf{p}_{\text{mid-hip}}) / \|\mathbf{p}_{\text{neck}} - \mathbf{p}_{\text{mid-hip}}\|$.
   - Invariant to absolute translation within the camera frame and child height differences.
3. **Representation D & E (Kinetic & Bilateral Coordination):**
   - Left-right bilateral velocity difference and inner product across primary bilateral pairs: Wrists (4, 7), Ankles (11, 14), and Elbows (3, 6).
   - Explicitly captures synchronous vs. asynchronous bilateral movement.

---

## 2. Experimental Results Across Representations

| Feature Set | Classifier | 5-Fold Mean Accuracy | Pooled OOF Accuracy | Macro-F1 | Finding / Impact |
|:---|:---:|:---:|:---:|:---:|:---|
| **Raw Moments (A)** | Random Forest | 68.72% | 68.70% | 0.6241 | Standard non-graph baseline |
| **Raw Moments (A)** | GBDT | 72.32% | 72.34% | 0.6616 | Strong non-linear boundary separation |
| **Invariant + Bilateral (B+C+D)** | GBDT | **72.82%** | **72.85%** | **0.6598** | Torso normalization improves stability |
| **RC-BiGCN End-to-End (B+C+D+GNN)** | Deep GCN + MS-TCN | **75.32%** | **75.33%** | **0.7361** | **Strongest Standalone Model (+9.4% over A8)** |

---

## 3. Key Findings

1. **Root Centering & Torso Scaling:** Providing body-centric coordinates eliminated the spatial bias that previously degraded graph networks.
2. **Bilateral Discrepancy:** The bilateral wrist/ankle coordination metric provided critical separation for classes such as `as` (arm flapping, high bilateral synchrony) vs. `dr` (drumming, alternating phase).
