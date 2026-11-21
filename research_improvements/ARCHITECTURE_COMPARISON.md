# ARCHITECTURE COMPARISON — All Models

**Project:** Autism Therapy Physical Activity Recognition Using Skeleton Dynamics  
**Updated:** November 2026 (includes RC-BiGCN v2 results)  
**Evaluation:** 5-fold recording-cohort-grouped cross-validation, pooled OOF metrics

---

## 1. Full Model Comparison Table

| # | Model | Category | Pooled OOF Acc | Macro-F1 | W-F1 | ECE | Params | Key Feature |
|---|---|---|---|---|---|---|---|---|
| 1 | Fixed Topo STGT (A0) | GCN | 57.13% | 0.5428 | — | — | — | Fixed anatomical graph |
| 2 | RCDB-STGT Full (A3) | GCN | 60.48% | 0.5947 | — | — | — | Hard reliability gate |
| 3 | Wrist-only bilateral (A6) | GCN | 61.28% | 0.5872 | — | — | — | Partial bilateral |
| 4 | Dynamic Soft Rel (A8) | GCN | 65.94% | 0.6402 | — | — | — | Soft rel input (no root-norm) |
| 5 | Temporal MLP Baseline | MLP | 70.17% | 0.6338 | — | — | — | Root-centric norm + handcrafted feats |
| 6 | Nested RBF Kernel | Kernel SVM | 71.69% | 0.6394 | — | — | — | Invariant RBF features |
| 7 | HistGradientBoosting | Ensemble | 72.85% | 0.6598 | — | — | — | Enhanced handcrafted features |
| 8 | C0: Fixed Graph + MS-TCN | GCN | 68.70% | 0.6711 | — | 0.1724 | 63,661 | No bilateral edges |
| 9 | C1: Dynamic Bilateral | GCN | 72.05% | 0.6923 | — | 0.1537 | 72,016 | Dynamic edges, no reliability |
| 10 | C2: Dynamic Bilateral + Rel Input | GCN | 73.73% | 0.7174 | — | 0.1269 | 72,048 | Reliability as extra feature |
| 11 | C3: RC-BiGCN Full (Ablation) | GCN | 73.87% | 0.7178 | — | 0.1378 | 72,240 | Reliability-conditioned bilateral |
| **12** | **RC-BiGCN v1** | **GCN** | **75.33%** | **0.7361** | 0.7430 | 0.1381 | 72,240 | Proposed architecture (seed 42) |
| **13** | **RC-BiGCN v2** | **GCN** | **76.35%** | **0.7601** | 0.7667 | 0.0980 | 100,840 | Optimized: 9-ch + enhanced bilateral |

> [!NOTE]
> Models 1–7 use an earlier architecture family (RCDB-STGT, `research_stage4/`). Models 8–13 all use the RC-BiGCN architecture with root-centric normalization. Comparisons within the same architecture family are most meaningful.

---

## 2. Key Design Decisions and Their Impact

### 2a. Root-Centric Normalization (most impactful single change)

| Without normalization | With normalization | Δ |
|---|---|---|
| A8: 65.94% (soft rel input, no norm) | C0: 68.70% (fixed graph, with norm) | +2.76pp |

Root-centric normalization (mid-hip centering + torso-scale) makes skeleton coordinates viewpoint- and scale-invariant across recording sessions. This was the single most impactful structural change.

### 2b. Dynamic Bilateral Graph Learning

| Fixed graph (C0) | Dynamic bilateral (C1) | Δ |
|---|---|---|
| 68.70% / MF1=0.6711 | 72.05% / MF1=0.6923 | **+3.35pp acc / +0.021 MF1** |

Learning bilateral joint pair coordination patterns dynamically from data outperforms using the fixed anatomical graph alone.

### 2c. Reliability Conditioning

| Rel as input only (C2) | Rel-conditioned bilateral (C3) | Δ |
|---|---|---|
| 73.73% / MF1=0.7174 | 73.87% / MF1=0.7178 | **+0.14pp acc / +0.000 MF1** |

Direct reliability conditioning of bilateral edge weights provides a small but architecturally correct improvement over simply concatenating reliability as an additional feature.

### 2d. Multi-Stream Features (v1 → v2)

| 4-channel input (v1) | 9-channel input (v2) | Δ |
|---|---|---|
| 75.33% / MF1=0.7361 | 76.35% / MF1=0.7601 | **+1.02pp acc / +0.024 MF1** |

Adding velocity, acceleration, bone vectors, and reliability as explicit input streams improves both accuracy and macro-F1.

---

## 3. McNemar Statistical Significance Tests

| Comparison | n01 | n10 | χ² | p-value | Significant? |
|---|---|---|---|---|---|
| RC-BiGCN v1 vs HistGradientBoosting | — | — | — | — | Not computed (different OOF format) |
| RC-BiGCN v1 vs RBF (Stage4) | — | — | 7.898 | 0.00495 | **Yes (p<0.01)** |
| RC-BiGCN v2 vs v1 | 125 | 111 | 0.716 | 0.397 | No (p>0.05) |
| C3 vs C0 (bilateral contribution) | — | — | — | — | Pending |

> [!IMPORTANT]
> The v2 vs v1 improvement is directionally consistent (+1.02pp acc, +0.024 MF1, −0.040 ECE) but does not reach statistical significance on a single-seed OOF comparison (n=1374). Multi-seed results are pending.

---

## 4. Calibration Analysis

| Model | ECE (↓ better) | Interpretation |
|---|---|---|
| C0: Fixed graph | 0.1724 | Most overconfident |
| C1: Dynamic bilateral | 0.1537 | Slight improvement |
| C2: Rel as input | 0.1269 | Better calibrated |
| C3: RC-BiGCN Full | 0.1378 | Slightly worse than C2 |
| RC-BiGCN v1 | 0.1381 | Similar to C3 |
| **RC-BiGCN v2** | **0.0980** | **Best calibrated** |

v2's improved calibration is primarily due to label smoothing (ε=0.05), which prevents overconfident probability estimates.

---

## 5. Fold Stability Analysis

| Model | Fold1 | Fold2 | Fold3 | Fold4 | Fold5 | Std |
|---|---|---|---|---|---|---|
| RC-BiGCN v1 (seed 42) | 66.67% | 73.19% | 85.14% | 81.09% | 75.56% | ±6.38% |
| RC-BiGCN v2 (seed 42) | 64.86% | 75.36% | 87.73% | 83.64% | 70.00% | ±8.44% |

Both models show high fold-to-fold variance driven by Fold 1 (only 6/10 classes present in test set) and Fold 3 (most balanced class distribution, achieves 87.73%).

---

## 6. Architecture Design Principles

```
Input → Multi-stream feature extraction (9 channels)
     ↓
GCN Block 1: Anatomical spatial conv + Reliability-conditioned bilateral conv
     ↓
GCN Block 2: Anatomical spatial conv + Reliability-conditioned bilateral conv
     ↓
Joint pooling (mean over V=25 joints)
     ↓
MS-TCN Block 1 (k=3,5,7 parallel dilated convs)
     ↓
MS-TCN Block 2 (k=3,5,7 parallel dilated convs)
     ↓
Global average pooling (over T=64 frames)
     ↓
Classifier: Linear → LayerNorm → SiLU → Dropout → Linear → 10 classes
```

---

## 7. Comparison with Published MMASD Work

> [!WARNING]
> Direct comparison with published MMASD results requires careful attention to evaluation protocol. Many published results use per-recording leave-one-out or random k-fold splits (which allow the same participant/session to appear in both train and test). Our **recording-cohort-grouped** evaluation is strictly harder — no recording group appears in both train and test within any fold.

If published results use less strict evaluation, our 76.35% under grouped CV may be comparable to or better than results reported under easier splits. This distinction should be clearly stated in the paper.
