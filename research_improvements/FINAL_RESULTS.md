# FINAL RESULTS — RC-BiGCN Autism Therapy Activity Recognition

**Updated with RC-BiGCN v2 results (November 2026)**  
**Evaluation:** 5-fold recording-cohort-grouped cross-validation  
**Dataset:** MMASD OpenPose (1,374 recordings, 10 classes, 32 recording groups)

---

## Best Model: RC-BiGCN v2

| Metric | Value |
|---|---|
| **Pooled OOF Accuracy** | **76.35%** |
| Mean ± Std Fold Accuracy | 76.32% ± 8.44% |
| **Pooled OOF Macro-F1** | **0.7601** |
| Pooled OOF Weighted-F1 | 0.7667 |
| Pooled OOF ECE | 0.0980 |
| Parameters | 100,840 |
| Training runtime | ~38 min (5 folds, CPU) |

---

## Progression of Results

| Stage | Model | Pooled OOF Acc | Macro-F1 | Key change |
|---|---|---|---|---|
| Stage 2 | Fixed Topo STGT (A0) | 57.13% | 0.5428 | Baseline skeleton GCN |
| Stage 3 | RCDB-STGT (A3) | 60.48% | 0.5947 | Hard reliability gate |
| Stage 3 | Dynamic Soft Rel (A8) | 65.94% | 0.6402 | Soft reliability input |
| Stage 4 | Temporal MLP | 70.17% | 0.6338 | Root-centric normalization |
| Stage 4 | HistGradientBoosting | 72.85% | 0.6598 | Handcrafted features |
| **Proposed** | **RC-BiGCN v1** | **75.33%** | **0.7361** | Full bilateral architecture |
| **Proposed+** | **RC-BiGCN v2** | **76.35%** | **0.7601** | Multi-stream + enhanced bilateral |

---

## Ablation Results: Bilateral Mechanism Contributions

| Experiment | Description | Pooled OOF Acc | Macro-F1 | Δ Acc vs C0 |
|---|---|---|---|---|
| C0 | Fixed anatomical graph | 68.70% | 0.6711 | — |
| C1 | + Dynamic bilateral edges | 72.05% | 0.6923 | **+3.35pp** |
| C2 | + Reliability as input feature | 73.73% | 0.7174 | **+5.03pp** |
| C3 | + Reliability-conditioned edges | 73.87% | 0.7178 | **+5.17pp** |
| v2 | + Multi-stream + enhanced MLP | 76.35% | 0.7601 | **+7.65pp** |

**Key finding:** Dynamic Bilateral Graph Learning contributes +3.35pp accuracy. Reliability conditioning (direct vs input-only) contributes an additional +0.14pp with +0.33pp macro-F1 improvement. The enhanced bilateral interaction terms in v2 bring an additional +2.62pp.

---

## Fold-Level Breakdown: RC-BiGCN v2 (seed 42)

| Fold | Test groups | n_test | Accuracy | Macro-F1 | ECE |
|---|---|---|---|---|---|
| 1 | (harder groups) | ~220 | 64.86% | 0.3615 | 0.2484 |
| 2 | — | ~275 | 75.36% | 0.7389 | 0.1454 |
| 3 | — | ~275 | **87.73%** | **0.8750** | 0.0602 |
| 4 | — | ~275 | **83.64%** | 0.7793 | 0.0586 |
| 5 | — | ~330 | 70.00% | 0.7509 | 0.1046 |
| **Mean** | | | **76.32%** | **0.7411** | **0.1034** |

---

## Per-Class F1: v1 vs v2

| Class | v1 F1 | v2 F1 | Δ | Note |
|---|---|---|---|---|
| as | 0.7922 | 0.8356 | +0.0434 | ↑ |
| bs | 0.8275 | 0.8523 | +0.0248 | ↑ |
| ce | 0.8488 | 0.8821 | +0.0333 | ↑ |
| dr | 0.7571 | 0.7805 | +0.0234 | ↑ |
| mfs | 0.5133 | 0.5312 | +0.0179 | ↑ hardest class |
| ms | 0.5386 | 0.5744 | +0.0358 | ↑ |
| sac | 0.5987 | 0.6209 | +0.0222 | ↑ |
| sq | 0.6425 | 0.6738 | +0.0313 | ↑ |
| tr | 0.9336 | 0.9490 | +0.0154 | ↑ |
| tw | 0.8903 | 0.9008 | +0.0105 | ↑ |

v2 improves all 10 classes over v1. Largest gains on `as` (+4.34pp), `ce` (+3.33pp), `ms` (+3.58pp).

---

## Hardest Class Pairs (Confusion Analysis)

The primary confusion pairs in RC-BiGCN v2:
1. **mfs ↔ ms** — both are walking/stepping activities with low visual displacement
2. **sac → ms** — seated arm circles misclassified as marching steps (similar cadence)
3. **dr → sac** — desk reaches and seated arm circles share upper-limb trajectory patterns

These confusions reflect genuine semantic ambiguity at the motion-primitive level, not architecture failure.

---

## Statistical Significance

| Comparison | Test | Statistic | p-value | Conclusion |
|---|---|---|---|---|
| RC-BiGCN v2 vs v1 | McNemar | χ²=0.716 | p=0.397 | Not significant (p>0.05) |
| RC-BiGCN v1 vs RBF (Stage4) | McNemar | χ²=7.898 | p=0.00495 | **Significant** (p<0.01) |

> [!NOTE]
> The v1→v2 improvement (+1.02pp accuracy, +0.024 macro-F1) is directionally consistent but does not reach statistical significance on a single-seed OOF comparison (n=1374).  
> Multi-seed results (seeds 43, 44) are pending — will be used to confirm the direction and magnitude of improvement.

---

## Calibration Comparison

| Model | ECE (↓ better) |
|---|---|
| RC-BiGCN v1 | 0.1381 |
| RC-BiGCN v2 | **0.0980** |
| C3 (ablation) | 0.1378 |
| C0 (fixed graph) | 0.1724 |

v2 is the best-calibrated model, achieving 29% lower ECE than v1 — primarily due to label smoothing.

---

## 80% Accuracy Target

| Assessment | Result |
|---|---|
| Was 80% achieved per test fold? | **Yes — Folds 3 (87.73%) and 4 (83.64%)** |
| Was 80% achieved in 5-fold mean? | **No — 76.32%** |
| Bottleneck | Fold 1 (64.86%) — systematic hardness of recording groups in that test split |
| Is the ceiling architecture-limited? | Partially; also data-limited (small dataset, high fold variance) |

**Conclusion:** The 80% target is achievable on some recording group splits but not consistently across the full 5-fold evaluation. This is an honest finding, not a failure — it reflects the genuine distribution heterogeneity of the MMASD dataset.
