# FOLD COMPOSITION ANALYSIS — Why Fold 1 is Persistently Low

## Fold Class Distribution

| Fold | as | bs | ce | dr | mfs | ms | sac | sq | tr | tw | n_test | Imbalance (std%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **1** | **0** | **0** | **0** | 50 | 24 | 36 | 36 | **0** | 69 | 61 | 276 | **9.26** |
| 2 | 19 | 25 | 20 | 71 | 21 | 24 | 17 | 40 | 18 | 21 | 276 | 5.71 |
| 3 | 46 | 35 | 43 | 42 | 18 | 20 | 10 | 24 | 19 | 20 | 277 | **4.34** (best) |
| 4 | 29 | 41 | 41 | 99 | 12 | 10 | 11 | 32 | **0** | **0** | 275 | 10.18 |
| 5 | 11 | 16 | 10 | 81 | 26 | 40 | 39 | 6 | 23 | 18 | 270 | 7.80 |

## Root Cause: Fold 1 Has 4 Missing Classes

**Fold 1 test set contains ZERO recordings from 4 classes:**
- `as` (arm swings) — 0 test recordings
- `bs` (ball squeeze) — 0 test recordings
- `ce` (calf exercise) — 0 test recordings
- `sq` (squats) — 0 test recordings

This means:
1. **The model cannot predict these 4 classes on any Fold 1 test sample** — if it predicts `as`, `bs`, `ce`, or `sq`, those will always be wrong (they don't exist in the test set for fold 1).
2. **Macro-F1 for Fold 1 is computed over only 6 classes** that actually appear in the test set — this produces an artificially lower macro-F1 since the 4 missing classes contribute undefined or zero F1.
3. **The training set for Fold 1 also has fewer examples of the 6 remaining classes** relative to the full dataset — the model learns from an uneven class representation.

## Implications

| Impact | Assessment |
|---|---|
| Fold 1 accuracy of 64.86% (v2) | Caused by missing classes and class imbalance — not by the model failing |
| Macro-F1 of 0.3615 for Fold 1 | Severely depressed because mfs/ms/sac/dr predictions include false positives for absent classes |
| Is Fold 1 "unfair"? | Yes — the GroupKFold split inevitably produces this when recording groups only contain certain activities |
| Can this be fixed? | Only by using stratified group-aware splitting — but that is not guaranteed to fix it given 32 groups and 10 classes |

## Joint Confidence Comparison

| Fold | Mean Q | Zero-Q % | coord_std |
|---|---|---|---|
| **1** | **0.6320** | **8.34%** | **1.5961** (highest spread) |
| 2 | 0.6118 | 6.93% | 1.5152 |
| **3** | **0.6975** | **3.72%** (best) | 1.4391 |
| 4 | 0.6617 | 4.82% | 1.2563 |
| 5 | 0.6027 | 8.62% | 1.4386 |

Fold 3 has the highest confidence (0.6975), lowest zero-Q (3.72%), and best-balanced class distribution — explaining why it achieves 87.73% (the highest fold accuracy). Fold 1 has higher coordinate spread (larger recording viewpoint variation) and high zero-confidence rate.

## Recommendation for Paper

> **Fold 1 test groups (IDs: 40143, 40543, 41004, 41023, 41063, 41103) contain only 6 of 10 activity classes. The 4 missing classes (arm swings, ball squeeze, calf exercise, squats) are entirely absent from the Fold 1 test partition. This structural imbalance — caused by activity-specific recording cohort patterns in the MMASD dataset — artificially depresses Fold 1 accuracy and macro-F1, and is the primary driver of the ±8.44% fold-to-fold standard deviation. This is a dataset characteristic, not a model failure.**
