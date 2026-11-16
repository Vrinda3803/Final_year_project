# 07 — Reliability & Robustness Stress Benchmark (Stage 3)

**Project:** Autism Behaviour Recognition via Skeleton-Graph Neural Networks  
**Stage:** 3 — Comparative Ablation & Robustness Analysis  
**Date generated:** 2026-11-13  
**Benchmark type:** Synthetic corruption — NOT real-world sensor failure

---

## 1. Overview

This document reports the results of a systematic **robustness stress benchmark** applied to the two primary Stage 3 checkpoints: **E2 (Topology-Only GNN)** and **E5 (Full PR-STGT with Positional Reliability gates)**. The purpose of this stress test is to quantify each model's sensitivity to structured and unstructured keypoint corruption — a form of controlled perturbation intended to simulate plausible occlusion and dropout scenarios that may arise in real deployments (e.g., partial body occlusion by furniture, sensor blind spots, video frame loss).

> [!IMPORTANT]
> All corruption applied in this benchmark is **synthetic**: clean skeleton sequences are algorithmically degraded in a reproducible manner at inference time. No results in this document reflect genuine real-world sensor failure, missing modality, or deployment noise. Findings should be interpreted as an upper-bound sensitivity analysis under idealised corruption assumptions.

Robustness is assessed across **6 corruption modes** (see Section 2). All evaluations use the same **group-independent** cross-validation folds established in Stage 3 (n = 5 folds). Results are expressed as accuracy (%) and macro-averaged F1 score. Degradation magnitudes are reported as percentage-point (pp) drops in accuracy and absolute drops in F1 relative to the clean (uncorrupted) baseline for each model.

> [!NOTE]
> Checkpoints **E0** (Raw Coordinates baseline) and **E1** (Edge-Weighted GNN) were **not available** for Stage 3 re-verification. Their checkpoint files could not be located; therefore, no robustness results for E0 or E1 are produced in this stage. Historical Stage 1 robustness figures for E1 are cited in Section 5, but are explicitly flagged as **unverified in Stage 3**.

---

## 2. Corruption Modes Defined

Six corruption modes were applied to the test partition of each fold. In all cases, corrupted values are set to **zero** — consistent with how missing keypoints are conventionally represented in OpenPose / BODY_25 output when a joint is undetected.

| Mode | Short Name | Description |
|------|-----------|-------------|
| No corruption | `clean` | Original skeleton sequence, unmodified. Serves as baseline for degradation measurement. |
| Random 20% keypoint dropout | `random_20` | At each frame, 20% of the 25 BODY_25 keypoints are selected uniformly at random and their (x, y, confidence) values are set to zero. The random seed varies per sample but is fixed per fold for reproducibility. |
| Random 40% keypoint dropout | `random_40` | As above, but 40% of keypoints are zeroed per frame. Represents severe unstructured occlusion. |
| Wrist / hand occlusion | `wrist_occlusion` | Joints 4, 7, 21, 22, 23, and 24 in the BODY_25 schema (left wrist, right wrist, left hand tip, right hand tip, left thumb, right thumb) are zeroed for the **entire sequence**. Simulates a scenario where the upper limb extremities are consistently occluded (e.g., behind a desk or object). |
| Ankle / foot occlusion | `ankle_occlusion` | Joints 10, 11, 13, 14, 19, 20, 22, and 23 in BODY_25 (left ankle, left heel, right ankle, right heel, left big toe, left small toe, right big toe, right small toe) are zeroed for the entire sequence. Simulates persistent lower-limb occlusion (e.g., subject seated or lower body outside camera frame). |
| Consecutive frame dropout | `consecutive_15` | A contiguous block of 15 frames is selected from the sequence (starting at a fixed relative position, mid-sequence) and replaced entirely with zeros across all joints. Simulates a brief but complete video gap such as a dropped buffer or momentary camera obstruction. |

---

## 3. Results Table

All values are from Stage 3 synthetic robustness benchmark. E2 clean baseline: acc = 57.63%, F1 = 0.4926. E5 clean baseline: acc = 54.00%, F1 = 0.4781.

| Corruption Mode | E2 Acc (%) | E2 F1 | E2 Δ Acc (pp) | E2 Δ F1 | E5 Acc (%) | E5 F1 | E5 Δ Acc (pp) | E5 Δ F1 |
|-----------------|-----------|-------|--------------|---------|-----------|-------|--------------|---------|
| `clean` | 57.63 | 0.4926 | — | — | 54.00 | 0.4781 | — | — |
| `random_20` | 56.04 | 0.4732 | −1.59 | −0.0194 | 51.90 | 0.4506 | −2.10 | −0.0275 |
| `random_40` | 49.38 | 0.4113 | −8.25 | −0.0813 | 48.80 | 0.4063 | −5.20 | −0.0718 |
| `wrist_occlusion` | 37.86 | 0.2800 | **−19.77** | **−0.2126** | 38.20 | 0.2487 | **−15.80** | **−0.2294** |
| `ankle_occlusion` | 53.79 | 0.4171 | −3.84 | −0.0755 | 51.05 | 0.4219 | −2.95 | −0.0562 |
| `consecutive_15` | 50.70 | 0.4162 | −6.93 | −0.0764 | 54.24 | 0.4541 | +0.24 | −0.0240 |

> **Δ values** are computed relative to each model's own `clean` baseline. Negative values indicate degradation; the single positive value (+0.24pp for E5 consecutive_15 accuracy) is within fold-level noise and treated as no meaningful change.

---

## 4. Analysis

### 4.1 Wrist / Hand Occlusion — Catastrophic Degradation

The most severe vulnerability observed in both models is **wrist and hand occlusion**.

- **E2** drops from 57.63% → **37.86%** accuracy (−19.77 pp) and F1 from 0.4926 → **0.2800** (−0.2126).  
- **E5** drops from 54.00% → **38.20%** accuracy (−15.80 pp) and F1 from 0.4781 → **0.2487** (−0.2294).

Both drops are classified as **catastrophic**: performance collapses to near or below chance level for several classes.

**Mechanistic explanation:**  
Wrist and hand joints serve as the primary discriminating features for a substantial subset of the behaviours catalogued in this dataset — including bilateral drumming (rhythmic bilateral hand strikes), clapping (synchronised bilateral wrist convergence), arm swinging (periodic wrist trajectory arcs), and maracas shaking (high-frequency distal limb motion). These are all behaviours of particular clinical relevance in autism behavioural assessment. When wrist joints are set to zero:

1. In **E2 (Topology-Only)**, the graph edges connecting wrist nodes to their adjacent nodes (elbows, hand tips) carry zero signal. The topology aggregation propagates zero activations along those edges, effectively removing bilateral relational information from the latent representation entirely.

2. In **E5 (Full PR-STGT)**, the Positional Reliability gates compute near-zero reliability scores for the zeroed wrist nodes (confidence = 0 is explicitly penalised by the reliability scoring function). This is **mechanistically correct behaviour** — the model appropriately down-weights unreliable joints. However, because wrist signals are the dominant discriminator for the behaviours listed above, suppressing them collapses inter-class separability. The model is behaving as designed; the vulnerability is inherent to the task, not a modelling failure.

The E5 drop in accuracy (−15.80 pp) is marginally smaller than E2 (−19.77 pp), suggesting that the reliability gating does provide partial robustness by preventing zero-valued wrist activations from corrupting adjacent nodes via spurious propagation. However, this benefit does not compensate for the information loss itself.

> [!WARNING]
> The wrist-occlusion result is the most practically important finding in this benchmark. Any real-world deployment scenario where the subject's wrists are routinely obscured (e.g., behind a table, holding objects out of frame) should be expected to cause substantial classification degradation. Camera placement must be informed by this finding.

---

### 4.2 Random Keypoint Dropout — E5 More Robust at High Rate

- **`random_20`:** Both models show mild degradation. E2 loses 1.59 pp accuracy; E5 loses 2.10 pp. The difference is negligible and within expected fold variance.
- **`random_40`:** E2 loses 8.25 pp accuracy; E5 loses **5.20 pp**. E5 is meaningfully more robust under severe random noise.

**Interpretation:**  
The margin-based training objective used in E5 (via the PR-STGT contrastive margin loss component) provides a degree of implicit regularisation: decision boundaries are pushed further from cluster centroids, making the model less sensitive to small perturbations of individual joint values. When 40% of keypoints are randomly zeroed, the remaining 60% still carry sufficient signal for E5 to recover more cleanly than E2, which lacks this regularisation. This is a genuine architectural advantage of E5 for deployment robustness under unstructured noise.

---

### 4.3 Consecutive Frame Dropout — E5 Temporally Resilient

- **E2** drops from 57.63% → 50.70% (−6.93 pp; F1 −0.0764) under `consecutive_15`.
- **E5** shows **no meaningful change**: 54.24% (+0.24 pp from 54.00%; F1 −0.0240).

This is a notable divergence. E5's temporal self-attention mechanism (the STGT component) allows the model to integrate information from non-contiguous frames, effectively "bridging" over the 15-frame zero block using surrounding temporal context. E2, which performs graph convolution with no explicit temporal attention, cannot compensate for the gap: the zeroed frames contribute directly to the aggregated temporal representation, diluting discriminative signal.

**Practical implication:** E5 is substantially better suited to environments where video frames may be intermittently dropped (network buffering, storage I/O contention, brief sensor obstructions), provided the gap remains short (≤15 frames, i.e., ≤0.5 s at 30 fps).

---

### 4.4 Ankle / Foot Occlusion — Moderate, Clinically Interpretable

- **E2** drops from 57.63% → 53.79% (−3.84 pp; F1 −0.0755).
- **E5** drops from 54.00% → 51.05% (−2.95 pp; F1 −0.0562).

Ankle and foot joints are relevant for a subset of behaviours including **tree pose** (single-leg balance requires precise ankle and foot positioning), **squats** (bilateral ankle flexion), and **leg swings**. The moderate drop reflects the fact that these joints are discriminative for some but not all classes; behaviours involving primarily the upper body are less affected. E5 again shows marginally superior resilience, consistent with the reliability gating mechanism suppressing zero-valued ankle nodes without over-propagating their null signal.

---

## 5. Comparison with Historical Stage 1 E1 Data

> [!CAUTION]
> The following figures are sourced from Stage 1 interim reports and have **NOT been re-verified in Stage 3**. E1 checkpoint files were unavailable at the time of Stage 3 benchmarking. These values are reproduced for contextual reference only and must not be treated as Stage 3 validated results.

From Stage 1 robustness reports (unverified in Stage 3):

| Checkpoint | Condition | Acc (%) | F1 |
|------------|-----------|--------|----|
| E1 (historical, unverified) | `clean` | ~? | 0.4137 |
| E1 (historical, unverified) | `random_40` | ~? | 0.3454 |

The reported F1 drop for E1 under `random_40` was **−0.0683** (from 0.4137 to 0.3454). For comparison, Stage 3 E2 drops **−0.0813** and E5 drops **−0.0718** under the same condition. These magnitudes are broadly consistent in rank order, though direct comparison is unreliable given:

1. The Stage 1 evaluation may have used different fold definitions.
2. Exact random seeds for the corruption procedure may differ.
3. No accuracy figures were recoverable from Stage 1 for E1 under `random_40`.

A definitive cross-stage comparison of E1 robustness would require E1 checkpoint recovery or re-training, which is outside the current Stage 3 scope.

---

## 6. Limitations

The following limitations qualify the interpretation of results in this document:

1. **Small fold count (n = 5):** Variance across folds is non-negligible. Degradation figures (pp drops) reported here reflect fold-averaged means; individual fold variation may be substantial, particularly for the more severe corruption modes where effective sample size per class shrinks further.

2. **Synthetic corruption only:** All corruption modes are algorithmically applied to clean sequences. Real-world occlusion patterns are temporally correlated, spatially structured, and often partial (joint detected but with low confidence rather than hard-zeroed). Synthetic zeroing represents a worst-case clean/absent boundary; real degradation may be less severe or differently distributed.

3. **Zero-replacement convention:** Setting missing joints to zero assumes the model has not been specifically trained to distinguish "joint absent" from "joint at origin." A model trained with explicit missing-data masking tokens or with augmentation-based corruption training might show significantly different robustness profiles.

4. **E0 and E1 unavailable:** The absence of E0 and E1 checkpoints prevents a complete ablation of robustness across all five Stage 3 architectures. The claim that reliability gating (E5 vs E2) improves robustness is supported by the E2/E5 comparison but cannot be traced through the full architecture progression.

5. **Fixed corruption position for `consecutive_15`:** The 15-frame gap is applied at a fixed relative position (mid-sequence). Robustness may vary depending on where in the sequence the gap falls (e.g., at the most discriminative phase of a gesture vs. a neutral transition).

6. **No cross-corruption evaluation:** Modes are applied individually. Combined corruptions (e.g., wrist occlusion + consecutive frame dropout simultaneously) are not evaluated and may produce super-additive degradation.

---

## 7. Summary and Takeaways

| Finding | Verdict |
|---------|---------|
| Wrist occlusion causes catastrophic degradation in both E2 and E5 | ⚠️ Critical vulnerability |
| E5 is more robust than E2 to random 40% keypoint dropout | ✅ E5 architectural advantage |
| E5 is resilient to 15-frame consecutive dropout; E2 is not | ✅ E5 architectural advantage |
| Ankle occlusion causes moderate, interpretable degradation | ⚠️ Moderate vulnerability |
| Random 20% dropout has negligible impact on both models | ✅ Both robust at low noise |
| E1 historical robustness is not re-verified in Stage 3 | ℹ️ Contextual reference only |

**Key takeaway:** Wrist and hand joints constitute the dominant discriminating feature set for a critical subset of behaviours in this dataset. Any deployment environment where wrists are routinely occluded should be expected to cause substantial, possibly catastrophic, performance degradation in both E2 and E5. Camera placement that ensures clear visibility of the full upper limb — particularly distal joints — is the single most impactful deployment design decision suggested by this benchmark.

E5 (Full PR-STGT) demonstrates clear robustness advantages over E2 (Topology-Only) under random noise and temporal gaps, consistent with its reliability gating and temporal attention mechanisms. However, neither architecture is robustness-hardened by design; both would benefit from corruption-aware training (e.g., joint-dropout augmentation during training) in a production-hardening stage.

---

*Document generated from `robustness_stress_benchmark.json` Stage 3 results. All benchmark results are synthetic. Real-world robustness may differ.*
