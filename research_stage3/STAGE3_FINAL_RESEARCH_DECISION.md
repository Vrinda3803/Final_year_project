# STAGE 3 FINAL RESEARCH DECISION

## Autism Therapy Physical Activity Recognition via 2D Skeleton Dynamics
### PR-STGT / RA-STGT v2 — Stage 3 Research Conclusion

**Status:** Final & Fully Verified (All E0–E9 & Baselines Reconciled)
**Date:** November 2026
**Dataset:** MMASD local 2D OpenPose skeleton dataset — 1,374 clips, 10 activity classes, 32 group IDs

---

> [!IMPORTANT]
> This document is the authoritative research decision summary for the Stage 3 audit. It synthesizes verified results from all 9 completed Stage 3 documents. It does **not** present aspirational claims — only results that have been experimentally verified and recorded in this project.

---

## 1. What We Set Out to Do

We aimed to develop a technically sound, experimentally validated research contribution extending the RA-STGT framework for autism therapy activity recognition on 2D skeleton data. Specifically, we wanted to:

1. Identify a genuinely underexplored research problem
2. Develop a technically meaningful contribution beyond existing RA-STGT v1
3. Improve activity-recognition performance through a demonstrably useful mechanism
4. Establish through ablation experiments that new components cause the improvements
5. Develop a defensible paper with reproducible evidence

We explicitly stated: **the goal is not to manufacture the highest possible accuracy, but to discover and validate a genuinely useful methodological improvement.**

---

## 2. Verified Results Summary

### 2.1 Complete Model Comparison (All Verified from OOF CSVs)

| Experiment | T | P | M | Mean Fold Acc | Pooled OOF F1 | Fold Std | Status |
|---|:---:|:---:|:---:|---:|---:|---:|---|
| E0 Vanilla STGT | 0 | 0 | 0 | 37.86% | 0.3644 | 10.81% | ✅ Verified |
| E1 RA-STGT v1 | — | — | — | 46.47% | 0.4510 | 7.84% | ✅ Verified |
| E3 Phase Only | 0 | 1 | 0 | 49.97% | 0.4780 | 4.80% | ✅ Verified |
| E4 Margin Only | 0 | 0 | 1 | 55.58% | 0.5249 | 3.58% | ✅ Verified |
| E6 Topo+Phase | 1 | 1 | 0 | 56.21% | 0.5375 | 10.24% | ✅ Verified |
| E7 Topo+Margin | 1 | 0 | 1 | 57.04% | 0.5117 | 11.25% | ✅ Verified |
| **E8 Phase+Margin** | **0** | **1** | **1** | **51.45%** | **0.4760** | **3.42%** | ✅ **Verified** |
| E5 Full PR-STGT (orig) | 1 | 1 | 1 | 54.00% | 0.5029 | 11.96% | ✅ Verified |
| E9 Full PR-STGT (repro)| 1 | 1 | 1 | 53.41% | 0.4868 | 11.00% | ✅ Verified |
| **E2 Topology Only** | **1** | **0** | **0** | **57.63%** | **0.5656** | **9.60%** | ✅ **Best deep model** |
| **RBF Kernel SVM** | — | — | — | **71.69%** | **0.6920** | 8.14% | ✅ Local upper bound |

### 2.2 Improvement Trajectory

```
E0 (37.86%) → E1 (46.47%) → E3 (49.97%) → E4 (55.58%) → E7 (57.04%) → E2 (57.63%)
                                                                              ↑
                                                                     Best deep model
                                                                     
RBF Kernel SVM: 71.69% ← NOT matched by any deep model
```

---

## 3. What We Found: Key Scientific Findings

### 3.1 Finding 1 — Augmented Bilateral Topology is the Strongest Single Contribution

The augmented bilateral relational topology (E2) achieved:
- Mean fold accuracy: **57.63%** (+11.16pp over E1 baseline)
- Pooled OOF F1: **0.5656** (+0.1146 over E1)
- Best overall improvement on bilateral activity classes:
  - Chest Expansion: E1 F1=0.1667 → E2 F1=0.6054 (+0.4387)
  - Tree Pose: E1=0.1600 → E2=0.3200 (+0.1600)
  - Arm Swing: substantial improvement (documented in Stage 2)
  - Maracas: 0.1667 → 0.3333 (+0.1666)

This is the **central empirical finding** of this research project.

### 3.2 Finding 2 — The Full Model Does Not Beat Its Best Component

E5 (Full PR-STGT, all three components active) achieves pooled OOF F1 = 0.5029, which is **lower** than E2 alone (0.5656). Every two-component combination that includes topology also underperforms topology alone:

| Combination | Pooled OOF F1 | vs E2 |
|---|---:|---|
| E2 (T only) | **0.5656** | Reference |
| E6 (T+P) | 0.5375 | −0.0281 |
| E7 (T+M) | 0.5117 | −0.0539 |
| E5 (T+P+M) | 0.5029 | −0.0627 |

**The more components added to the topology mechanism, the worse the pooled F1.** This is a clear negative interaction.

This null result — that the full model does not outperform the simplest effective component — is **itself a scientifically valuable finding**. It prevents overclaiming and directs attention to the topology mechanism as the primary contribution.

### 3.3 Finding 3 — Full 2³ Factorial Analysis Reveals Dominant Negative Interactions (NEW)

With all 7 non-reproduction experiments complete (E1–E8), we computed full factorial main effects and interaction estimates:

| Factor / Interaction | Average Effect | Interpretation |
|---|---:|---|
| T (Topology) main effect | **+5.35pp** | Strongest positive main effect |
| M (Margin) main effect | **+1.95pp** | Modest positive when averaged |
| P (Phase) main effect | **−1.27pp** | Net *negative* across all contexts |
| T×P interaction | **−1.92pp** | Moderate negative: topology suppresses phase benefit |
| P×M interaction | **−4.63pp** | Strong negative: phase and margin conflict |
| **T×M interaction** | **−6.70pp** | **Dominant negative: topology and margin compete** |

**Key insight:** Margin loss is very effective in isolation (E1→E4: +9.11pp) but strongly counterproductive with topology (T×M = −6.70pp). Phase provides benefit in isolation (+3.51pp) but becomes net negative (−1.27pp average) when combined with other components. The gradient signals from margin targeting and topological relational learning are incompatible in the current joint training scheme.

### 3.3 Finding 3 — Phase Is Not the Cleanest Discriminator

Stage 3 Document 06 empirically measured:
- Drumming mean bilateral phase = **+0.1287** (only 23.6% antiphase tendency)
- Clapping mean bilateral phase = **+0.2103** (88.5% in-phase)

The original theoretical motivation ("drumming = antiphase, clapping = in-phase") is a **simplification**. The phase signal still contributes discriminative value, but its effectiveness is class-dependent and less clean than predicted.

### 3.4 Finding 4 — Wrist Occlusion Is Catastrophic

Both E2 and E5 suffer severe degradation under wrist occlusion:
- E2: 57.63% → 37.86% (−19.77pp accuracy)
- E5: 54.00% → 38.20% (−15.80pp accuracy)

The bilateral relational topology inherently depends on wrist signals. When wrists are occluded, the relational edges are suppressed (via reliability gating), which is mechanistically correct but still results in large performance loss because wrists are the primary discriminating feature for upper-body activities (drumming, clapping, arm swing, maracas).

**Deployment implication:** Camera positioning and wrist visibility are critical for reliable inference with this framework.

### 3.5 Finding 5 — The Local Tabular Baseline Is NOT Beaten

The RBF kernel SVM with hand-crafted features achieves 71.69% mean fold accuracy and OOF F1 = 0.6920 — both substantially above our best deep model (E2: 57.63%, F1=0.5656). This gap of ~14pp is not statistically significant (p=0.0760) due to n=5 folds but the effect size is large (Cohen's d = 1.06).

This means: **our graph neural network approach does not outperform a well-engineered tabular approach on this dataset.**

Possible reasons:
1. The 2D OpenPose skeleton quality may be insufficient for graph-based dynamics to generalize reliably
2. The 1,374-clip dataset with 32 groups is small for training deep GNNs
3. Hand-crafted features may already capture the discriminative signals better than learned representations

### 3.6 Finding 6 — Dataset Mismatch Prevents External Benchmarking

Our dataset (1,374 clips) does not match the published MMASD benchmark (1,202 clips). The difference is concentrated in the drumming class (343 clips vs 168). No comparison to RC-ProtoGCN (89.36%) or other published MMASD numbers is valid without establishing the source of this discrepancy.

---

## 4. The Recommended Research Contribution

### 4.1 Central Contribution

**Reliability-Conditioned Bilateral Relational Graph Learning for Autism Therapy Activity Recognition**

The core contribution is the augmented bilateral relational topology (E2 component): explicit left↔right joint relational edges weighted by the product of per-joint confidence scores from 2D pose estimation, integrated into a spatio-temporal graph transformer for autism therapy activity recognition.

This contribution is:
- **Empirically validated:** +11.16pp accuracy over RA-STGT v1 baseline
- **Mechanistically justified:** Receptive field analysis shows disjoint left/right wrist coverage in 2-layer GNN without bilateral edges
- **Class-specific evidence:** Bilateral activities (chest expansion, tree pose, arm swing) show the largest improvements
- **Contextually novel:** Combined bilateral relational topology + reliability-product weighting appears to be a distinguishing formulation in the ASD therapy recognition context

### 4.2 Honest Limitation Acknowledgements

The following must be explicitly stated in any publication:

1. **Full model null result:** The full PR-STGT (all three components combined) does not outperform the topology-only variant in pooled OOF F1. Papers should not claim the full combination improves on E2.

2. **Tabular baseline not beaten:** The RBF kernel SVM achieves substantially higher accuracy. The GNN contribution is in terms of interpretability, structural motivation, and class-specific improvements — not overall accuracy leadership.

3. **Statistical underpowering:** With n=5 group-independent folds, no pairwise comparison (except E1 vs RBF) achieves p<0.05. All results are consistent trends, not statistically proven improvements.

4. **Dataset version mismatch:** Our 1,374 clips vs published 1,202 clips prevents direct external benchmark comparison.

5. **Group IDs, not verified subject IDs:** The 32 group IDs are derived from filenames and have not been verified as unique biological individuals. The evaluation is "group-independent" not "subject-independent."

6. **Phase signal complexity:** Drumming shows only 23.6% antiphase tendency; claims about motor coordination interpretation must be tempered.

---

## 5. What Remains to Be Done Before Publication

| Task | Priority | Effort |
|---|---|---|
| Run multi-seed experiments (5 seeds × 5 folds) | High | ~40 hrs GPU or ~200 hrs CPU |
| Establish bilateral edge prior art in gesture/rehab literature | High | 1-2 days literature search |
| Match experimental protocol to Li et al. 2021 (1,202 clips) if possible | High | Identify source of extra 172 clips |
| Compare E2 with RC-ProtoGCN on matched data | High | Requires either obtaining RC-ProtoGCN code or matched dataset |
| Run Leave-One-Group-Out CV for statistical power | Medium | ~32× computation |
| Ablation on 2D vs. 3D (if 3D data obtained) | Low | Requires MMASD 3D data |
| AV-ASD dataset integration | Low | Requires separate dataset preprocessing |

---

## 6. Stage 3 Document Completion Status

| Document | Title | Status |
|---|---|---|
| 01_result_reconciliation.md | Result Audit and Reconciliation | ✅ Complete |
| 02_dataset_and_leakage_audit.md | Dataset and Leakage Audit | ✅ Complete |
| 03_matched_baseline_comparison.md | Matched Baseline Comparison | ✅ Complete |
| 04_component_interaction_ablation.md | Component Interaction Ablation | ✅ Complete |
| 05_bilateral_topology_investigation.md | Bilateral Topology Investigation | ✅ Complete |
| 06_bilateral_phase_analysis.md | Bilateral Phase Analysis | ✅ Complete |
| 07_reliability_robustness.md | Reliability Robustness Benchmark | ✅ Complete |
| 08_statistical_validation.md | Statistical Validation | ✅ Complete |
| 09_novelty_matrix.md | Novelty Assessment Matrix | ✅ Complete |
| STAGE3_FINAL_RESEARCH_DECISION.md | Final Research Decision | ✅ This document |

---

## 7. Recommended Paper Framing

### 7.1 Recommended Title Options

**Option A (Methods focus):**
> *Reliability-Conditioned Bilateral Relational Graph Transformers for Autism Therapy Activity Recognition from 2D Skeleton Data*

**Option B (Analysis focus):**
> *Bilateral Motor Coordination in Skeleton Graph Neural Networks: An Ablation Study on Autism Therapy Activity Recognition*

**Option C (Challenge/Dataset focus):**
> *Challenges in Deep Graph Learning for Autism Therapy Activity Recognition: A Systematic Ablation Study on 2D Pose Skeletons*

Option C is the most defensible given that the full model does not beat the tabular baseline and the full combination does not outperform its best component.

### 7.2 Contribution Statement Template

> We introduce a reliability-conditioned bilateral relational graph topology for autism therapy activity recognition from 2D OpenPose skeletons. Our approach augments the standard skeleton adjacency graph with explicit left↔right joint relational edges, weighted by the product of per-joint detection confidence scores. Through a systematic 3-factor ablation study across 8 component combinations, we demonstrate that this bilateral topology mechanism alone provides the strongest improvement (+11.2pp mean accuracy, +0.115 pooled F1 over the RA-STGT v1 baseline) among all evaluated PR-STGT components. We also report a negative interaction finding: combining multiple novel components does not additively improve performance, with the full model underperforming the best single component. All experiments use group-independent 5-fold cross-validation on a local MMASD 2D skeleton dataset (1,374 recordings, 10 therapy activity classes, 32 recording cohorts).

---

## 8. Final Verdict

| Question | Answer |
|---|---|
| Is there a genuine research contribution? | **Yes** — bilateral relational topology with reliability weighting for therapy recognition |
| Does the full model beat baselines? | **No** — topology-only (E2) beats the full model; RBF SVM still leads overall |
| Is the contribution statistically proven? | **No** — consistent trend (Cohen's d=0.68) but p=0.2031 with n=5 folds |
| Is the finding reproducible? | **Yes** — OOF CSVs, checkpoints, and audit scripts fully preserved |
| Is the novelty defensible? | **Partially** — contextually novel in ASD therapy context; prior art search incomplete |
| Is the paper publishable in current state? | **Not yet** — multi-seed validation and external benchmark comparison needed |
| What is the strongest honest claim? | Bilateral relational topology + reliability weighting is the most effective single mechanism among those evaluated, with consistent directional improvement and mechanistic justification |

---

*All results referenced in this document are traceable to OOF prediction CSVs and audit scripts in `research_stage3/`. No results have been modified, fabricated, or selectively omitted.*
