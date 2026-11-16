# Novelty Assessment Matrix — PR-STGT Framework
## Autism Therapy Physical Activity Recognition via 2D Skeleton Graph Neural Networks

**Document status:** Internal research assessment — honest, defensible framing only  
**Framework:** PR-STGT (Phase-Relational and Reliability-Gated Spatio-Temporal Graph Transformer)  
**Dataset:** MMASD (our version: 1,374 clips; published Li et al. 2021 version: 1,202 clips)  
**Date:** 2026-11

---

## 1. Executive Summary

PR-STGT combines five components. After rigorous internal assessment, we claim **no component is fully novel in isolation** — all draw on established ideas from action recognition, pose estimation, and metric learning. However, the framework makes **contextually novel contributions** in how these ideas are combined and adapted for the specific challenges of autism therapy recognition: low-confidence 2D pose estimation, bilateral motor coordination analysis, and fine-grained confusable therapy activity pairs.

### What we claim (at appropriate confidence levels):

| Claim level | Component | Defensibility |
|---|---|---|
| **Contextually novel combination** | Augmented Bilateral Relational Topology (E2) with edge-conditioned reliability weighting | Moderate — bilateral relational edges + reliability product weighting together may not appear in prior skeleton GCN work, but each ingredient is known |
| **Transfer novelty** | Reliability-Weighted Graph Attention (E1) applied to therapy-domain 2D pose | Moderate — Cheng et al. 2020 is prior art for confidence-aware skeleton; our specific attention-weight formulation may differ |
| **Domain-specific application** | Bilateral Phase Velocity Stream (E3) as explicit motor coordination signal | Low-to-moderate — phase signals in gait/rehab literature likely have prior art; bilateral cosine-of-angle-difference formulation in GNN may be novel in ASD context |
| **Domain-specific application** | Confusable-Pair Margin Loss (E4) targeting known error pairs | Low-to-moderate — ArcFace/Circle Loss are strong prior art; our per-class-pair targeting in logit space for a small therapy set may be a minor adaptation |
| **Standard post-hoc calibration** | Temperature Calibration | Low — well-established technique; no novelty claimed |

> [!IMPORTANT]
> The full model (E5) does **not** outperform the best single component (E2). This is an honest null result that must be reported. The paper cannot claim the full combination is better than its parts. The contribution framing must shift accordingly.

> [!WARNING]
> Our dataset version (1,374 clips) does not match published MMASD benchmarks (1,202 clips). No direct comparison to RC-ProtoGCN (89.36%) or other published MMASD numbers is valid. This must be stated explicitly in any submission.

---

## 2. Novelty Matrix

| Component | Prior art in general action rec? | Prior art in ASD/therapy rec? | Our adaptation / difference | Novelty level | Confidence in claim |
|---|---|---|---|---|---|
| **E1: Reliability-Weighted Graph Attention** | **Yes** — Cheng et al. CVPR 2020 (confidence-aware skeleton); attention-based edge weighting common in GAT, CTR-GCN | **Probably not** — confidence-gated attention in ASD/therapy recognition context not found in search | Confidence scores from OpenPose used directly as edge-weight multipliers in spatial attention; simpler than Cheng et al.'s full pipeline | **Low-to-Medium (transfer novelty)** | Moderate — Cheng et al. 2020 is close prior art; novelty depends on exact formulation difference |
| **E2: Augmented Bilateral Relational Topology** | **Partial** — dynamic topology in CTR-GCN; bilateral/symmetry edges explored in some gesture/gait works (specific papers unknown to us) | **Probably not** — bilateral relational edges in ASD therapy skeleton GCN not found | Adds explicit left↔right joint edges (wrists, elbows, knees, ankles) weighted by r_u × r_v reliability product; extends static BODY_25 adjacency dynamically | **Medium (contextual novelty)** | Moderate — bilateral edges per se may appear in gesture/rehab literature; combined with reliability weighting in this exact form is likely novel in context |
| **E3: Bilateral Phase Velocity Stream** | **Partial** — phase signals common in gait analysis; velocity features standard in action rec; bilateral coordination metrics in biomechanics | **Probably not** — explicit φ(t) = cos(θ_L − θ_R) phase stream as GNN feature in ASD therapy not found | Derives bilateral phase signal from inter-frame velocity directions; added as secondary feature stream to capture motor coordination disruptions characteristic of ASD | **Low-to-Medium (domain application)** | Low — gait/rehab literature likely has bilateral phase metrics; our specific implementation in GNN for ASD therapy may be novel but we cannot confirm without exhaustive search |
| **E4: Confusable-Pair Margin Loss** | **Yes** — ArcFace (Deng et al. 2019), Circle Loss (Sun et al. 2020) are strong prior art; targeted pair losses used in metric learning | **Probably not** — domain-specific confusable pair targeting in therapy activity recognition not found | Applies cosine-margin penalty to specific error-analysis-identified pairs (drumming/clapping, maracas/maracas-fast-shake, squats/arm-circle) in logit space rather than embedding space | **Low-to-Incremental (domain-specific adaptation)** | Moderate-high — ArcFace/Circle Loss cover the concept well; our adaptation is domain-driven selection, not a new mathematical technique |
| **E5: Temperature Calibration** | **Yes** — Guo et al. ICML 2017 (temperature scaling), Platt scaling | **Unknown** — not specifically searched | Standard post-hoc calibration applied after training | **Low (standard technique)** | High — this is well-established; no novelty claimed |

---

## 3. Detailed Component Assessments

### 3.1 Reliability-Weighted Graph Attention (E1)

**What we do:** Per-joint confidence scores from OpenPose are used to scale spatial graph attention edge weights. Joints with low detection confidence contribute less to message-passing in the GNN.

**Prior art:** Cheng et al. (CVPR 2020) is the most directly relevant work, addressing noisy/incomplete skeleton data with confidence-aware processing. Graph attention networks (Veličković et al. 2018) generically learn edge weights. CTR-GCN (Chen et al. ICCV 2021) refines topology dynamically, though without explicit confidence weighting.

**Our difference:** We use raw OpenPose confidence scores as multiplicative modifiers rather than learning separate confidence representations. This is simpler and more directly tied to the pose estimator's uncertainty.

**Honest caveat:** Whether this specific formulation constitutes a meaningful enough difference from Cheng et al. 2020 to claim novelty is debatable. If Cheng et al.'s method covers attention-weight scaling by confidence, our contribution in E1 alone is incremental at best. The empirical contribution (E2 >> E1 in ablation) also weakens the centrality of this claim.

**Novelty level: Low-to-Medium (transfer novelty in ASD context, possibly incremental vs. Cheng et al.)**

---

### 3.2 Augmented Bilateral Relational Topology (E2)

**What we do:** We add relational edges between bilateral joint pairs (left wrist ↔ right wrist, left elbow ↔ right elbow, left knee ↔ right knee, left ankle ↔ right ankle) to the static BODY_25 skeleton adjacency graph. These added edges are weighted by the product of the two joints' reliability scores (r_u × r_v), creating dynamic connectivity that is both structurally motivated and uncertainty-aware.

**Prior art:** Bilateral symmetry in human pose is well-known. Some works in gesture recognition and gait analysis incorporate symmetry-based features. Dynamic/adaptive topology is core to CTR-GCN. However, we have not found a specific prior work that (a) adds explicit bilateral relational edges to the skeleton graph AND (b) weights them by the product of per-joint confidence scores from a 2D pose estimator.

**Our difference:** The combination of explicit bilateral structural edges + reliability-product weighting + application to a therapy activity recognition context appears to be a distinguishing contribution. This is our strongest empirical result (+11.46% F1 over baseline).

**Honest caveat:** Bilateral edges in skeleton graphs may appear in gesture/hand gesture or rehabilitation literature that we have not exhaustively searched. The reliability-product weighting is a natural choice that could have been proposed elsewhere. Our claim is strongest when framed as a *combination* novel in the ASD/therapy context, not as wholly new components.

**Novelty level: Medium (contextual novelty; combined formulation likely novel in therapy/ASD recognition)**

---

### 3.3 Bilateral Phase Velocity Stream (E3)

**What we do:** We compute inter-frame velocity vectors for each joint, derive velocity-direction angles (θ), and form a bilateral phase signal φ(t) = cos(θ_L(t) − θ_R(t)) for each bilateral joint pair. This scalar time series captures synchrony/asynchrony between left-right limb movements and is added as a secondary feature stream to the GNN.

**Prior art:** Phase signals are foundational in gait analysis and clinical biomechanics. Bilateral coordination metrics (inter-limb phase coherence, relative phase) are well-established in motor control research. Velocity features are standard in action recognition. However, their encoding as a GNN feature stream specifically for ASD therapy activity recognition is less clearly documented.

**Our adaptation:** The motivation is domain-specific: ASD is associated with altered bilateral motor coordination patterns. By making this signal explicit as a network input rather than relying on the GNN to learn it implicitly, we incorporate domain knowledge about ASD motor characteristics.

**Honest caveat:** This is the component with the *lowest* confidence in novelty claims. The bilateral phase concept is not new in biomechanics. The specific cos(θ_L − θ_R) formulation as a skeleton GNN feature stream may or may not have prior art — we cannot confirm without an exhaustive literature search across gait analysis, rehabilitation, and human activity recognition. E3's empirical contribution was also modest relative to E2.

**Novelty level: Low-to-Medium (domain application of established biomechanics concept; novelty in GNN integration for ASD uncertain)**

---

### 3.4 Confusable-Pair Margin Loss (E4)

**What we do:** Based on error analysis identifying specific confusable pairs (drumming ↔ clapping; maracas ↔ maracas-fast-shake; squats ↔ arm-circle), we add a cosine-margin penalty in logit space that explicitly discourages the model from confusing these pairs. The loss targets specific class pairs rather than applying a general metric learning objective.

**Prior art:** ArcFace (Deng et al. 2019) and Circle Loss (Sun et al. 2020) are the central prior works. Both apply margin-based objectives to push apart similar-but-distinct classes in embedding space. Targeted pair losses in fine-grained recognition are not unprecedented.

**Our difference:** We apply the margin in logit space (classification head) rather than embedding space, and we select pairs based on domain-specific error analysis rather than general similarity metrics. This is a practical engineering choice driven by known class confusion in the therapy dataset.

**Honest caveat:** The mathematical novelty here is low. The technique is a domain-driven instantiation of well-understood metric learning ideas. The contribution is primarily *methodological*: demonstrating that targeted pair selection based on error analysis improves confusable class discrimination. The empirical gains from E4 alone were also not the primary driver of overall performance.

**Novelty level: Low-to-Incremental (domain-specific selection strategy; mathematical framework is prior art)**

---

### 3.5 Temperature Calibration

**What we do:** Standard post-hoc temperature scaling (Guo et al. ICML 2017) applied to the trained model's logits.

**Prior art:** Fully established technique. Guo et al. 2017 is the canonical reference.

**Novelty level: Low — no novelty claimed. Included for completeness and practical reliability of confidence outputs.**

---

## 4. What We Do NOT Claim as Novel

The following are explicitly **not** claimed as novel contributions:

1. **2D skeleton-based action recognition** — foundational since ST-GCN (Yan et al. AAAI 2018) and extensively developed since.
2. **Graph neural networks for skeleton data** — well-established; this paper is an application, not a GNN architecture paper.
3. **Using OpenPose confidence scores** — OpenPose outputs confidence as a standard feature; using it is not novel.
4. **Temperature scaling / Platt calibration** — standard post-processing.
5. **Margin-based loss functions** — ArcFace, Circle Loss are prior art.
6. **Velocity features in action recognition** — standard feature engineering.
7. **The MMASD dataset** — created by Li et al. ACM MM 2021; we are users, not creators.
8. **Any performance claim relative to published MMASD benchmarks** — our dataset version differs; direct comparison is invalid.

---

## 5. Limitations of the Novelty Claims

### 5.1 Non-exhaustive literature search
Our prior art review prioritized action recognition (ST-GCN, CTR-GCN, SkateFormer), confidence-aware skeleton work (Cheng et al.), and metric learning (ArcFace, Circle Loss). We have **not** exhaustively searched:
- Rehabilitation and clinical motion analysis literature (gait, Parkinson's, CP rehabilitation)
- ASD motor behavior analysis (non-deep-learning clinical studies)
- Gesture recognition with bilateral/symmetry features
- Full-text searches of ICLR, ECCV, BMVC 2022–2025 proceedings for bilateral skeleton topology

It is **possible** that prior art exists for E2 or E3 in adjacent fields we did not fully survey. Claims should be qualified accordingly in any submission.

### 5.2 Single dataset, single domain
All empirical results are from a single dataset (MMASD, our 1,374-clip version). Generalizability of the proposed components to other action recognition datasets is undemonstrated.

### 5.3 Full model underperforms best component
E5 (full model) achieves pooled F1 = 0.5029, lower than E2 alone (F1 = 0.5656). This is a null result for the combined framework. The paper cannot claim the full PR-STGT framework outperforms ablated variants — it does not. The contribution must be framed around the analysis and the individual component findings, not system-level superiority.

### 5.4 Local RBF baseline not beaten
A local RBF (hand-crafted feature) baseline achieves 71.69%, which no deep model in our experiments beats. This is an important honest disclosure: the deep learning approach does not outperform a well-tuned classical baseline on this dataset. This may reflect data scarcity, label quality, or domain mismatch effects.

### 5.5 Dataset version mismatch
Our 1,374-clip dataset differs from the 1,202-clip published MMASD. The source of the discrepancy is unresolved. No comparison to RC-ProtoGCN (89.36%) or other published results on MMASD is valid until dataset alignment is confirmed.

### 5.6 Confidence score quality
Reliability weights depend entirely on OpenPose confidence quality. If confidence scores are poorly calibrated (common in therapy settings with occlusion, non-standard poses), the reliability-weighted components may be less effective or even counterproductive. We have not evaluated OpenPose calibration quality on this specific dataset.

---

## 6. Recommended Framing for Research Paper

### 6.1 Recommended title options (honest framing)

> **Option A (component-focused):**
> *"Bilateral Relational Topology for Reliability-Aware Skeleton Graph Learning in Autism Therapy Activity Recognition"*

> **Option B (analysis-focused, honest about null results):**
> *"Toward Robust 2D Skeleton-Based Autism Therapy Activity Recognition: Components, Ablations, and Challenges"*

> **Option C (dataset/challenge framing):**
> *"Challenges in Deep Skeleton-Based Recognition for Autism Therapy: Bilateral Structure, Reliability, and the Limits of Deep Learning on Small Clinical Datasets"*

Option A is appropriate if the paper is pitched as a methods contribution. Options B/C are more appropriate if the paper is pitched as an empirical study or systems paper with honest reporting of null results.

### 6.2 Recommended contribution statement

> We propose the **Augmented Bilateral Relational Topology** — a lightweight extension to static skeleton adjacency graphs that adds explicit left-right joint relational edges weighted by the product of per-joint pose estimation confidence scores. Applied to autism therapy activity recognition on the MMASD dataset, this component yields the largest empirical gain in our ablation study (+11.46% F1 over baseline). We additionally investigate phase velocity streams encoding bilateral motor coordination, reliability-weighted graph attention, and confusable-pair margin loss, finding that the full combined model does not outperform the topology component alone — a result that itself provides insight into the challenges of this domain. All components are evaluated under consistent conditions on a 1,374-clip version of MMASD; we note this dataset configuration differs from the 1,202-clip version used in published comparisons, precluding direct benchmark comparison.

### 6.3 What to avoid claiming

- Do **not** claim state-of-the-art on MMASD (dataset mismatch; baseline not beaten).
- Do **not** claim the framework is universally better (null result from E5 vs E2).
- Do **not** claim bilateral phase signal is a novel idea (prior art likely exists in biomechanics).
- Do **not** claim confusable-pair loss is mathematically novel (ArcFace/Circle Loss cover the concept).
- Do **not** cite papers you have not read or cannot verify.

### 6.4 Strongest defensible claims

1. **E2 (Augmented Bilateral Relational Topology)** is the most defensible novel contribution: its specific combination of bilateral structural edges + reliability-product weighting + application to ASD therapy recognition appears not to have a direct prior art equivalent, though adjacent work in gesture/gait must be acknowledged as potentially related.
2. **E1 (Reliability-Weighted Graph Attention)** is a legitimate transfer of confidence-aware methods (Cheng et al.) to the ASD context, with the honest caveat that it is transfer novelty, not conceptual novelty.
3. **The ablation and null result** (full model < best component) is itself a valuable honest contribution to the field, documenting the difficulty of combining these components on a small clinical dataset.

---

## Appendix: Summary Confidence Table

| Component | Novelty claim | Confidence we can defend it | Action |
|---|---|---|---|
| E1: Reliability-Weighted Attention | Transfer novelty (ASD context) | Moderate | Cite Cheng et al. 2020; clarify difference |
| E2: Augmented Bilateral Topology | Contextual novelty (combined formulation) | Moderate | Search gesture/gait lit before submission; qualify if found |
| E3: Bilateral Phase Stream | Domain application; GNN integration possibly novel | Low | Explicitly acknowledge biomechanics prior art; check gait lit |
| E4: Confusable-Pair Margin Loss | Domain-specific instantiation of known technique | High (that it is incremental) | Cite ArcFace/Circle Loss as prior; frame as engineering choice |
| E5: Temperature Calibration | No novelty | High (that it is not novel) | Cite Guo et al. 2017; include for completeness only |

---

*This document is intended for internal use in guiding paper writing. All novelty claims should be verified against a thorough literature review before submission. Honest reporting of null results and dataset discrepancies is a scientific obligation, not a weakness.*
