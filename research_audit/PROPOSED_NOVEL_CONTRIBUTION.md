# PROPOSED NOVEL RESEARCH CONTRIBUTION

**Framework Title:** **RA-STGT: Reliability-Aware Spatio-Temporal Graph Transformer with Calibrated Selective Prediction for Autism Therapy Activity Recognition**  
**Role:** Senior AI/ML Research Engineer & Computer Vision Researcher  
**Target Submission / Milestone:** Final-Year BTech Thesis / Academic Conference Paper (e.g., IEEE EMBC, ACM ICMI, or Sensors)  
**Date:** November 13, 2026  

---

## 1. Research Problem

In clinical autism therapy (such as play, dance, or music therapy), tracking children's motor activities provides objective evidence of motor coordination, engagement, and therapeutic progress. Skeleton-based action recognition using Graph Convolutional Networks (GCNs) or Transformers offers a privacy-preserving alternative to raw RGB video by abstracting away identifiable facial and environmental features.

However, skeleton data captured in naturalistic therapy rooms suffer from severe **observation fragility**:
1. **Physical Occlusion:** Children frequently occlude their limbs with therapy objects (e.g., drums, maracas, chairs) or therapist interactions.
2. **Dynamic Pose Estimation Failure:** Off-the-shelf 2D pose estimators (e.g., OpenPose) frequently lose tracking or produce erratic coordinate jitter during fast, stereotypic, or atypical movements.
3. **Overconfident Predictions:** Existing deep models treat all detected coordinates as equally reliable, propagating corrupted spatial features into the temporal backbone and producing confident errors when limbs are missing.

---

## 2. Existing Research Gap

1. **Ay et al. (Sensors 2026, RC-ProtoGCN):** Proposed a training-only margin loss between confusable classes on MMASD 2D skeletons, reaching 89.29% subject-independent accuracy on a curated 1,202-clip subset. However, their model **discards OpenPose detection confidence entirely**, operates exclusively on clean coordinates, and degrades sharply when keypoints are missing.
2. **Li, Bhat, & Barmaki (ACM ICMI 2023):** Integrated OpenPose confidence into a spatial-temporal graph transformer for continuous **interpersonal synchrony estimation** between child-therapist dyads. However, they did not address **discrete activity recognition**, multi-class discrimination, frame-level temporal attention gating, or predictive uncertainty.
3. **Standard Occlusion GCNs (e.g., ROC-GCN, RA-GCN):** Rely on synthetic Bernoulli dropout or assume left-right bilateral anatomical symmetry to compensate for missing joints. In autism therapy, stereotypic movements are frequently **asymmetric** (e.g., unilateral hand-flapping, asymmetric reaching), violating bilateral symmetry assumptions.
4. **Our Historical Baseline Finding:** Simply concatenating confidence values or quality masks to feature vectors or GCN nodes (as in `experiments/mmasd_openpose_reliability_features.py` and `mmasd_openpose_quality_graph.py`) failed to improve performance (34.88% – 70.24% accuracy) because un-gated features allow noise to propagate.

---

## 3. Central Research Question

> **Can a spatial-temporal architecture learn to dynamically attenuate noisy joint observations during message passing, re-weight temporal frame importance based on physical and detector reliability, and provide calibrated uncertainty estimates to enable selective, safe predictions in clinical autism activity monitoring?**

---

## 4. Proposed Methodology & Mathematical Formulation

The proposed **RA-STGT** framework operates on a sequence of 2D skeleton frames $\mathbf{X} \in \mathbb{R}^{T \times V \times C}$ and their associated measured detection confidences $\mathbf{Q} \in \mathbb{R}^{T \times V}$, where $T=64$ frames, $V=25$ OpenPose BODY_25 joints, and $C=4$ channels (normalized $x, y$ coordinates and first-order velocities $\Delta x, \Delta y$).

```
                    +------------------------------------+
                    | Input: Skeleton (X) & Confidence (Q)|
                    +-----------------+------------------+
                                      |
                                      v
       +--------------------------------------------------------------+
       | Stage 1: Physical & Kinematic Reliability Estimation         |
       |  - PAF Detector Confidence: c_{v,t} in [0, 1]                |
       |  - Velocity Plausibility Gate: v_{v,t} in [0, 1]             |
       |  -> Combined Joint Reliability: r_{v,t} in [0, 1]            |
       +------------------------------+-------------------------------+
                                      |
                                      v
       +--------------------------------------------------------------+
       | Stage 2: Spatial Reliability-Gated Attention (SR-GAT)        |
       |  - Physical Bone Adjacency + Adaptive Spatial Attention     |
       |  - Attenuated Message Passing: A~_{i,j,t} * r_{j,t}          |
       |  - Self-loop preservation: prevents node feature collapse     |
       +------------------------------+-------------------------------+
                                      |
                                      v
       +--------------------------------------------------------------+
       | Stage 3: Reliability-Aware Temporal Attention (RATA)         |
       |  - Frame-Level Reliability Summary: R_t = mean_v(r_{v,t})    |
       |  - Temporal Self-Attention modulated by R_t                  |
       |  - Downweights occluded frames without breaking sequence     |
       +------------------------------+-------------------------------+
                                      |
                                      v
       +--------------------------------------------------------------+
       | Stage 4: Calibrated Selective Classification                 |
       |  - Class Logits & Temperature Scaling Calibration (ECE)      |
       |  - Selective Prediction: Predict only when Confidence >= tau |
       |  - Rejection / Abstention Rate for Clinical Safety           |
       +--------------------------------------------------------------+
```

### Module 1: Joint-Level Kinematic Reliability Estimation

For joint $v$ at time $t$, we combine the measured detector confidence $c_{v,t} \in [0, 1]$ with an anatomical velocity plausibility prior $v_{v,t} \in [0, 1]$:

$$v_{v,t} = \exp\left( - \frac{\|\Delta \mathbf{p}_{v,t}\|^2}{2 \sigma_v^2} \right)$$

where $\Delta \mathbf{p}_{v,t} = \mathbf{p}_{v,t} - \mathbf{p}_{v,t-1}$ is the inter-frame coordinate displacement, and $\sigma_v$ is the empirical velocity variance of joint $v$ estimated from the training set. The joint reliability score $r_{v,t}$ is:

$$r_{v,t} = c_{v,t} \cdot v_{v,t}$$

When a joint suffers from false-positive detection jitter or detector loss, $r_{v,t} \to 0$. When tracking is smooth and confident, $r_{v,t} \to 1$.

---

### Module 2: Spatial Reliability-Gated Graph Attention (SR-GAT)

In standard spatial graph attention (GAT), node $i$ updates its hidden state $\mathbf{h}_{i,t}$ by aggregating features from all neighbors $j \in \mathcal{N}(i)$:

$$\mathbf{h}_{i,t}' = \sum_{j \in \mathcal{N}(i)} \alpha_{ij,t} \mathbf{W}_v \mathbf{h}_{j,t}$$

If neighbor $j$ is occluded or corrupted, $\mathbf{h}_{j,t}$ contains noise, polluting node $i$. In **SR-GAT**, we modulate the attention coefficients $\alpha_{ij,t}$ by the source node reliability $r_{j,t}$:

$$\tilde{\alpha}_{ij,t} = \frac{\alpha_{ij,t} \cdot r_{j,t}}{\sum_{k \in \mathcal{N}(i)} \alpha_{ik,t} \cdot r_{k,t} + \epsilon}$$

To prevent feature collapse when all neighbors of a limb are occluded, we ensure the self-loop ($j = i$) retains a minimum identity path:

$$\mathbf{h}_{i,t}' = \mathbf{W}_{self} \mathbf{h}_{i,t} + \sum_{j \in \mathcal{N}(i), j \neq i} \tilde{\alpha}_{ij,t} \mathbf{W}_v \mathbf{h}_{j,t}$$

---

### Module 3: Reliability-Aware Temporal Attention (RATA)

After spatial aggregation, each frame $t$ is represented by a pooled spatial token $\mathbf{z}_t \in \mathbb{R}^D$ and a frame-level reliability scalar:

$$R_t = \frac{1}{V} \sum_{v=1}^V r_{v,t} \quad \in [0, 1]$$

Standard temporal self-attention computes query-key affinity matrices:

$$\mathbf{A}_{t, \tau}^{temp} = \text{softmax}\left( \frac{\mathbf{q}_t \mathbf{k}_\tau^\top}{\sqrt{d_k}} \right)$$

In **RATA**, we modulate temporal attention by the target frame's reliability $R_\tau$, ensuring that corrupted or completely occluded frames do not dominate the sequence representation:

$$\tilde{\mathbf{A}}_{t, \tau}^{temp} = \text{softmax}\left( \frac{\mathbf{q}_t \mathbf{k}_\tau^\top}{\sqrt{d_k}} + \log(R_\tau + \epsilon) \right)$$

When frame $\tau$ has near-zero skeletal visibility ($R_\tau \to 0$), $\log(R_\tau + \epsilon) \to -\infty$, naturally suppressing attention to frame $\tau$ while allowing clean adjacent frames to carry the motion dynamics.

---

### Module 4: Uncertainty Estimation & Calibrated Selective Prediction

Standard softmax probabilities $\hat{p}_k = \frac{\exp(z_k)}{\sum_j \exp(z_j)}$ are notoriously overconfident. We introduce post-hoc **temperature scaling** fitted on the inner validation fold:

$$\hat{p}_k(T) = \frac{\exp(z_k / T)}{\sum_j \exp(z_j / T)}$$

where optimal temperature $T^*$ minimizes Negative Log-Likelihood (NLL) on held-out validation subjects.

For clinical utility, we evaluate **Selective Classification** with abstention threshold $\tau$:

$$\hat{y}(x) = \begin{cases} \arg\max_k \hat{p}_k(T^*), & \text{if } \max_k \hat{p}_k(T^*) \geq \tau \\ \text{ABSTAIN (Unreliable Observation)}, & \text{otherwise} \end{cases}$$

This allows reporting **Risk-Coverage Curves**: as the model abstains on ambiguous or occluded clips (coverage $100\% \to 80\%$), does accuracy and macro-F1 rise towards 90%+?

---

## 5. Architectural Differences vs Closest Published Work

| Architectural Dimension | Ay et al. (Sensors 2026) | Li et al. (ACM ICMI 2023) | Our Proposed RA-STGT |
|---|---|---|---|
| **Target Task** | 10-Class Activity Recognition | Continuous Dyadic Synchrony | 10-Class Activity Recognition |
| **Backbone Architecture** | ProtoGCN (ST-GCN) | Spatial-Temporal Graph Transformer | Reliability-Gated STGT |
| **Handling of Pose Confidence** | **Discarded** ($x, y$ coordinates only) | Scalar spatial attention bias | **Dual-stage:** Spatial Gating + Temporal Attention Modulation |
| **Occlusion Robustness** | None (degrades on missing joints) | Evaluated on dyad synchrony | Explicit joint & frame gating; evaluated under real & synthetic occlusion |
| **Uncertainty & Calibration** | None (uncalibrated softmax) | None (deterministic regression) | **Temperature-calibrated logits, ECE, Brier Score, and Selective Prediction** |
| **Inference Efficiency** | Lightweight (ProtoGCN) | Heavy (Full Dyadic Transformer) | Lightweight single-actor STGT (~1.2M parameters) |

---

## 6. What is Genuinely New vs What is Not Novel

### What IS Genuinely New
1. **First Reliability-Aware Graph Transformer for Autism Activity Recognition:** Formulating joint- and frame-level reliability gating specifically for therapeutic activity classification on MMASD.
2. **Dual-Stage Attenuation (SR-GAT + RATA):** A principled formulation that prevents occluded joints from polluting spatial neighbors and suppresses corrupted frames in temporal sequence attention without discarding sequence order.
3. **Calibrated Selective Prediction for Clinical Safety:** Moving beyond raw accuracy to evaluate calibration error (ECE, Brier score) and risk-coverage trade-offs, enabling the model to safely flag unclassifiable observations for clinical review.
4. **Empirical Robustness Benchmarking:** First study on MMASD to systematically compare clean performance against controlled missing-joint levels (0%, 20%, 40%) using true detection confidence.

### What is NOT Novel (Explicit Non-Claims)
1. We did not invent Graph Transformers (Vaswani et al., Dwivedi et al.).
2. We did not invent OpenPose confidence scores (standard output of OpenPose).
3. We did not create MMASD (created by Li, Bhat, & Barmaki).
4. We do not claim an autonomous diagnostic tool (activity recognition is an observational aid, not a diagnostic test).

---

## 7. Experimental Design & Ablation Protocol

### 7.1 Dataset & Subject-Independent Partitioning

* **Data Source:** MMASD 2D OpenPose cache (`data/processed/mmasd_openpose/mmasd_openpose_64frames.npz`).
* **Samples & Classes:** 1,374 recordings across 10 classes (`as`, `bs`, `ce`, `dr`, `mfs`, `ms`, `sac`, `sq`, `tr`, `tw`).
* **Evaluation Scheme:** Strict 5-fold `GroupKFold` grouped by the 32 stored subject IDs.
  * Outer folds: 5 folds (each fold evaluates on ~6–7 completely unseen subjects).
  * Inner validation: 18% of training subjects reserved for early stopping and temperature scaling calibration.
  * Zero subject overlap between training, validation, and test partitions.
* **Repetition:** 3 random seeds (Seed 42, 100, 2026) to report Mean $\pm$ Standard Deviation and confidence intervals.

---

### 7.2 Core Baselines & Incremental Ablation Matrix

Every component is introduced and ablated individually under the **exact same 5-fold subject splits**:

| Model ID | Architecture / Ablation Description | Purpose of Experiment |
|---|---|---|
| **B1** | Historical 11-Class Feature Ensemble (`mmasd_advanced_90.py`) | Contextual reference for historical tree ensemble (82.87% acc) |
| **B2** | 2D OpenPose Nested RBF Kernel Ridge (`mmasd_openpose_rbf_kernel.py`) | Strongest local 2D baseline (71.69% mean acc / 0.6920 OOF F1) |
| **B3** | Standard ST-GCN Baseline (Yan et al. 2018 / Ay et al. base) | Standard graph convolution baseline without reliability |
| **B4** | Standard Spatial-Temporal Graph Transformer (Uncertainty-Blind) | Establishes the transformer backbone without reliability modules |
| **M1** | STGT + Naive Confidence Concatenation (Feature Concat) | Tests whether simple input concatenation is sufficient (Hypothesis: No) |
| **M2** | STGT + Spatial Reliability Gating (SR-GAT only) | Measures the isolated contribution of spatial message attenuation |
| **M3** | STGT + Reliability-Aware Temporal Attention (RATA only) | Measures the isolated contribution of frame-level temporal attention modulation |
| **M4 (Full)** | **RA-STGT (SR-GAT + RATA Dual Stage)** | Full proposed framework |
| **M5** | RA-STGT + Calibrated Temperature Scaling & Selective Prediction | Full framework with risk-coverage and abstention evaluation |

---

### 7.3 Robustness Evaluation under Occlusion & Noise

To prove that reliability-aware learning provides tangible robustness benefits, models B3, B4, and M4 will be evaluated under:
1. **Clean Input:** Raw detected keypoints and confidence.
2. **Controlled Synthetic Joint Missingness:** 10%, 20%, 30%, and 40% random joint dropout (confidence set to 0, coordinates zeroed).
3. **Controlled Synthetic Limb Occlusion:** Completely dropping entire arm or leg chains (simulating occlusion by drums/props).
4. **Natural High-Noise Subset:** Subsetting clips where mean OpenPose confidence is in the lowest quartile ($Q < 0.50$).

---

## 8. Expected Risks & Feasibility for a 3-Member BTech Team

| Risk Factor | Severity | Mitigation Strategy |
|---|---|---|
| **GPU / Computational Budget** | Low | The 2D OpenPose cache is compact (36.5 MB; 1,374 sequences). Training a 5-fold RA-STGT takes < 15 minutes per seed on a standard GPU or < 45 minutes on modern CPU. |
| **Class Imbalance (Drumming has 343 clips)** | Medium | Use balanced class-weighted focal cross-entropy loss and report macro-F1 and per-class recall alongside overall accuracy. |
| **Risk of Graph Transformer Overfitting** | Medium | Use compact embedding dimensions ($D=64$, 2 spatial heads, 2 temporal heads), dropout (0.3), and strict early stopping on validation subjects. |
| **Risk that Model Does Not Exceed Tree Ensemble** | Low | If the neural model scores ~75–80% (comparable to or slightly below tree ensembles on clean data), its primary novel contribution is demonstrated via **occlusion robustness (+15% higher F1 under 40% missingness)** and **predictive calibration (reducing ECE from 0.18 to < 0.04)**. In research, robustness and calibration are first-class contributions. |
| **Work Distribution for 3-Member Team** | Low | **Member 1:** Reliability estimation and spatial SR-GAT module.<br>**Member 2:** Temporal RATA module and baseline benchmarking.<br>**Member 3:** Calibration, ECE evaluation, and selective prediction curves. |
