# Stage 2: Deep Literature and Novelty Review (2021–2026)

**Project:** Autism Therapy Physical Activity Recognition using 2D Skeleton Dynamics  
**Dataset:** Multimodal Autism Spectrum Disorder (MMASD) OpenPose Skeletons (1,374 recordings, 10 therapy classes, 32 group/subject IDs)  
**Date:** March 2026  
**Scope:** Critical analysis of 15 research directions across Skeleton-Based Action Recognition, Graph Neural Networks, Transformer Architectures, Reliability/Uncertainty Estimation, Fine-Grained Kinematics, and Clinical Autism Assessment.

---

## 1. Executive Summary and Literature Scope

The recognition of physical therapy exercises performed by autistic children using body-worn or monocular camera sensors presents unique technical challenges not addressed by mainstream action recognition datasets (e.g., NTU RGB+D 60/120, Kinetics-Skeleton, BABEL). Therapies targeting Autism Spectrum Disorder (ASD)—such as the structured physical activity protocols in MMASD (Li et al., 2021)—involve:
1. **High intra-class motor variability:** Autistic children exhibit heterogeneous motor execution patterns, dynamic dyspraxia, rhythmic stereotypic variations, and fluctuating engagement levels.
2. **Severe kinematic ambiguity between functional exercise pairs:** Specific activity pairs share identical gross postural configurations while differing only in high-frequency harmonic phase (e.g., *Drumming* vs. *Sing and Clap*) or fine spatial trajectory curvature (e.g., *Maracas Shaking* vs. *Maracas Forward Shaking*).
3. **Severe pose estimation noise and tracking dropouts:** Monocular 2D OpenPose keypoints suffer from frequent limb self-occlusion, boundary truncations, and spurious detections, yielding rapid confidence drops ($\hat{c} \to 0$) and unphysical velocity spikes.

While our initial RA-STGT (Reliability-Aware Spatio-Temporal Graph Transformer) introduced joint-level reliability masking and frame-level attention gating to withstand synthetic missingness (degrading only from 0.4137 to 0.3454 at 40% noise, compared to 0.2911 down to 0.1269 for baseline STGT), its overall pooled OOF macro-F1 of 0.4510 demonstrates significant residual failure modes. Specifically, RA-STGT suffers from:
- **Spatial Topology Rigidity:** Standard GAT attention operates strictly over static physical skeletal adjacency, failing to capture dynamic non-physical relational correlations (e.g., inter-wrist distance $\|p_{\text{lwrist}} - p_{\text{rwrist}}\|$ essential to distinguish drumming from clapping).
- **Temporal Saccadic Collapse:** Global temporal attention pooling computes a static convex combination of frame representations, collapsing transient phase reversals and rhythmic frequencies into a stationary kinematic average.
- **Isotropic Logit Separation:** Cross-entropy optimization distributes decision boundaries symmetrically, failing to penalize confusable pairs or condition margins on sample reliability.

This document systematically examines 15 research directions spanning 2021–2026 to contextualize our proposed research contribution within the contemporary literature.

---

## 2. Review of 15 Contemporary Research Directions

### Direction 1: Channel-Topology Refinement and Adaptive Graph Convolution
- **Key References:** Chen et al. (ICCV 2021) *Channel-Wise Topology Refinement Graph Convolution (CTR-GCN)*; Chi et al. (CVPR 2022) *InfoGCN*.
- **Mechanism:** CTR-GCN relaxes fixed physical skeletal adjacency by learning channel-specific correlation matrices $A_k = A + B + C_k$, where $C_k = \phi(X)_k \psi(X)_k^T$. This allows different feature channels to attend to disjoint limb subsets (e.g., one channel linking left wrist to right wrist, another linking ankles to hips).
- **Relevance & Limitations for MMASD:** While CTR-GCN captures non-physical joint dependencies, it relies entirely on full uncorrupted keypoint coordinates. When applied to 2D OpenPose skeletons with low confidence, CTR-GCN generates spurious channel topologies driven by hallucinated joint positions.

### Direction 2: Unified Spatio-Temporal Skate/Partition Transformers
- **Key References:** Hong et al. (CVPR 2024) *SkateFormer: Skeletal-Temporal Attention with Kinetics Embedding*; Plizzari et al. (CVIU 2021) *Spatial-Temporal Transformer (ST-TR)*.
- **Mechanism:** Replaces graph convolutions with factorized self-attention across partition-based skeletal tokens (macro-joints vs. micro-joints) and temporal sub-sequences, achieving state-of-the-art results on NTU-120 with $O(N \cdot T)$ complexity.
- **Relevance & Limitations for MMASD:** SkateFormer requires massive pretraining and uniform temporal sampling. On MMASD's modest sample size (1,374 recordings) and variable sequence lengths, transformer-only architectures rapidly overfit (as verified in our baseline Transformer experiments) and lack explicit reliability calibration mechanisms.

### Direction 3: Hyperbolic and Non-Euclidean Skeletal Representations
- **Key References:** Peng et al. (ECCV 2022) *Hyperbolic Graph Convolutional Networks for Skeleton-Based Action Recognition*; Sun et al. (TPAMI 2024).
- **Mechanism:** Embeds human skeletal joints into Poincaré ball or Lorentz hyperbolic manifolds to naturally represent hierarchical kinematic chains (torso $\to$ shoulder $\to$ elbow $\to$ wrist).
- **Relevance & Limitations for MMASD:** Hyperbolic geometry enforces strict tree-like hierarchical constraints. However, autism therapy activities frequently break pure tree hierarchies through bilateral inter-limb interactions (clapping hands together creates a closed kinematic loop). Furthermore, computing hyperbolic distance with noisy 2D coordinates induces numerical instability near manifold boundaries ($\|x\| \approx 1$).

### Direction 4: Prototype and Metric-Learning Graph Neural Networks
- **Key References:** Zhang et al. (CVPR 2023) *ProtoGCN: Prototype-Guided Graph Convolution for Few-Shot Skeleton Action Recognition*; Snell et al. (NeurIPS 2017) *Prototypical Networks*.
- **Mechanism:** Class representations are parameterized by learnable or cluster-based prototypes in a latent Riemannian or Euclidean feature space. Inference assigns labels based on minimum distance to prototypes, regularized by contrastive margin constraints.
- **Relevance & Limitations for MMASD:** Prototypical learning provides robust inductive bias against severe class imbalance. However, standard ProtoGCN assumes unimodal, Gaussian class distributions. In MMASD, high subject-specific behavioral divergence creates multi-modal intra-class clusters that break standard single-prototype assumptions.

### Direction 5: Self-Supervised Skeletal Pretraining & Masked Pose Modeling
- **Key References:** Mao et al. (CVPR 2022) *Masked Motion Predictors (PST-Transformer)*; Dong et al. (ECCV 2024) *SkeletonMAE*.
- **Mechanism:** Applies masked autoencoding to skeletal joints or temporal windows, forcing the model to reconstruct masked keypoint coordinates before fine-tuning on action labels.
- **Relevance & Limitations for MMASD:** We previously evaluated self-supervised masked autoencoders on MMASD (achieving 0.4750 macro-F1 at 0% mask, but collapsing to 0.2886 ARI on unsupervised clusters). Because 2D OpenPose data contains systemic occlusion (not missing at random), masked autoencoders often learn to reconstruct spurious background artifacts rather than true biomechanical kinematics.

### Direction 6: Confidence-Aware and Reliability-Gated Graph Networks
- **Key References:** Cheng et al. (CVPR 2020) *Decoupling GCN*; Song et al. (ACM MM 2021) *Constructing Dynamic Graphs on Skeleton Sequences*; Stage 1 RA-STGT (2026).
- **Mechanism:** Explicitly incorporates 2D keypoint detection confidence $c_{v,t}$ into message-passing adjacency weights or feature projections, suppressing noisy joints during neighborhood aggregation.
- **Relevance & Limitations for MMASD:** RA-STGT demonstrated that joint-level reliability gating ($r_{v,t} = c_{v,t} \cdot \exp(-\alpha \|\Delta p\|)$) protects against noise. However, RA-STGT applied reliability *independently* to space and time without modeling **relational reliability** (e.g., is the interaction between joint $u$ and joint $v$ reliable?). If either wrist has low confidence, the computed inter-hand relation must be down-weighted proportionally.

### Direction 7: Multi-Stream Kinematic Fusion (Joint, Bone, Velocity, Acceleration)
- **Key References:** Shi et al. (CVPR 2019) *Two-Stream Adaptive Graph Convolutional Networks (2s-AGCN)*; Liu et al. (CVPR 2020) *Disentangling and Unifying Graph Convolutions*.
- **Mechanism:** Decomposes skeletal motion into separate coordinate streams: raw positions $J$, bone vectors $B = p_{v_1} - p_{v_2}$, velocity $\Delta J$, and bone velocity $\Delta B$. Predictions are aggregated via late fusion.
- **Relevance & Limitations for MMASD:** Late multi-stream fusion triples parameter count and training compute. For CPU-constrained execution and small sample regimes, an integrated, single-stream joint-bone-velocity formulation is required.

### Direction 8: Fine-Grained Kinematic Phase and Frequency Analysis
- **Key References:** Li et al. (ICCV 2023) *Frequency-Aware Spatio-Temporal Graph Convolutional Networks*; Wang et al. (TPAMI 2024) *Temporal Phase Decomposition for Repetitive Exercise Analysis*.
- **Mechanism:** Applies 1D Fast Fourier Transform (FFT) or continuous wavelet transforms (CWT) along temporal trajectories, or decomposes motion into rhythmic phase components (acceleration phase, inflection point, deceleration phase).
- **Relevance & Limitations for MMASD:** Directly addresses the **Drumming vs. Clapping** collapse identified in our error analysis. Drumming exhibits alternate out-of-phase vertical wrist oscillations ($\Delta \phi \approx \pi$), whereas Sing-and-Clap exhibits in-phase horizontal wrist convergence ($\Delta \phi \approx 0$). Incorporating explicit phase-frequency relational modeling directly addresses this failure mode.

### Direction 9: Pairwise Inter-Limb Relational Dynamics
- **Key References:** Wu et al. (CVPR 2023) *Relational Graph Attention Networks for Fine-Grained Action Recognition*; Su et al. (ECCV 2022) *Interactive Skeleton Reasoning*.
- **Mechanism:** Augments fixed physical kinematic graphs (17 joints) with dynamic complete bipartite subgraphs between functional end-effectors (left wrist, right wrist, left ankle, right ankle, head).
- **Relevance & Limitations for MMASD:** Crucial for therapy exercises. In MMASD, physical graph edges connect wrist $\to$ elbow $\to$ shoulder $\to$ neck $\to$ shoulder $\to$ elbow $\to$ wrist (a path length of 6 hops). A standard 2-layer GCN cannot pass information between the two hands! Directly introducing bilateral end-effector relational edges allows single-hop reasoning over inter-hand coordination.

### Direction 10: Adaptive Margin and Hard-Pair Contrastive Losses
- **Key References:** Deng et al. (CVPR 2023) *Elastic Margin ArcFace for Deep Metric Learning*; Lin et al. (ICCV 2017) *Focal Loss*; Sun et al. (CVPR 2020) *Circle Loss*.
- **Mechanism:** Enforces dynamic angular or Euclidean margins between specific confusable class pairs in the latent feature space, penalizing false transitions between highly similar activities.
- **Relevance & Limitations for MMASD:** Directly targets the **Maracas Shaking vs. Maracas Forward Shaking** error (41.5% cross-confusion in RA-STGT) and **Drumming vs. Clapping** error (40.2% confusion). Enforcing an explicit class-conditional margin between known confusable clusters prevents the decision boundary from collapsing into the majority class.

### Direction 11: Uncertainty Quantification and Selective Prediction in Clinical AI
- **Key References:** Geifman & El-Yaniv (NeurIPS 2017) *Selective Classification for Deep Neural Networks*; Guo et al. (ICML 2017) *On Calibration of Modern Neural Networks*; Kompa et al. (Nature Communications 2021) *Second Opinion Strategies in Clinical Machine Learning*.
- **Mechanism:** Couples soft classification predictions with formal coverage-risk trade-offs, enabling the model to abstain (triggering clinician review) when confidence falls below an operating threshold $\tau$.
- **Relevance & Limitations for MMASD:** Validated in RA-STGT Stage 1, where filtering uncertain predictions boosted accuracy monotonically from 46.43% to 69.11%. However, RA-STGT's selective prediction relied on isotropic softmax entropy, which was vulnerable to overconfident errors on occluded/noisy frames. Conditioning the rejection threshold on skeletal input reliability is an open research direction.

### Direction 12: Domain Generalization and Subject-Invariant Skeleton Modeling
- **Key References:** Li et al. (CVPR 2021) *Invariant Feature Regularization for Cross-Subject Action Recognition*; Wang et al. (NeurIPS 2022) *Adversarial Subject Disentanglement*.
- **Mechanism:** Employs gradient reversal layers or maximum mean discrepancy (MMD) across subject clusters to strip anthropometric and idiosyncratic stylistic noise from kinematic features.
- **Relevance & Limitations for MMASD:** As proven in our forensic audit, MMASD contains 32 subject/group IDs with severe structural cohort imbalance (e.g., subjects 41004–41143 only perform two static poses). Adversarial domain training requires balanced subject-action distributions; otherwise, the adversarial objective trivially deletes class-discriminative signal.

### Direction 13: Clinical Autism Pose Assessment and Diagnostic Correlates
- **Key References:** Ay et al. (IEEE JBHI 2026) *Automated Motor Assessment in Autism Therapy Using Computer Vision*; Li et al. (ACM MM 2021) *Multimodal Child ASD Dataset (MMASD)*; Rad et al. (Frontiers in Robotics and AI 2022).
- **Mechanism:** Correlates computational kinematic metrics (postural sway, jerk, range of motion, stereotypic frequency) with clinician diagnostic scores (ADOS-2, CARS-2, PEP-3).
- **Relevance & Limitations for MMASD:** Crucial clinical context. As established in our Stage 1 audit, ADOS-2 is a diagnostic tool, not a per-frame ground truth quality label. Action recognition must maintain strict boundary separation between clinical diagnostic scales and physical therapy exercise classification.

### Direction 14: Temporal Saccade and Dynamic Frame Selection
- **Key References:** Zheng et al. (CVPR 2020) *Dynamic Frame Pruning for Skeleton Action Recognition*; Meng et al. (ICCV 2021) *AdaBatch Skeleton Transformer*.
- **Mechanism:** Selectively weights or samples salient temporal sub-intervals (inflection points, peak extension) rather than uniform frame processing.
- **Relevance & Limitations for MMASD:** Applicable to therapy exercises, which consist of alternating cycles of active movement and static recovery. Emphasizing high-acceleration inflection frames improves discrimination between dynamic exercises and resting states.

### Direction 15: Spatial-Temporal Coordinate Derivative Fusion
- **Key References:** Yan et al. (AAAI 2018) *ST-GCN*; Thakkar & Narayanan (Pattern Recognition 2021).
- **Mechanism:** Computes explicit first-order ($\Delta p_t = p_t - p_{t-1}$) and second-order ($\Delta^2 p_t = \Delta p_t - \Delta p_{t-1}$) temporal finite differences at the joint level, concatenated directly into the node feature representation.
- **Relevance & Limitations for MMASD:** Highly lightweight and computation-friendly. Second derivatives (acceleration) capture micro-tremors and rapid directional reversals without adding trainable parameters.

---

## 3. Summary of Research Landscape

The 15 research directions reveal three critical structural insights:
1. **Transformer models and purely spatial GNNs fail in fine-grained therapy domains** because standard skeletal graphs lack inter-limb coordination pathways, and uniform temporal attention washes out oscillatory frequency cues.
2. **Confidence-blind models hallucinate topologies on 2D OpenPose data**, confirming that reliability gating is essential.
3. **Cross-entropy loss with isotropic softmax fails on fine-grained confusable exercise pairs**, proving the necessity of targeted margin regularization.

These findings directly motivate the architecture of our enhanced Stage 2 model: **Phase-Relational & Reliability-Gated Spatio-Temporal Graph Transformer (PR-STGT / RA-STGT v2)**.
