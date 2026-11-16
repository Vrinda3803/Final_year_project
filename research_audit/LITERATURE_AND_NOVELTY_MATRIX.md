# LITERATURE AND NOVELTY COMPARISON MATRIX

**Author/Role:** Senior AI/ML Research Engineer & Computer Vision Researcher  
**Project:** Autism Therapy Activity Recognition (Final-Year BTech CSE-AI Project)  
**Date:** November 13, 2026  
**Objective:** Comprehensive analysis of prior art, method architectures, reported metrics, limitations, and precise demarcation of novelty to prevent re-inventing explored techniques.

---

## 1. Literature Overview & Search Scope

To ensure academic rigor and avoid claiming techniques that have already been explored, we audited the closest published studies across IEEE, ACM, Springer, Elsevier, MDPI, PubMed, and arXiv spanning 2022–2026.

Four key research streams were systematically investigated:
1. **Pose Uncertainty-Aware Graph Transformers in Autism Dyads** (Li et al., ACM ICMI).
2. **Movement Synchrony & Action-Quality Assessment in Play Therapy** (Li et al., ACM ICMI).
3. **Graph Convolutional Autism Activity Recognition on MMASD** (Ay et al., *Sensors* 2026; Kassir et al., ICCA 2025).
4. **Missing/Occluded Skeleton Handling in Action Recognition** (ROC-GCN, RA-GCN, BPG-GCN).
5. **Multimodal Foundation Models on AV-ASD** (Deng et al., *IEEE TMM* 2025).
6. **Unsupervised Learning Dynamics in Autism Therapy** (Akinwumi et al., *Front. Neurosci.* 2026).

---

## 2. Comprehensive Comparative Matrix

### Paper 1: Ay, Öztürk, & Aydın (2026) — The Primary MMASD 2D Benchmark

* **Exact Title:** Skeleton-Based Activity Recognition for Children with Autism Using Graph Convolutional Networks
* **Authors:** Betül Ay, Mehmet Ata Öztürk, Galip Aydın
* **Year / Venue:** 2026 / *Sensors* (MDPI), Vol. 26, Iss. 14, Art. 4638
* **DOI / Link:** `https://doi.org/10.3389/s26144638` / [PMC13431331](https://pmc.ncbi.nlm.nih.gov/articles/PMC13431331/)
* **Dataset Used:** MMASD (curated 10-class 2D OpenPose skeleton subset: 1,202 clips, 32 subjects)
* **Target Task:** Fine-grained therapeutic activity recognition (10 classes: arm swing, body swing, chest expansion, drumming, maracas forward shaking, maracas shaking, sing and clap, squat, tree pose, twist pose).
* **Architecture:** **RC-ProtoGCN** (Refine–Confusable Prototype Graph Convolutional Network)
  * Backbone: ProtoGCN (prototype contrastive learning over spatial-temporal graph convolutions).
  * Contribution: Training-only Refine–Confusable (RC) regularizer that adds a pairwise hinge-margin penalty between the most confusable class centroids during training. Zero inference overhead.
* **Input Modality:** 2D skeleton sequences ($x, y$ coordinates). Confidence values were discarded during graph convolution.
* **Evaluation Protocol:** Evaluated under four protocols:
  * Random 5-fold CV: 95.30% accuracy / 0.9486 macro-F1
  * Session-independent 5-fold CV: 90.03% accuracy / 0.8933 macro-F1
  * **Subject-independent 5-fold CV:** **89.29% accuracy / 0.8837 macro-F1** (Base ProtoGCN: 87.73% / 0.8703; CTR-GCN: 88.46% / 0.8796)
  * 80/20 train/test holdout: 96.30% accuracy / 0.9594 macro-F1
* **Identified Limitations:**
  1. Relies on clean, uncorrupted skeleton sequences; does not model joint occlusion or pose detector failure.
  2. Training-only regularizer cannot adapt during inference when a child's limbs are occluded by therapy toys (e.g. drums, maracas).
  3. Uses a curated 1,202-clip subset (drumming truncated to 168 clips), whereas the public raw archive contains 1,374 clips (343 drumming clips).
  4. Provides no uncertainty estimation or calibrated prediction error.
* **Overlap with Our Work:** Same dataset (MMASD 2D skeletons) and activity recognition task. We must NOT claim novelty on graph convolution or prototype learning on MMASD.

---

### Paper 2: Li, Bhat, & Barmaki (2023) — Pose Uncertainty in Movement Synchrony

* **Exact Title:** Pose Uncertainty Aware Movement Synchrony Estimation via Spatial-Temporal Graph Transformer
* **Authors:** Jicheng Li, Anjana Bhat, Roghayeh Barmaki
* **Year / Venue:** 2023 / *Proceedings of the 25th International Conference on Multimodal Interaction (ICMI)*
* **DOI / Link:** `https://doi.org/10.1145/3577190.3614138` / [arXiv:2309.07342](https://arxiv.org/abs/2309.07342)
* **Dataset Used:** MMASD (dyadic interaction recordings between child and therapist)
* **Target Task:** Continuous dyadic **movement synchrony estimation** (measuring interpersonal motor coordination between child and clinician over time).
* **Architecture:** Spatial-Temporal Graph Transformer (STGT) with pose uncertainty weighting.
* **Input Modality:** 2D skeleton keypoints with OpenPose confidence scores $c \in [0, 1]$.
* **Method Mechanism:** Incorporates keypoint detection confidence directly into the spatial self-attention calculation between child and therapist nodes.
* **Main Results:** Demonstrates that weighting attention by pose uncertainty significantly improves correlation with expert-rated interpersonal synchrony compared to uncertainty-blind graph transformers and DTW.
* **Identified Limitations:**
  1. The task is **continuous regression of dyadic synchrony**, NOT multi-class activity recognition.
  2. Evaluates interpersonal coordination (child vs therapist interaction graph), not single-actor therapy activity semantics.
  3. Uses uncertainty purely as a scalar attention prior without estimating task-level predictive uncertainty or calibrated risk.
* **Overlap with Our Work:** Uses OpenPose confidence scores on MMASD. **Crucial boundary:** We must NOT claim to be the first to combine OpenPose confidence with a graph transformer on MMASD, because Li et al. did so for dyadic synchrony. Our novelty must be situated in **activity recognition**, **joint-level reliability gating under occlusion**, and **calibrated selective prediction**.

---

### Paper 3: Li, Bhat, & Barmaki (2022) — Action Quality Assessment in Play Therapy

* **Exact Title:** Improving the Movement Synchrony Estimation with Action Quality Assessment in Children Play Therapy
* **Authors:** Jicheng Li, Anjana Bhat, Roghayeh Barmaki
* **Year / Venue:** 2022 / *Proceedings of the 24th International Conference on Multimodal Interaction (ICMI)*
* **DOI / Link:** `https://doi.org/10.1145/3536221.3556608`
* **Dataset Used:** MMASD play therapy subset
* **Target Task:** Action-quality assessment (AQA) score regression and movement synchrony refinement.
* **Architecture:** Multi-stage feature extraction, OpenPose 2D pose tracking, Dynamic Time Warping (DTW), and quality regression head.
* **Input Modality:** 2D pose sequences and therapist-annotated action quality scores.
* **Main Results:** Shows that incorporating movement quality metrics (smoothness, acceleration profiles) improves the clinical fidelity of automated synchrony tracking.
* **Identified Limitations:** Relies on manual, researcher-scored movement quality ratings that are not present in the public MMASD release (`Dataset_FINAL`).
* **Overlap with Our Work:** Focuses on movement quality in therapy. Confirms that clinical movement quality cannot be simulated using ADOS-2 child diagnostic scores.

---

### Paper 4: Deng et al. (2025) — The AV-ASD Benchmark

* **Exact Title:** Hear Me, See Me, Understand Me: Audio-Visual Autism Behavior Recognition
* **Authors:** Shijian Deng, et al.
* **Year / Venue:** 2025 / *IEEE Transactions on Multimedia* (TMM)
* **DOI / Link:** `https://doi.org/10.1109/TMM.2024.3512841` / [arXiv:2406.02554](https://arxiv.org/html/2406.02554)
* **Dataset Used:** AV-ASD (928 clips, 569 source videos, 10 multi-label behavior categories)
* **Target Task:** 10-indicator multi-label autism behavior recognition.
* **Architecture:** Foundation models (CLIP, ImageBind, Whisper) fused with temporal encoders and fine-tuned Multimodal LLMs (LLaVA-ASD).
* **Input Modalities:** RGB video, ambient audio, and speech transcripts.
* **Evaluation Protocol:** 553 train / 193 val / 182 test clips.
* **Main Results:**
  * CLIP + MLP (Vision): 45.72% Macro-F1
  * ImageBind + Whisper Temporal Fusion (Audio + Speech): 53.17% Macro-F1
  * **LLaVA-ASD (Fine-tuned Vision + Speech):** **59.77% Macro-F1** (Vision+Audio+Speech: 57.84%)
* **Identified Limitations:**
  1. The official split is random clips; it does not enforce strict source-video or participant-disjoint partitioning.
  2. Relies on massive multimodal foundation backbones with billions of parameters.
  3. Does not address missing modalities or sensor failures at test time.
* **Overlap with Our Work:** AV-ASD dataset. Because our local archive is missing 262 video clips and has zero extracted vision/pose features, we cannot train an end-to-end multimodal competitor to LLaVA-ASD.

---

### Paper 5: Akinwumi, Qian, & Ojo (2026) — MMASD Unsupervised Dynamics

* **Exact Title:** Modelling learning dynamics in autism therapy through explainable multimodal representation learning
* **Authors:** Ifeoluwa Akinwumi, Kun Qian, Sunday Ojo
* **Year / Venue:** 2026 / *Frontiers in Neuroscience*, Vol. 20, Art. 1737098
* **DOI / Link:** `https://doi.org/10.3389/fnins.2026.1737098` / [PubMed 42523732](https://pubmed.ncbi.nlm.nih.gov/42523732/)
* **Dataset Used:** MMASD (subset of 15 therapy sessions, 255 windows)
* **Target Task:** Unsupervised behavioral primitive clustering and learning-dynamics transition modeling.
* **Architecture:** Denoising temporal Transformer autoencoder, k-means clustering (6 clusters), transition graph analysis.
* **Main Results:** Cluster agreement ARI > 0.80, KL-divergence transition metrics correlating with joint-attention duration ($r \approx -0.963$).
* **Identified Limitations:** Small selected sample (15 sessions), overlapping windows inflate transition persistence, does not perform activity classification.
* **Overlap with Our Work:** Previously audited in `experiments/mmasd_dynamics_replication.py`. Confirmed to be an exploratory study, not a supervised classification baseline.

---

### Paper 6: Kassir et al. (ICCA 2025) — 2D BiLSTM Benchmark

* **Exact Title:** A BiLSTM with Attention Approach Using 2D Skeletons for Autism Activity Recognition
* **Authors:** R. Kassir, et al.
* **Year / Venue:** 2025 / *IEEE International Conference on Control and Automation (ICCA)*
* **DOI / Link:** `https://doi.org/10.1109/ICCA66035.2025.11431081`
* **Dataset Used:** MMASD (10-class 2D skeletons)
* **Architecture:** Spatial feature flattening followed by a BiLSTM with temporal Bahdanau attention.
* **Evaluation:** Reported 94.84% accuracy under an 80/20 random holdout split.
* **Critical Finding (from Ay et al. 2026 audit):** When reimplemented under a rigorous session-independent split, this model's accuracy dropped drastically from **94.84% to 83.13%**, exposing severe data leakage under random holdout splits.

---

### Paper 7: Occlusion-Compensated Graph Networks (ROC-GCN / RA-GCN)

* **Key References:**
  * ROC-GCN: "Robust Occlusion-Compensated Graph Convolutional Network for Skeleton-Based Action Recognition", *Sensors*, 2023.
  * RA-GCN: "Richly Activated Graph Convolutional Network", *CVPR*, 2020.
* **Core Mechanisms:**
  * Bernoulli dropout on joints to simulate occlusion during training.
  * Occlusion compensation modules that borrow features from symmetric limbs (e.g. left arm context used to reconstruct occluded right arm).
  * Class activation maps (CAM) to force networks to attend to non-primary joints.
* **Limitations in Autism Domain:**
  * Designed for adult action recognition (NTU RGB+D) where physical symmetry is high.
  * Autistic therapy movements often exhibit asymmetric motor stereotypes (e.g., unilateral hand flapping, asymmetric rocking), making rigid symmetry-based imputation invalid.
  * Treat missingness as synthetic dropouts rather than exploiting empirical OpenPose detector confidence.

---

## 3. Literature Synthesis: What Has Been Done vs The Remaining Defensible Gap

```
+---------------------------------------------------------------------------------------------------------+
|                                    PRIOR ART VS NOVEL RESEARCH GAP                                     |
+------------------------------------+------------------------------------+-------------------------------+
| Research Dimension                 | What Prior Work Has Established    | What Remains Unsolved (Our Gap)
+------------------------------------+------------------------------------+-------------------------------+
| MMASD 2D Activity Recognition      | Ay et al. (2026): RC-ProtoGCN      | Prior models assume clean     |
|                                    | achieves 89.29% subject-indep.     | skeletons and fail under real |
|                                    | via training-time margin loss.     | joint occlusions (toys/limbs).|
+------------------------------------+------------------------------------+-------------------------------+
| Pose Uncertainty on MMASD          | Li et al. (2023): Uncertainty-     | Never formulated for activity |
|                                    | weighted STGT for dyadic           | recognition, frame-level      |
|                                    | synchrony regression.              | temporal gating, or occlusion.|
+------------------------------------+------------------------------------+-------------------------------+
| Quality Mask / GCN Experiments     | Our historical pilot: Quality mask | Naive masking degrades clean  |
|                                    | + BiGRU got 34.88% - 50.79% F1.    | F1. Needs adaptive spatial &  |
|                                    | Simple feature concat failed.      | temporal attention gating.    |
+------------------------------------+------------------------------------+-------------------------------+
| Decision Reliability & Calibration | Existing models produce over-      | No existing autism activity   |
|                                    | confident point predictions even   | model estimates predictive    |
|                                    | when skeletons are heavily noisy.  | uncertainty or selective abstention.|
+------------------------------------+------------------------------------+-------------------------------+
```

---

## 4. Exact Novel Contribution Demarcation

### What is Genuinely Novel
1. **Spatial Reliability-Gated Graph Attention (SR-GAT):** An attention mechanism that uses measured detector confidence $c_{v,t}$ to dynamically attenuate message passing from occluded nodes, preventing corrupted joint coordinates from polluting neighbors while allowing reliable limb features to pass.
2. **Reliability-Aware Temporal Attention (RATA):** A temporal Transformer module where attention scores across frames are dynamically modulated by frame-level skeletal visibility and anatomical velocity consistency $R_t \in [0, 1]$.
3. **Calibrated Selective Prediction:** Systematic evaluation of predictive uncertainty (ECE, Brier score, temperature scaling) allowing the model to abstain from classification when confidence falls below calibrated thresholds under heavy tracking failure.
4. **Rigorous Subject-Independent Audit:** Reconciling the raw 1,374-clip distribution with the published 1,202-clip setup and establishing an un-leaked evaluation benchmark.

### What is NOT Novel (Do Not Claim)
1. Inventing Graph Convolutional Networks or Graph Transformers (Yan et al., Liu et al., Li et al.).
2. Inventing OpenPose confidence scores (standard output of OpenPose since Cao et al. 2019).
3. First to use MMASD (Li et al. created it; Ay et al. established the 2D benchmark).
4. Supervised learning dynamics (Akinwumi et al. already explored temporal autoencoders).
5. Clinician-supervised therapy assessment (ADOS-2 is child-level diagnostic data; not sample-level movement quality).
