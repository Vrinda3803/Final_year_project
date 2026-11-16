# Stage 2: Controlled Ablation Plan

**Project:** Autism Therapy Physical Activity Recognition using 2D Skeleton Dynamics  
**Date:** March 2026  
**Scope:** Systematic single-factor ablation experiments to rigorously prove the causal contribution of each proposed mechanism in PR-STGT.

---

## 1. Scientific Objective and Ablation Principles

In academic research, an improvement in classification accuracy is unconvincing if it cannot be causally attributed to specific architectural innovations. To defend our contribution before examiners and peer reviewers, we define a controlled single-factor ablation hierarchy from E0 (vanilla STGT) to E5 (full PR-STGT).

Each experiment isolates **exactly one design factor**, holding all other hyperparameters, optimizer settings, learning rates, epochs, random seeds (`seed=42`), and 5-fold cross-validation splits identical.

---

## 2. Controlled Experiment Matrix (E0 to E5)

| Experiment ID | Configuration Name | Physical Graph | Relational Edges ($\mathcal{E}_{\text{rel}}$) | Node Reliability ($r_v$) | Edge Reliability ($r_{uv}$) | Phase Stream ($\Delta \phi_t$) | Margin Loss ($\mathcal{L}_{\text{CPM}}$) | Research Hypothesis Tested |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **E0** | **Baseline STGT** | Yes | No | No | No | No | No | Establishes the performance of confidence-blind spatial GAT + temporal self-attention. |
| **E1** | **RA-STGT v1** | Yes | No | Yes | No | No | No | Validates the impact of single-joint velocity priors and frame reliability gating. |
| **E2** | **PR-STGT (Topology Only)** | Yes | **Yes** | Yes | **Yes** | No | No | Isolates the causal effect of augmented bi-relational end-effector edges (wrists/ankles). |
| **E3** | **PR-STGT (Phase Only)** | Yes | No | Yes | No | **Yes** | No | Isolates the causal effect of the temporal harmonic phase velocity stream ($\Delta \phi_t$). |
| **E4** | **PR-STGT (Margin Only)** | Yes | No | Yes | No | No | **Yes** | Isolates the causal effect of confusable-pair margin regularization ($\mathcal{L}_{\text{CPM}}$). |
| **E5** | **Full PR-STGT (v2)** | Yes | **Yes** | Yes | **Yes** | **Yes** | **Yes** | Evaluates the full synergistic combination of relational topology, phase dynamics, and margin. |

---

## 3. Detailed Hypothesis and Diagnostic Target for Each Ablation

### Experiment E0: Vanilla STGT Baseline (Historical Stage 1)
- **Status:** Completed.
- **Observed Metrics:** Mean Fold Acc: 37.86% ± 10.81%, Mean Fold Macro-F1: 0.2911 ± 0.0791, Pooled OOF Macro-F1: 0.3644.
- **Role:** Lower performance bound. Demonstrates that deep spatio-temporal transformers fail without reliability modeling when trained on noisy 2D OpenPose skeletons.

### Experiment E1: Stage 1 RA-STGT Baseline (Historical Stage 1)
- **Status:** Completed.
- **Observed Metrics:** Mean Fold Acc: 46.47% ± 7.84%, Mean Fold Macro-F1: 0.4137 ± 0.1328, Pooled OOF Macro-F1: 0.4510.
- **Role:** Direct predecessor. Proves that joint-level reliability gating yields +0.1226 macro-F1 over E0 and preserves performance under missingness.

### Experiment E2: Augmented Bi-Relational Topology ($\mathcal{E}_{\text{rel}}$)
- **Modification:** Adds direct bilateral edges between wrists $(10, 9)$ and ankles $(16, 15)$ with relational reliability gating $\tilde{\mathcal{R}}_{uv} = r_u \cdot r_v$. Phase stream and margin loss remain disabled.
- **Specific Hypothesis:** Eliminating the 6-hop path between wrists will allow spatial GAT to directly attend to inter-hand distance, reducing confusion between bilateral and unilateral hand exercises.
- **Diagnostic Success Metric:** A measurable drop in `dr` misclassified as `sac`, and an increase in overall spatial attention sharpness across end-effectors.

### Experiment E3: Temporal Harmonic Phase Velocity Stream ($\Delta \phi_t$)
- **Modification:** Adds the instantaneous vertical velocity correlation stream $\Delta \phi_t = \tanh(10 \cdot \dot{y}_L \dot{y}_R)$ to the temporal attention module. Relational edges and margin loss remain disabled.
- **Specific Hypothesis:** Providing explicit antiphase vs. in-phase temporal velocity signals will resolve the temporal saccadic collapse of rhythmic movements.
- **Diagnostic Success Metric:** Direct reduction of the 138 drumming-clapping errors, specifically elevating `dr` F1 from 0.5816 and `sac` F1 from 0.4107.

### Experiment E4: Targeted Confusable-Pair Margin Loss ($\mathcal{L}_{\text{CPM}}$)
- **Modification:** Adds the confusable-pair margin loss penalty on $\{(\text{dr}, \text{sac}), (\text{ms}, \text{mfs})\}$ to the training objective. Relational edges and phase stream remain disabled.
- **Specific Hypothesis:** Forcing a metric separation margin $m=0.5$ between twin class logits will prevent the optimizer from collapsing into majority-class predictions.
- **Diagnostic Success Metric:** Significant improvement on `ms` (Maracas Shaking, currently 0.3062 F1) and `mfs` (Maracas Forward Shaking, currently 0.2992 F1), reducing cross-confusion below 25%.

### Experiment E5: Full Proposed PR-STGT
- **Modification:** Enables all components simultaneously.
- **Specific Hypothesis:** Synergistic interaction between relational topology (spatial), phase difference (temporal), and margin regularization (objective) will produce the highest overall Macro-F1 (>0.510 pooled OOF) while retaining high calibration (ECE < 0.08) and superior missingness robustness.

---

## 4. Execution Protocol and Resource Budget

- **Folds:** 5-fold GroupKFold (identical to Stage 1).
- **Epochs per Fold:** 40 epochs with EarlyStopping (patience = 12).
- **Batch Size:** 32.
- **Estimated Runtime per Fold on CPU:** ~2.5 minutes.
- **Estimated Total Runtime for E2–E5 (16 runs):** ~40 minutes total CPU compute.
- **Artifacts Saved:**
  - `research_stage2/experiments/checkpoints/` (best fold models)
  - `research_stage2/experiments/logs/` (epoch loss/metric histories)
  - `research_stage2/experiments/predictions/` (out-of-fold CSVs with probabilities and confidences)
