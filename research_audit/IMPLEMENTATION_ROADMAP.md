# INCREMENTAL IMPLEMENTATION ROADMAP

**Project:** Reliability-Aware Temporal Learning for Autism Therapy Activity Recognition (RA-STGT)  
**Role:** Senior AI/ML Research Engineer & Computer Vision Researcher  
**Target Audience:** Final-Year BTech CSE (AI) 3-Member Student Team  
**Date:** November 13, 2026  
**Implementation Rule:** No code implementation shall commence until the research audit and contribution proposal have been reviewed and approved by the team. All new code must reside strictly in an isolated directory.

---

## 1. Directory Structure for Implementation

To guarantee the complete preservation of all historical experiments and existing scripts, all novel code will be implemented in a dedicated directory:

```
Autism_Project/
|-- novel_contribution/                     <-- ALL NEW IMPLEMENTATIONS LIVE HERE
|   |-- configs/                            <-- YAML/JSON experiment configurations
|   |   |-- base_stgt.json
|   |   |-- ra_stgt_full.json
|   |   |-- ablations_config.json
|   |-- data_processing/                    <-- Data loading, normalization, subject split
|   |   |-- __init__.py
|   |   |-- dataset_loader.py               <-- PyTorch Dataset & subject-grouped CV split
|   |   |-- pose_transforms.py              <-- Normalization & coordinate centering
|   |-- reliability_estimation/             <-- Joint & frame reliability computation
|   |   |-- __init__.py
|   |   |-- detector_confidence.py          <-- OpenPose PAF confidence extractor
|   |   |-- velocity_prior.py               <-- Kinematic velocity plausibility gating
|   |-- models/                             <-- Neural network architectures
|   |   |-- __init__.py
|   |   |-- spatial_gat.py                  <-- Standard spatial GAT & SR-GAT module
|   |   |-- temporal_attention.py           <-- Standard temporal attention & RATA module
|   |   |-- ra_stgt.py                      <-- Full combined RA-STGT model class
|   |-- uncertainty_estimation/             <-- Calibration & selective classification
|   |   |-- __init__.py
|   |   |-- temperature_scaling.py          <-- Post-hoc NLL temperature scaling
|   |   |-- calibration_metrics.py          <-- ECE, Brier score, Reliability Diagrams
|   |   |-- selective_prediction.py         <-- Risk-coverage curves & abstention logic
|   |-- training/                           <-- Training loops and optimization
|   |   |-- __init__.py
|   |   |-- trainer.py                      <-- Fold trainer with strict validation & early stopping
|   |   |-- loss_functions.py               <-- Class-weighted focal cross-entropy
|   |-- evaluation/                         <-- Metrics, test evaluation, robustness
|   |   |-- __init__.py
|   |   |-- evaluate_folds.py               <-- Fold-level and pooled OOF metric evaluation
|   |   |-- robustness_benchmarking.py      <-- Controlled missingness (0-40%) stress test
|   |-- ablations/                          <-- Controlled single-factor ablation runners
|   |   |-- run_spatial_ablation.py
|   |   |-- run_temporal_ablation.py
|   |   |-- run_calibration_ablation.py
|   |-- results/                            <-- Experiment outputs, JSONs, checkpoints
|   |   |-- checkpoints/                    <-- Saved model weights per fold
|   |   |-- metrics/                        <-- JSON metric logs and CSV prediction logs
|   |-- visualizations/                     <-- Publication-ready plots
|   |   |-- plot_reliability_maps.py        <-- Spatial-temporal attention vs confidence heatmaps
|   |   |-- plot_robustness_curves.py       <-- Accuracy vs missingness percentage curves
|   |   |-- plot_calibration_curves.py      <-- Reliability diagrams (ECE) & Risk-Coverage
|   |-- documentation/                      <-- Method specifications & user guide
```

---

## 2. Phased Implementation Milestones

```
+---------------------------------------------------------------------------------------------------------+
|                                    IMPLEMENTATION TIMELINE & MILESTONES                                 |
+---------+----------------------------------------------+---------------+--------------------------------+
| Phase   | Milestone Goal                               | Estimated Time| Primary Deliverable            |
+---------+----------------------------------------------+---------------+--------------------------------+
| Phase 0 | Audit Approval & Environment Setup           | Day 1         | Research Audit Sign-off        |
| Phase 1 | Data Pipeline & Zero-Leakage Split Lock      | Day 2         | Verified DatasetLoader + Folds |
| Phase 2 | Reliability Extraction & Kinematic Priors    | Day 3         | Reliability Modules Unit Tests |
| Phase 3 | RA-STGT Core Architecture Implementation     | Days 4 - 5    | SR-GAT & RATA Modules          |
| Phase 4 | Clean Benchmarking & Single-Factor Ablations | Days 6 - 7    | 5-Fold Comparison vs Baselines |
| Phase 5 | Uncertainty, Calibration (ECE) & Selection   | Days 8 - 9    | Calibrated Selective Classifier|
| Phase 6 | Occlusion Robustness Stress Testing (0-40%)  | Day 10        | Robustness Curves & Stress Logs|
| Phase 7 | AV-ASD Forensic Audit & Case Study Summary   | Day 11        | Standalone AV-ASD Report       |
| Phase 8 | Paper Figures, Viva Deck & Final Artifacts   | Days 12 - 14  | Project Report & Defense Slides|
+---------+----------------------------------------------+---------------+--------------------------------+
```

---

## 3. Detailed Phase Breakdown

### Phase 0: Audit Sign-Off & Environment Freezing
* **Objective:** Freeze existing directories (`experiments/`, `results/`, `data/`).
* **Actions:**
  1. Confirm that `.codex-deps` has `torch`, `scikit-learn`, `joblib`, `numpy`, and `scipy`.
  2. Verify that Python 3.12 executes smoothly with `sys.path.insert(0, str(ROOT / ".codex-deps"))`.
  3. Ensure no existing script is edited or executed with write flags.

---

### Phase 1: Data Pipeline & Zero-Leakage Split Lock
* **Objective:** Construct a leak-proof PyTorch dataset pipeline.
* **Actions:**
  1. Implement `novel_contribution/data_processing/dataset_loader.py` loading `mmasd_openpose_64frames.npz`.
  2. Implement `SubjectGroupedKFold`:
     * Partition the 32 subject IDs into 5 distinct outer folds.
     * In each outer training fold, partition subjects into 82% fit subjects and 18% validation subjects.
     * **Hard Assertions:** Add programmatic assertions verifying zero subject overlap between fit, validation, and test sets:
       ```python
       assert len(set(train_subjects).intersection(val_subjects)) == 0
       assert len(set(train_subjects).intersection(test_subjects)) == 0
       assert len(set(val_subjects).intersection(test_subjects)) == 0
       ```
  3. Fit coordinate normalization statistics ($\mu, \sigma$) strictly on the fit partition and apply to val/test without leakage.

---

### Phase 2: Reliability Extraction & Kinematic Prior Modeling
* **Objective:** Formalize joint- and frame-level reliability indicators.
* **Actions:**
  1. Extract PAF OpenPose detector confidence $c_{v,t} \in [0, 1]$ directly from array $Q$.
  2. Implement kinematic velocity prior $v_{v,t} = \exp(-\|\Delta p\|^2 / (2 \sigma_v^2))$ based on inter-frame coordinate displacement.
  3. Form combined joint reliability $r_{v,t} = c_{v,t} \cdot v_{v,t}$ and frame reliability $R_t = \frac{1}{V} \sum_v r_{v,t}$.
  4. Unit test: Assert that $0 \le r_{v,t} \le 1$ and $0 \le R_t \le 1$ for all samples.

---

### Phase 3: RA-STGT Core Architecture Implementation
* **Objective:** Implement the dual-stage reliability-gated network.
* **Actions:**
  1. **Spatial Module (`spatial_gat.py`):**
     * Input projection: Linear layer $(4 \to 64)$.
     * Physical bone adjacency matrix over 25 joints + adaptive spatial attention.
     * Reliability-weighted message attenuation: $\tilde{\alpha}_{ij,t} \propto \alpha_{ij,t} \cdot r_{j,t}$.
     * Self-loop identity retention to prevent node representation collapse.
  2. **Temporal Module (`temporal_attention.py`):**
     * Multi-head temporal self-attention over 64 time steps.
     * Frame-level reliability modulation: $\tilde{A}_{t,\tau}^{temp} = \text{softmax}( (q_t k_\tau^\top / \sqrt{d}) + \log(R_\tau + \epsilon) )$.
  3. **Classification Head:**
     * Temporal average pooling over valid frames $\to$ LayerNorm $\to$ Dropout(0.3) $\to$ Linear(64, 10).
  4. Ensure overall model is lightweight (< 1.5M parameters) to prevent overfitting on 1,374 samples.

---

### Phase 4: Clean Benchmarking & Single-Factor Ablations
* **Objective:** Run controlled 5-fold cross-validation across 3 seeds.
* **Comparison Setup:**
  * B2: Nested RBF Kernel Ridge (historical baseline: 71.69% mean acc / 0.6920 OOF F1).
  * B3: Standard ST-GCN (confidence-blind).
  * B4: Standard STGT (confidence-blind).
  * M1: STGT + Naive Confidence Concatenation.
  * M2: STGT + Spatial Reliability Gating (SR-GAT only).
  * M3: STGT + Reliability-Aware Temporal Attention (RATA only).
  * M4: Full RA-STGT (SR-GAT + RATA).
* **Logging:** Save per-fold accuracy, macro-F1, per-class F1, confusion matrices, and training curves into `novel_contribution/results/metrics/`.

---

### Phase 5: Predictive Uncertainty, Calibration (ECE) & Selective Prediction
* **Objective:** Calibrate probabilities and evaluate clinical selective prediction.
* **Actions:**
  1. Fit temperature parameter $T^*$ on inner validation folds using `L-BFGS-B` to minimize NLL.
  2. Measure calibration error:
     * Expected Calibration Error (ECE) with 10 bins.
     * Maximum Calibration Error (MCE) and Brier Score before and after temperature scaling.
  3. Implement Selective Classification:
     * Evaluate coverage $C \in [0.5, 1.0]$ by sweeping confidence threshold $\tau$.
     * Plot Risk-Coverage curve: demonstrate that abstaining on the 15–20% most uncertain/occluded clips boosts accuracy on the retained subset.

---

### Phase 6: Occlusion Robustness Stress Testing (0% – 40%)
* **Objective:** Empirically validate the hypothesis that reliability-aware models resist sensor failure.
* **Actions:**
  1. Evaluate trained models under controlled joint dropouts: 0%, 10%, 20%, 30%, and 40% random joint masking.
  2. Evaluate under limb-block occlusion (dropping entire left arm or right arm).
  3. Compare performance degradation slope:
     $$\Delta \text{Macro-F1} = \text{F1}_{\text{clean}} - \text{F1}_{40\%}$$
     Prove that RA-STGT maintains higher macro-F1 than standard ST-GCN and RBF baselines when limbs are missing.

---

### Phase 7: AV-ASD Forensic Audit & Case Study Summary
* **Objective:** Document the multimodal and dataset boundaries of AV-ASD.
* **Actions:**
  1. Produce a clean standalone report in `novel_contribution/results/avasd_case_study.md`.
  2. Document the 262 missing video clips, the 227 usable audio clips, and the source-video grouping constraints.
  3. Re-affirm why AV-ASD is kept as an independent case study rather than forced into a merged model.

---

### Phase 8: Paper Figures, Viva Deck & Final Artifacts
* **Objective:** Generate publication-ready visualizations and presentation assets.
* **Actions:**
  1. Generate SVG and high-resolution PNG figures:
     * Fig 1: RA-STGT Architecture Diagram with dual spatial-temporal reliability gates.
     * Fig 2: Spatial attention attenuation maps showing suppressed occluded limbs.
     * Fig 3: Robustness curves (Macro-F1 vs missingness percentage).
     * Fig 4: Reliability diagram (ECE) and Risk-Coverage curve.
  2. Prepare Project Viva Q&A Guide addressing all expected examiner questions.

---

## 4. Team Task Allocation (3-Member BTech Team)

| Team Member | Assigned Modules & Responsibilities | Core Viva Defense Area |
|---|---|---|
| **Member 1** | **Data Pipeline & Spatial Reliability (SR-GAT):**<br>- Subject-grouped data loader and leakage checks.<br>- OpenPose confidence extraction and velocity prior.<br>- Spatial attention attenuation implementation. | Why fixed physical adjacency fails and how spatial reliability gating prevents feature corruption. |
| **Member 2** | **Temporal Modeling & Ablations (RATA):**<br>- Reliability-Aware Temporal Attention module.<br>- Baseline ST-GCN and STGT training.<br>- Single-factor ablation experiments (M1, M2, M3, M4). | Why naive confidence concatenation fails and how temporal reliability gating preserves valid motion dynamics. |
| **Member 3** | **Uncertainty, Calibration & Robustness:**<br>- Temperature scaling, ECE, and Brier score evaluation.<br>- Risk-Coverage curves and selective prediction logic.<br>- Synthetic occlusion stress tests (0%–40% missingness). | How selective prediction ensures clinical safety and why calibration is as important as raw accuracy. |
