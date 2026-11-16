# REPRODUCIBILITY & ARTIFACT REPLICATION GUIDE: STAGE 4

**Project:** Autism Therapy Physical Activity Recognition Using 2D Skeleton Dynamics  
**Framework:** Reliability-Conditioned Dynamic Bilateral Graph Learning (RCDB-STGT)  
**Date:** November 2026  
**Status:** Completed & Validated  

---

## 1. Overview & Verification Philosophy

This research project adheres to strict open-science reproducibility standards. Every numerical metric, statistical test, figure, and model checkpoint reported in Stage 4 can be reproduced deterministically from raw data files.

All new code, configurations, checkpoints, predictions, metrics, and reports are isolated exclusively inside:
```
research_stage4/
├── audit/                  # Audit scripts and data verification logs
├── checkpoints/            # PyTorch model weights (.pt) per ablation and fold
├── configs/                # Model hyperparameter JSON configurations
├── experiments/            # Ablation runner scripts (A0 through A9)
├── literature/             # Novelty matrices and related work taxonomies
├── metrics/                # Out-of-fold summary JSON files
├── models/                 # PyTorch module implementations (rcdb_stgt.py)
├── predictions/            # Out-of-fold prediction CSVs (1,374 clips)
├── reports/                # Final research reports and executive briefings
├── robustness/             # Sensor degradation benchmarking suite
├── statistics/             # Formal hypothesis testing & bootstrap analysis
└── visualizations/         # 300 DPI high-resolution figures (.png and .svg)
```

---

## 2. Computational Environment Setup

### 2.1 Hardware & Operating System Specifications
- **Operating System:** Windows 11 Enterprise (64-bit)
- **Processor:** Intel Core i7 / 64-bit x64 Family 6 Model 186
- **System Memory:** 16+ GB RAM
- **Compute Device:** CPU Execution (native multithreaded PyTorch)

### 2.2 Software & Isolated Python Environment
- **Python Version:** 3.12.3
- **Deep Learning Framework:** PyTorch 2.5.1+cpu
- **Isolated Vendor Dependencies:** All external packages are vendored in `d:\Documents\Autism_Project\.codex-deps`. No global pip installs or administrative privileges are required.

To activate the isolated environment in any script:
```python
import sys
from pathlib import Path
ROOT = Path("d:/Documents/Autism_Project")
sys.path.insert(0, str(ROOT / ".codex-deps"))
sys.path.insert(0, str(ROOT))
```

---

## 3. Dataset Integrity & Checksum Verification

The MMASD 2D OpenPose skeleton dataset must contain exactly:
- **Total Clips:** 1,374 recordings
- **Physical Activity Classes:** 10 classes (`arm_spring`, `ball_slap`, `clapping`, `drumming`, `jump_mode`, `marching_forward_and_side`, `marching_in_place`, `stand_and_clap`, `squats`, `tree_pose`)
- **Recording Cohorts:** 32 unique directory prefixes
- **Tensor Input Dimensions:** $(B, 64, 25, 4)$ where channels denote $(x, y, v_x, v_y)$ and $Q \in \mathbb{R}^{B \times 64 \times 25}$ denotes OpenPose detection confidence.

Run the unit test verification suite to verify dataset integrity:
```powershell
python test_pipeline.py
```
*Expected Output:* `5/5 unit tests passed successfully.`

---

## 4. Reproducing the Stage 4 Ablation Matrix (A0 to A9)

### 4.1 Running Individual Ablations
To execute any specific ablation experiment across all 5 cross-validation folds:
```powershell
# A0: Fixed Bilateral Topology (E2 Reproduction)
python research_stage4/experiments/run_stage4_ablations.py --exp a0 --seed 42

# A1: Dynamic Bilateral without Reliability Gate
python research_stage4/experiments/run_stage4_ablations.py --exp a1 --seed 42

# A2: Reliability Gated without Dynamic Edge Learner
python research_stage4/experiments/run_stage4_ablations.py --exp a2 --seed 42

# A3: Full RCDB-STGT (Main Proposed Framework)
python research_stage4/experiments/run_stage4_ablations.py --exp a3 --seed 42

# A4: RCDB-STGT without 1D Temporal Edge Smoothing
python research_stage4/experiments/run_stage4_ablations.py --exp a4 --seed 42

# A5: RCDB-STGT with Harmonic Phase Velocity Stream
python research_stage4/experiments/run_stage4_ablations.py --exp a5 --seed 42

# A6: RCDB-STGT with Wrists Only [(4, 7)]
python research_stage4/experiments/run_stage4_ablations.py --exp a6 --seed 42

# A7: RCDB-STGT with Multi-Pair Bilateral Limbs
python research_stage4/experiments/run_stage4_ablations.py --exp a7 --seed 42

# A8: RCDB-STGT with Reliability as MLP Input Only
python research_stage4/experiments/run_stage4_ablations.py --exp a8 --seed 42

# A9: RCDB-STGT with Confusable-Pair Margin Loss
python research_stage4/experiments/run_stage4_ablations.py --exp a9 --seed 42
```

### 4.2 Running the Full Suite
To run all 10 ablations sequentially:
```powershell
python research_stage4/experiments/run_stage4_ablations.py --exp all --seed 42
```
*Outputs Generated:*
- Checkpoints: `research_stage4/checkpoints/<exp_id>/fold_0.pt` through `fold_4.pt`
- Out-of-fold CSV: `research_stage4/predictions/<exp_id>_oof.csv`
- Fold summary JSON: `research_stage4/metrics/<exp_id>_summary.json`
- Master summary JSON: `research_stage4/metrics/stage4_ablation_master_summary.json`

---

## 5. Reproducing Robustness Stress Benchmarking

To evaluate models under sensor degradation regimes (Clean, 20% Missingness, 40% Missingness, Wrist Occlusion, Ankle Occlusion, 15-Frame Temporal Drop):
```powershell
python research_stage4/robustness/benchmark_robustness_stage4.py
```
*Outputs Generated:*
- Metrics JSON: `research_stage4/robustness/stage4_robustness_benchmark_results.json`

---

## 6. Reproducing Statistical Significance Analysis

To execute paired $t$-tests, Wilcoxon signed-rank tests, Cohen's $d$, McNemar tests, and 10,000-iteration bootstrap confidence intervals:
```powershell
python research_stage4/statistics/statistical_testing_stage4.py
```
*Outputs Generated:*
- Statistical Results JSON: `research_stage4/statistics/stage4_statistical_significance_results.json`

---

## 7. Generating Publication Figures (300 DPI)

To render all 300 DPI figures and vector SVGs:
```powershell
python research_stage4/visualizations/generate_stage4_visualizations.py
```
*Outputs Generated:*
- `research_stage4/visualizations/fig1_ablation_matrix_benchmark.png` (.svg)
- `research_stage4/visualizations/fig2_robustness_stress_comparison.png` (.svg)
- `research_stage4/visualizations/fig3_dynamic_bilateral_heatmap_drumming.png` (.svg)
- `research_stage4/visualizations/fig4_confusion_matrices_comparison.png` (.svg)

---

## 8. Clinical CLI Inference & Model Deployment

To run single-sample inference with temperature-calibrated confidence scores and automated wrist occlusion warning alerts:
```powershell
python inference_pipeline.py --clip sample_001.json --model a3
```
*Sample JSON Response:*
```json
{
  "clip": "sample_001.json",
  "predicted_class": "drumming",
  "confidence": 0.8421,
  "calibrated": true,
  "wrist_occlusion_detected": false,
  "inference_latency_ms": 3.82,
  "class_probabilities": {
    "drumming": 0.8421,
    "stand_and_clap": 0.0812,
    "arm_spring": 0.0315,
    "marching_in_place": 0.0142,
    ...
  }
}
```
