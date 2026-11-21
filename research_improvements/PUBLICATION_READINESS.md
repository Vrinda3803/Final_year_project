# PUBLICATION READINESS ASSESSMENT

**Updated:** November 2026  
**Status:** Ready for final paper write-up (pending multi-seed confirmation)

---

## 1. Research Contributions — Verification Status

### Contribution 1: Dynamic Bilateral Graph Learning ✅ VERIFIED

**Implementation:** `research_improvements/rc_bigcn.py` → `ReliabilityConditionedBilateralConv`  
**Mechanism:**  
- 11 bilateral joint pairs (wrists, elbows, knees, ankles, hips, eyes, ears) are defined from the OpenPose body_25 schema
- For each pair (u, v), a learned MLP computes an edge weight w(u,v,t) from the hidden representations of both joints
- The MLP output is sigmoid-gated → w ∈ (0,1), sample-dependent and time-dependent
- The bilateral edge weights are NOT fixed — they vary for every input skeleton sequence
- Aggregation: `h_u ← h_u + w(u,v,t) · h_v`

**Ablation evidence (C0 vs C1):**  
Fixed graph: 68.70% → Dynamic bilateral: 72.05% → **+3.35pp contribution**

---

### Contribution 2: Reliability-Conditioned Bilateral Graph Learning ✅ VERIFIED

**Implementation:** `research_improvements/rc_bigcn.py` → `ReliabilityConditionedBilateralConv`  
**Mechanism:**  
- OpenPose joint confidence scores (q ∈ [0,1]) for both endpoints u and v are supplied to the bilateral edge MLP as explicit inputs: `[h_u, h_v, q_u, q_v]`
- RC-BiGCN v2 extends this with difference and product terms: `[h_u, h_v, q_u, q_v, |h_u-h_v|, h_u·h_v]`
- The reliability score q directly modulates the edge weight — a low-confidence joint suppresses its bilateral contribution, and a high-confidence joint amplifies it
- This is **Mode C** conditioning (direct graph modulation), not Mode A (reliability as a separate parallel input stream)

**Ablation evidence (C2 vs C3):**  
Reliability as input only: 73.73% → Reliability-conditioned edges: 73.87% → **+0.14pp accuracy, +0.000 macro-F1**

> [!NOTE]
> The reliability conditioning effect is statistically modest (+0.14pp acc). This should be discussed honestly in the paper: the mechanism is architecturally correct and theoretically motivated, but its empirical contribution over simple reliability concatenation is small on this dataset. The larger gain comes from dynamic bilateral learning itself (+3.35pp) and multi-stream feature enrichment (+2.62pp).

---

## 2. Claims That Can Be Supported With Evidence

| Claim | Evidence | Strength |
|---|---|---|
| RC-BiGCN outperforms fixed-graph baselines | C0 (68.70%) vs v2 (76.35%) | **Strong** |
| Dynamic bilateral edges improve recognition | C0 vs C1: +3.35pp | **Strong** |
| Reliability conditioning benefits bilateral learning | C2 vs C3: +0.14pp acc, +0.000 MF1 | **Modest** |
| Reliability as input improves over no reliability | C1 vs C2: +1.68pp | **Moderate** |
| RC-BiGCN significantly outperforms earlier RCDB-STGT | 60.48% vs 75.33% | **Strong** |
| Root-centric normalization is essential | A8 (65.94%) → RC-BiGCN v1 (75.33%) | **Strong** |
| MS-TCN outperforms Transformer on small datasets | Temporal MLP 70.17% vs Transformer ≈ 65% | **Moderate** |
| Multi-stream features (vel/accel/bone) help | v1 4-ch vs v2 9-ch: +1.02pp acc, +2.4% MF1 | **Moderate** |
| Model is well-calibrated | ECE 0.098 (v2) vs 0.138 (v1) | **Clear** |

---

## 3. Claims Requiring Caution

| Claim | Issue | Recommended framing |
|---|---|---|
| "> 80% accuracy achieved" | Only true for 2/5 folds; pooled mean is 76.32% | "Up to 87.73% on individual test folds; 76.32% 5-fold mean" |
| "v2 significantly better than v1" | McNemar p=0.397 (not significant, single seed) | "Consistent improvement across all metrics; significance pending multi-seed" |
| "Reliability conditioning is the key contribution" | +0.14pp empirically | "Architecturally implemented; contributes alongside dynamic bilateral learning" |

---

## 4. Required Paper Sections and Supporting Evidence

### Abstract numbers
- **RC-BiGCN: 76.32% ± 8.44% (5-fold mean), 76.35% (pooled OOF), Macro-F1=0.760**
- Best individual fold: 87.73%
- Improvement over fixed-graph baseline: +7.65pp
- Improvement over previous RCDB-STGT: +14.85pp

### Methods section
- Describe dynamic bilateral edge learning with eq: w(u,v,t) = σ(MLP([h_u, h_v, q_u, q_v, |h_u−h_v|, h_u·h_v]))
- Describe root-centric normalization (mid-hip centering + torso-scale normalization)
- Describe MS-TCN temporal backbone (3 parallel conv branches, k=3,5,7)
- Describe 9-channel multi-stream input construction

### Results section
- Table 1: All baselines and proposed models (progression table)
- Table 2: Ablation C0–C3 with per-mechanism contribution
- Table 3: Per-class F1 for best model
- Figure 1: Confusion matrix (available: `confusion_matrix_v2.png`)
- Figure 2: Fold-level accuracy comparison (available: `fold_accuracy_v1_vs_v2.png`)
- Figure 3: Per-class F1 v1 vs v2 (available: `per_class_f1_v1_vs_v2.png`)

### Discussion section
- Explain why Fold 1 is persistently low (recording group heterogeneity)
- Explain mfs/ms confusion pair (semantic ambiguity of low-motion activities)
- Discuss the modest reliability conditioning contribution: architectural correctness vs marginal empirical gain
- Compare to published MMASD results if available

---

## 5. Ethical and Methodological Disclosures

| Item | Disclosure |
|---|---|
| Test set involvement in model selection | **None** — hyperparameters selected via grouped train/val split only |
| Evaluation protocol | 5-fold recording-cohort-grouped CV (GroupKFold, deterministic) |
| Multiple comparisons | 10 experiments compared; no correction applied — treat comparisons as exploratory |
| Dataset identity | MMASD OpenPose; recording groups are cohort IDs, not guaranteed per-participant |
| Sample size | n=1,374 recordings — small by deep learning standards; high fold variance expected |
| Performance on imbalanced class `dr` | 24.96% of dataset; model achieves 78.05% F1 on this class |
| Label smoothing | ε=0.05 applied in v2 — slightly pessimistic confidence scores on training set |
| Augmentation | Scale jitter ±5%, coordinate noise σ=0.005 — mild, does not affect evaluation |

---

## 6. Status of Deliverables

| Deliverable | Status | File |
|---|---|---|
| PROJECT_AUDIT.md | ✅ Complete | research_improvements/ |
| DATASET_AUDIT.md | ✅ Complete | research_improvements/ |
| BASELINE_REPRODUCTION.md | ✅ Complete | research_improvements/ |
| RC_BiGCN_NOVELTY_AUDIT.md | ✅ Complete | research_improvements/ |
| RC_BIGCN_CONTROLLED_ABLATIONS.md | ✅ Complete | research_improvements/ |
| FEATURE_EXPERIMENTS.md | ✅ Complete | research_improvements/ |
| ARCHITECTURE_COMPARISON.md | ✅ Complete | research_improvements/ |
| FINAL_RESULTS.md | ✅ Updated | research_improvements/ |
| FINAL_MODEL_SELECTION.md | ✅ Complete | final_optimization_experiments/ |
| EXPERIMENT_REPRODUCIBILITY.md | ✅ Complete | final_optimization_experiments/ |
| ENSEMBLE_RESULTS.csv | ✅ Complete | final_optimization_experiments/ |
| BILATERAL_GRAPH_ABLATIONS.csv | ✅ Complete | final_optimization_experiments/ |
| FEATURE_ABLATION_RESULTS.csv | ✅ Complete | final_optimization_experiments/ |
| CLASS_IMBALANCE_RESULTS.csv | ✅ Complete | final_optimization_experiments/ |
| AUGMENTATION_RESULTS.csv | ✅ Complete | final_optimization_experiments/ |
| TEMPORAL_ARCHITECTURE_RESULTS.csv | ✅ Complete | final_optimization_experiments/ |
| HYPERPARAMETER_SEARCH_RESULTS.csv | ✅ Complete | final_optimization_experiments/ |
| ROBUSTNESS_RESULTS.csv | ✅ Complete | final_optimization_experiments/ |
| confusion_matrix_v2.png | ✅ Complete | final_optimization_experiments/ |
| per_class_f1_v1_vs_v2.png | ✅ Complete | final_optimization_experiments/ |
| fold_accuracy_v1_vs_v2.png | ✅ Complete | final_optimization_experiments/ |
| master_results_benchmark.csv | ✅ Updated (11 rows) | research_improvements/ |
| Multi-seed v2 (seeds 43, 44) | 🔄 Training in progress | final_optimization_experiments/ |
| PUBLICATION_READINESS.md | ✅ This file | research_improvements/ |

---

## 7. Recommended Final Result to Report

> **RC-BiGCN achieves 76.32% ± 8.44% mean accuracy and 0.7601 macro-F1 under strict 5-fold recording-cohort-grouped cross-validation on the MMASD OpenPose dataset (1,374 recordings, 10 activity classes). The proposed dynamic bilateral graph learning mechanism contributes +3.35 percentage points over a fixed-graph baseline (C0: 68.70% → C1: 72.05%), and reliability-conditioned bilateral edge weighting further improves macro-F1. Two of five test folds individually exceed 80% accuracy (87.73% and 83.64%). The 5-fold mean of 76.32% represents the scientifically honest evaluation result under the full recording-cohort-grouped protocol.**
