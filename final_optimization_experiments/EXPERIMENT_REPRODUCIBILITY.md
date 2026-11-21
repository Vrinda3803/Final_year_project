# EXPERIMENT REPRODUCIBILITY LOG

**Project:** Autism Therapy Physical Activity Recognition Using Skeleton Dynamics  
**Researcher:** Final-year project  
**Last updated:** November 2026

---

## Environment

| Component | Version |
|---|---|
| Python | 3.12 (C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe) |
| PyTorch | 2.15.0+cpu |
| scikit-learn | 1.6.1 |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| Dependencies path | D:\Documents\Autism_Project\.codex-deps |

---

## Dataset

| Property | Value |
|---|---|
| File | data/processed/mmasd_openpose/mmasd_openpose_64frames.npz |
| Total recordings | 1,374 |
| Classes | 10 (as, bs, ce, dr, mfs, ms, sac, sq, tr, tw) |
| Frames per recording | 64 |
| Joints | 25 (OpenPose body_25) |
| Input shape | X: (1374, 64, 25, 4) — x, y, z, q per joint |
| Reliability shape | Q: (1374, 64, 25) |
| Groups | 32 unique recording cohort groups |

---

## Cross-Validation Protocol

```python
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(X, y, groups))
# Splits are deterministic given (X, y, groups) — no random seed needed
```

- **GroupKFold** ensures no recording group appears in both train and test within any fold.
- Splits are **deterministic** (GroupKFold does not use random state).
- The 5 folds are identical across all experiments (v1, v2, C0–C3, Stage 4).

---

## RC-BiGCN v1 (Original Proposed Architecture)

### Key Hyperparameters
| Param | Value |
|---|---|
| Random seed | 42 |
| base_channels | 32 |
| n_gcn_layers | 2 |
| dropout | 0.3 |
| learning_rate | 3e-4 |
| weight_decay | 1e-3 |
| batch_size | 32 |
| max_epochs | 80 |
| early_stop_patience | 20 |
| optimizer | AdamW |
| scheduler | CosineAnnealingLR (T_max=80, eta_min=1e-5) |
| loss | CrossEntropyLoss + inverse-frequency weights |
| augmentation | None |
| input_channels | 4 |

### Reproduce command
```powershell
cd D:\Documents\Autism_Project
C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe `
  research_improvements\run_rc_bigcn.py
```

### Results (seed 42)
- Mean fold accuracy: 75.32% ± 6.38%
- Pooled OOF accuracy: 75.33%
- Pooled OOF Macro-F1: 0.7361
- Results file: `research_improvements/rc_bigcn_results.json`
- OOF file: `research_improvements/rc_bigcn_oof.npz`

---

## RC-BiGCN v2 (Optimized Architecture)

### Key Hyperparameters
| Param | Value |
|---|---|
| Random seed | 42 (primary), 43, 44 (multi-seed verification) |
| base_channels | 36 |
| n_gcn_layers | 2 |
| dropout | 0.3 |
| learning_rate | 3e-4 |
| weight_decay | 1e-3 |
| batch_size | 32 |
| max_epochs | 120 |
| early_stop_patience | 20 |
| optimizer | AdamW |
| scheduler | CosineAnnealingLR (T_max=80, eta_min=1e-5) |
| loss | CrossEntropyLoss + label_smoothing(ε=0.05) + eff_number_weights(β=0.99) |
| edge_temporal_reg | λ=0.05 |
| augmentation | scale_jitter(±5%) + coord_noise(σ=0.005) |
| input_channels | 9 (xy + velocity + acceleration + bone + reliability) |

### Architecture innovations over v1
1. **Multi-stream input:** coords, velocity (Δxy), acceleration (Δ²xy), bone vectors (joint-root), reliability (q) → 9 channels
2. **Enhanced bilateral edge MLP:** `[hu, hv, qu, qv, |hu-hv|, hu*hv]` — includes difference and product interaction terms that encode directional asymmetry and magnitude correlation
3. **LayerNorm + SiLU** replace BatchNorm + ReLU (better gradient flow for small batch sizes)
4. **Effective number class weighting** (β=0.99) instead of simple inverse frequency

### Reproduce command (seed 42)
```powershell
cd D:\Documents\Autism_Project
C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe `
  final_optimization_experiments\run_optimized_suite.py
```

### Reproduce multi-seed (43, 44)
```powershell
cd D:\Documents\Autism_Project
C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe `
  final_optimization_experiments\run_multiseed_v2.py
```

### Results (seed 42)
- Mean fold accuracy: 76.32% ± 8.44%
- Pooled OOF accuracy: 76.35%
- Pooled OOF Macro-F1: 0.7601
- Pooled OOF ECE: 0.0980
- Results file: `final_optimization_experiments/rc_bigcn_v2_seed42_summary.json`
- OOF file: `final_optimization_experiments/rc_bigcn_v2_seed42_oof.npz`

---

## Controlled Ablation Study (C0–C3)

Ablations share the RC-BiGCN v2 preprocessing and training configuration.

### Reproduce
```powershell
cd D:\Documents\Autism_Project
C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe `
  research_improvements\run_controlled_ablations.py
```

### Results
| Exp | Config | Pooled OOF Acc | Macro-F1 | ECE |
|---|---|---|---|---|
| C0 | Fixed graph + MS-TCN | 68.70% | 0.6711 | 0.1724 |
| C1 | Dynamic bilateral (no reliability) | 72.05% | 0.6923 | 0.1537 |
| C2 | Dynamic bilateral + reliability input | 73.73% | 0.7174 | 0.1269 |
| C3 | Full RC-BiGCN (reliability-conditioned) | 73.87% | 0.7178 | 0.1378 |

Results file: `research_improvements/rc_bigcn_controlled_ablations_summary.json`

---

## Stage 4 Historical Baselines (reference only, do not reproduce)

| Model | Pooled OOF Acc | Script |
|---|---|---|
| A0: Fixed Topo STGT | 57.13% | research_stage4/run_ablation.py |
| A3: RCDB-STGT | 60.48% | research_stage4/run_ablation.py |
| A8: Dynamic Soft Rel | 65.94% | research_stage4/run_ablation.py |
| Temporal MLP | 70.17% | research_stage4/... |
| Nested RBF Kernel | 71.69% | research_stage4/... |
| HistGradientBoosting | 72.85% | research_stage4/... |

---

## Random Seeds & Reproducibility Notes

1. **GroupKFold** does not use a random seed — fold splits are deterministic.
2. **torch.manual_seed(seed)** and **random.seed(seed)** are set at the top of each fold.
3. **np.random.RandomState(seed)** is used for train/validation split permutation.
4. PyTorch CPU operations are deterministic for the same seed on the same hardware.
5. Training-validation split within each fold uses the last 15% of a permuted index.
6. Results may differ slightly across different hardware or OS due to floating-point non-determinism in CPU math libraries.

---

## File Integrity

All result files are write-once. No existing result file was overwritten during this study.  
To verify: check file modification timestamps against experiment run dates.

| Critical file | Expected not to be modified after | Description |
|---|---|---|
| research_improvements/rc_bigcn_results.json | First run (seed 42) | v1 official results |
| research_improvements/rc_bigcn_oof.npz | First run (seed 42) | v1 OOF predictions |
| research_improvements/run_controlled_ablations.py | Ablation design lock | Ablation runner |
| research_improvements/rc_bigcn_controlled_ablations_summary.json | Ablation run | C0–C3 results |
| final_optimization_experiments/rc_bigcn_v2_seed42_summary.json | v2 first run | v2 results |
| final_optimization_experiments/rc_bigcn_v2_seed42_oof.npz | v2 first run | v2 OOF predictions |
