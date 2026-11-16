# RCDB-STGT IMPLEMENTATION: SOURCE CODE STRUCTURE & MODULAR INTERFACES

**Document Status:** Complete & Verified  
**Project:** Autism Therapy Physical Activity Recognition Using 2D Skeleton Dynamics  
**Source Location:** `research_stage4/models/rcdb_stgt.py`  
**Date:** November 2026  

---

## 1. Module Overview & Dependency Graph

The Stage 4 implementation is contained in `research_stage4/models/rcdb_stgt.py`. It is designed as an independent, modular architecture that imports shared utilities from `novel_contribution/` without modifying any historical Stage 1–3 code.

### Dependency Graph:
```
research_stage4/models/rcdb_stgt.py
    |-- novel_contribution.reliability_estimation.velocity_prior.ReliabilityEstimator
    |-- novel_contribution.data_processing.dataset_loader.BODY_25_EDGES
    |-- research_stage2.models.phase_rata.compute_harmonic_phase_difference
    |-- torch.nn
    |-- torch.nn.functional
```

---

## 2. Component Class Specifications

### 2.1 `DynamicBilateralEdgeLearner`
- **Purpose:** Estimates instantaneous and temporally smoothed dynamic connection weights between homologous contralateral joint pairs $(u, v)$.
- **Inputs:**
  - `h_u`: Embedding of joint $u$, shape `(B, T, d_model)`
  - `h_v`: Embedding of joint $v$, shape `(B, T, d_model)`
  - `r_u`: Calibrated reliability of joint $u$, shape `(B, T, 1)`
  - `r_v`: Calibrated reliability of joint $v$, shape `(B, T, 1)`
- **Symmetric Invariance Formulation:**
  ```python
  diff = torch.abs(h_u - h_v)
  prod = h_u * h_v
  feat_uv = torch.cat([h_u, h_v, diff, prod, r_u, r_v], dim=-1)
  feat_vu = torch.cat([h_v, h_u, diff, prod, r_v, r_u], dim=-1)
  feat_sym = 0.5 * (feat_uv + feat_vu)
  ```
- **Temporal Edge Convolution:** A 1D depthwise convolution (`kernel_size=5`, `padding=2`) filters the scalar edge logits across sequence length $T$ before the final sigmoid activation, suppressing erratic single-frame jumps while preserving rhythmic oscillations.

---

### 2.2 `RCDB_SpatialAttention`
- **Purpose:** Spatial graph convolution layer executing parallel message passing across anatomical physical bone edges and learned dynamic bilateral edges.
- **Anatomical Branch:**
  - Uses fixed OpenPose `BODY_25` bone adjacency (24 connections).
  - Attention logits are masked with $-10^9$ for non-adjacent joints.
  - Modulated by edge reliability $\tilde{r}_{ij} = \sqrt{r_i \cdot r_j}$.
- **Dynamic Bilateral Branch:**
  - Operates over candidate homologous limb pairs $\mathcal{P}_{\text{bi}}$.
  - Injects dynamic edge weights $g_{uv}(t) = q_{uv}(t) \cdot a_{uv}(t)$.
  - Projects messages via $\mathbf{W}_{\text{bi}}$ and aggregates symmetrically into joint representations.
- **Edge Weight Cache:** Exposes `self.last_edge_weights` dictionary mapping pair names (e.g., `"4_7"`, `"11_14"`) to their dynamic transmission coefficients `(B, T, 1)` for visualization and interpretability.

---

### 2.3 `RCDB_TemporalAttention`
- **Purpose:** Multi-head temporal self-attention across the 64 time frames.
- **Components:**
  - `MultiheadAttention(embed_dim=64, num_heads=2, dropout=0.25)`
  - Residual connection with LayerNorm
  - Position-wise Feedforward Network (Linear $64 \to 128 \to 64$, ReLU, Dropout)
  - Secondary LayerNorm

---

### 2.4 `RCDB_STGT` Master Model
- **Constructor Parameters:**
  - `in_channels` (int, default=4): Coordinates and velocities $(x, y, v_x, v_y)$.
  - `num_classes` (int, default=10): MMASD activity classes.
  - `d_model` (int, default=64): Hidden channel dimensionality.
  - `num_spatial_heads` (int, default=2), `num_temporal_heads` (int, default=2).
  - `num_spatial_layers` (int, default=2), `num_temporal_layers` (int, default=2).
  - `dropout` (float, default=0.25).
  - `bilateral_pairs` (list of tuples, default=[(4, 7), (11, 14)]).
  - `use_dynamic_edges` (bool, default=True): If False, bilateral branch is bypassed (enabling A0/E1/E2 comparisons).
  - `use_reliability_gate` (bool, default=True): If False, $q_{uv}=1$ (enabling A1/A8 ablation).
  - `use_temporal_smoothing` (bool, default=True): If False, temporal 1D convolution is disabled (enabling A4 ablation).
  - `use_phase_stream` (bool, default=False): If True, computes and projects harmonic phase difference $\Delta \phi_t$ (enabling A5 ablation).

---

## 3. Configuration Mapping for Stage 4 Ablation Suite (A0 to A9)

| Ablation ID | Model Name | `use_dynamic_edges` | `use_reliability_gate` | `use_temporal_smoothing` | `bilateral_pairs` | `use_phase_stream` | Loss Objective |
|:---:|---|:---:|:---:|:---:|:---:|:---:|---|
| **A0** | Fixed Topology E2 (Repro) | False (Fixed mask) | True (Static E2) | False | Default (4,7), (11,14) | False | Focal Loss |
| **A1** | Dynamic without Reliability | True | False ($q_{uv}=1$) | True | Default (4,7), (11,14) | False | Focal Loss |
| **A2** | Reliability without Dynamic | False (Fixed mask) | True ($a_{uv}=1$) | False | Default (4,7), (11,14) | False | Focal Loss |
| **A3** | **Full RCDB-STGT (Proposed)**| **True** | **True** | **True** | **Default (4,7), (11,14)**| **False** | **Focal Loss** |
| **A4** | A3 without Temporal Smoothing | True | True | False | Default (4,7), (11,14) | False | Focal Loss |
| **A5** | A3 with Harmonic Phase Stream | True | True | True | Default (4,7), (11,14) | True | Focal Loss |
| **A6** | A3 with Wrist-Only Edges | True | True | True | Wrists only [(4, 7)] | False | Focal Loss |
| **A7** | A3 with Broad Bilateral Topology | True | True | True | Wrists, Ankles, Elbows, Knees, Hips | False | Focal Loss |
| **A8** | A3 without Multiplicative Gate | True | False (Input only) | True | Default (4,7), (11,14) | False | Focal Loss |
| **A9** | A3 with Confusable Margin Loss | True | True | True | Default (4,7), (11,14) | False | Focal + Margin ($\mathcal{L}_{\text{CPM}}$) |

---

## 4. Verification & Unit Testing

The model implementation was validated through automated forward-pass unit tests:
```bash
python research_stage4/models/rcdb_stgt.py
```
**Output Trace:**
```text
Testing RCDB_STGT model initialization...
RCDB_STGT Parameter Count: 143,336
Forward pass output shape: torch.Size([2, 10])
Captured Bilateral Edges: ['4_7', '11_14']
  Edge 4_7 shape: torch.Size([2, 64, 1]), Mean Strength: 0.0344
  Edge 11_14 shape: torch.Size([2, 64, 1]), Mean Strength: 0.0254
RCDB_STGT test passed successfully!
```
The architecture executes with zero dimension errors, verifies symmetry, and correctly tracks dynamic edge strengths across all 64 frames.
