# RC-BiGCN Architecture Description
## For Research Paper Methods Section

---

## 1. Overview

RC-BiGCN (Reliability-Conditioned Bilateral Graph Convolutional Network) is a skeleton-based action recognition architecture designed for autism therapy activity classification from OpenPose 2D skeleton sequences.

The architecture has three main components:
1. **Multi-stream feature extraction** — encodes spatial, temporal, and structural properties of the skeleton
2. **Reliability-conditioned bilateral graph convolution** — applies graph convolution over an anatomical joint graph augmented with dynamically-weighted bilateral edges conditioned on joint reliability
3. **Multi-scale temporal convolution (MS-TCN)** — models activity dynamics across multiple temporal scales

---

## 2. Input Representation

Given a skeleton sequence $X \in \mathbb{R}^{T \times V \times 2}$ of $T=64$ frames and $V=25$ joints, with OpenPose confidence scores $Q \in \mathbb{R}^{T \times V}$, we construct a 9-channel feature tensor:

$$
\mathbf{f}_{t,v} = \left[ x_{t,v},\; y_{t,v},\; \dot{x}_{t,v},\; \dot{y}_{t,v},\; \ddot{x}_{t,v},\; \ddot{y}_{t,v},\; \tilde{x}_{t,v},\; \tilde{y}_{t,v},\; q_{t,v} \right]
$$

where:
- $(x_{t,v}, y_{t,v})$ — root-centric normalized coordinates (mid-hip centering, torso-scale normalization)
- $(\dot{x}, \dot{y})$ — first-order velocity: $\dot{x}_{t,v} = x_{t,v} - x_{t-1,v}$
- $(\ddot{x}, \ddot{y})$ — second-order acceleration: $\ddot{x}_{t,v} = \dot{x}_{t,v} - \dot{x}_{t-1,v}$
- $(\tilde{x}_{t,v}, \tilde{y}_{t,v})$ — bone vector: displacement from root joint $v_0$
- $q_{t,v}$ — OpenPose joint confidence score (reliability)

**Root-centric normalization:** The mid-hip point (average of joints 8 and 11) is subtracted from all coordinates, then divided by the mean torso length (distance from neck to nose, computed on the training set), making the representation translation- and scale-invariant across recording sessions.

---

## 3. Graph Convolution with Bilateral Edges

### 3.1 Anatomical Graph

The fixed anatomical adjacency matrix $A \in \mathbb{R}^{V \times V}$ is constructed from the OpenPose body_25 skeleton topology with 24 edges (limbs and spine connections). It is normalized using symmetric degree normalization:

$$
\hat{A} = D^{-1/2} A D^{-1/2}
$$

### 3.2 Bilateral Joint Pairs

We define 11 bilateral pairs $\mathcal{B} = \{(u_k, v_k)\}_{k=1}^{11}$ connecting anatomically symmetric joint pairs:

| Pair | Left joint | Right joint |
|---|---|---|
| Shoulders | 2 | 5 |
| Elbows | 3 | 6 |
| Wrists | 4 | 7 |
| Hips | 8 | 11 |
| Knees | 9 | 12 |
| Ankles | 10 | 13 |
| Eyes | 14 | 15 |
| Ears | 16 | 17 |
| Small toes | 19 | 22 |
| Big toes | 20 | 23 |
| Heels | 21 | 24 |

### 3.3 Reliability-Conditioned Bilateral Edge Weights

For each bilateral pair $(u, v)$ at time $t$, the edge weight is computed by a small MLP conditioned on both joint hidden representations and their reliability scores:

$$
w_{uv}^{(t)} = \sigma\!\left(\text{MLP}\!\left(\left[\mathbf{h}_u^{(t)},\; \mathbf{h}_v^{(t)},\; q_u^{(t)},\; q_v^{(t)},\; \left|\mathbf{h}_u^{(t)} - \mathbf{h}_v^{(t)}\right|,\; \mathbf{h}_u^{(t)} \odot \mathbf{h}_v^{(t)}\right]\right)\right)
$$

where:
- $\mathbf{h}_u^{(t)} \in \mathbb{R}^{C}$ — hidden representation of joint $u$ at time $t$
- $q_u^{(t)} \in [0,1]$ — OpenPose confidence score for joint $u$
- $|\mathbf{h}_u - \mathbf{h}_v|$ — element-wise absolute difference (asymmetry signal)
- $\mathbf{h}_u \odot \mathbf{h}_v$ — element-wise product (co-activation signal)
- $\sigma(\cdot)$ — sigmoid activation (constrains $w_{uv} \in (0,1)$)

The MLP has architecture: Linear(4C+2) → LayerNorm → SiLU → Dropout(0.1) → Linear(1) → Sigmoid.

**Reliability conditioning mechanism:** Because $q_u^{(t)}$ and $q_v^{(t)}$ are direct inputs to the edge MLP, low-confidence joints (occluded or poorly tracked) produce smaller edge weights, reducing their influence on bilateral message passing. This is a direct conditioning of the graph structure on joint reliability — not merely supplying reliability as a parallel input stream.

### 3.4 Bilateral Aggregation

After computing bilateral edge weights, bilateral message passing updates each joint's representation:

$$
\mathbf{h}_u^{\text{agg}} = \mathbf{h}_u + \sum_{(u,v) \in \mathcal{B}} w_{uv}^{(t)} \cdot \mathbf{h}_v
$$

$$
\mathbf{h}_v^{\text{agg}} = \mathbf{h}_v + \sum_{(u,v) \in \mathcal{B}} w_{uv}^{(t)} \cdot \mathbf{h}_u
$$

### 3.5 Full GCN Block

Each graph convolution block applies:

1. Anatomical spatial conv: $\mathbf{H} = \hat{A} \mathbf{X} W_s$
2. Reliability-conditioned bilateral aggregation: $\mathbf{H}^{\text{bil}}, W_{\text{bil}} = \text{RC-BilConv}(\mathbf{H}, Q)$
3. LayerNorm + SiLU activation
4. Residual connection: $\mathbf{H}^{\text{out}} = \text{SiLU}(\text{LN}(\mathbf{H}^{\text{bil}} + \mathbf{X} W_r))$

---

## 4. Multi-Scale Temporal Convolution (MS-TCN)

After spatial graph convolution, joint features are pooled across the $V$ dimension (mean pooling), yielding a sequence $\mathbf{z} \in \mathbb{R}^{T \times C}$. Two MS-TCN blocks are applied:

Each MS-TCN block uses three parallel dilated 1D convolution branches with kernel sizes $k \in \{3, 5, 7\}$:

$$
\mathbf{z}^{\text{out}} = \text{SiLU}\!\left(\text{Concat}_{k}\left[\text{Conv}_{1D}(k, d_k)\right](\mathbf{z}) + W_p \mathbf{z}\right)
$$

where $d_k = \max(1, k // 2)$ is the dilation factor. The receptive field spans up to 7 frames per branch, covering ~28 frames across two stacked MS-TCN blocks.

---

## 5. Classification

After temporal modeling, global average pooling over $T$ produces a fixed-size representation $\mathbf{g} \in \mathbb{R}^{C}$. A two-layer classifier with LayerNorm and dropout produces the final logits:

$$
\hat{y} = W_2\;\text{SiLU}(\text{LN}(W_1 \mathbf{g}))
$$

---

## 6. Training Details

| Setting | Value |
|---|---|
| Loss | Cross-entropy + label smoothing ($\varepsilon=0.05$) + temporal edge regularizer ($\lambda=0.05$) |
| Class weighting | Effective number weighting ($\beta=0.99$) to handle imbalanced classes |
| Optimizer | AdamW (lr=3×10⁻⁴, weight_decay=10⁻³) |
| Scheduler | Cosine annealing (T_max=80, η_min=10⁻⁵) |
| Batch size | 32 |
| Max epochs | 120 with early stopping (patience=20 on validation loss) |
| Augmentation | Scale jitter (±5%), coordinate noise (σ=0.005), applied to training only |
| Validation split | 15% of training fold (randomly sampled, no group leakage) |
| Parameters | 100,840 |

### Edge Temporal Regularizer

To encourage smooth bilateral edge weights across time and prevent degenerate edge patterns, we add a regularization term:

$$
\mathcal{L}_{\text{reg}} = \lambda \cdot \overline{w}_{\text{bil}}
$$

where $\overline{w}_{\text{bil}}$ is the mean bilateral edge weight across the batch. This soft constraint discourages the model from always activating all bilateral connections uniformly.

---

## 7. Parameter Count

| Component | Parameters |
|---|---|
| Input projection (9→36) | 324 + 36 (bias/norm) |
| GCN layer 1 (36→36) | ~8,500 |
| GCN layer 2 (36→72) | ~18,000 |
| Bilateral edge MLPs (×2 layers, ×11 pairs) | ~25,000 |
| MS-TCN blocks (×2) | ~35,000 |
| Classifier (72→36→10) | ~3,000 |
| **Total** | **~100,840** |
