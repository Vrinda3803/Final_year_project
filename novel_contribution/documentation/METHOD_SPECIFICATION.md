# RA-STGT: METHOD SPECIFICATION AND MATHEMATICAL FORMULATION

**Project:** Reliability-Aware Spatio-Temporal Graph Transformer with Calibrated Selective Prediction  
**Target:** Autism Therapy Activity Recognition  
**Role:** Senior AI/ML Research Engineer & Computer Vision Researcher  
**Document Type:** Formal Method Specification & Engineering Whitepaper  

---

## 1. Problem Formulation and Input Representation

Let an input therapy recording be denoted by $\mathcal{S} = (\mathbf{X}, \mathbf{Q})$, where:
* $\mathbf{X} \in \mathbb{R}^{T \times V \times C}$ represents the normalized skeletal trajectory over $T = 64$ frames and $V = 25$ OpenPose `BODY_25` joints.
* $C = 4$ channels consisting of normalized 2D spatial coordinates and first-order inter-frame velocities:
  $$\mathbf{x}_{v,t} = [x_{v,t}, y_{v,t}, \Delta x_{v,t}, \Delta y_{v,t}]^\top$$
* $\mathbf{Q} \in [0, 1]^{T \times V}$ denotes the matrix of measured OpenPose detection confidences $c_{v,t}$ derived from the Part Affinity Fields (PAF) network.
* $y \in \{0, 1, \dots, K-1\}$ represents the ground-truth therapeutic activity label ($K = 10$ classes).

---

## 2. Mathematical Architecture

### Module 1: Dual Physical-Kinematic Reliability Gating

For every joint $v \in \{1, \dots, V\}$ at time step $t \in \{1, \dots, T\}$, raw coordinates are subject to detector dropout ($c_{v,t} \to 0$) and tracking jitter (false positive detections producing anatomically impossible jumps).

We formulate the **Kinematic Plausibility Prior** $v_{v,t}$:

$$v_{v,t} = \exp\left( - \frac{\|\Delta \mathbf{p}_{v,t}\|_2^2}{2 \sigma_v^2} \right)$$

where $\Delta \mathbf{p}_{v,t} = [x_{v,t} - x_{v,t-1}, y_{v,t} - y_{v,t-1}]^\top$, and $\sigma_v^2$ is the empirical velocity variance of joint $v$ estimated strictly on training-fold data where $c_{v,t} > 0.5$.

The **Joint Reliability Indicator** $r_{v,t} \in [0, 1]$ is the product of the detector confidence and kinematic prior:

$$r_{v,t} = c_{v,t} \cdot v_{v,t}$$

The **Frame-Level Reliability Scalar** $R_t \in [0, 1]$ aggregates joint reliability across the physical body:

$$R_t = \frac{1}{V} \sum_{v=1}^V r_{v,t}$$

---

### Module 2: Spatial Reliability-Gated Graph Attention (SR-GAT)

Let $\mathbf{A} \in \{0, 1\}^{V \times V}$ be the physical anatomical adjacency matrix defined by the 24 physiological bone connections of `BODY_25` with self-loops.

In spatial attention layer $l$, the node embeddings $\mathbf{H}^{(l)} \in \mathbb{R}^{B \times T \times V \times D}$ are projected to queries, keys, and values:

$$\mathbf{Q}^{(l)} = \mathbf{H}^{(l)} \mathbf{W}_Q^{(l)}, \quad \mathbf{K}^{(l)} = \mathbf{H}^{(l)} \mathbf{W}_K^{(l)}, \quad \mathbf{V}^{(l)} = \mathbf{H}^{(l)} \mathbf{W}_V^{(l)}$$

Raw spatial attention logits between target joint $i$ and source neighbor $j$ are computed as:

$$S_{ij,t} = \frac{\mathbf{q}_{i,t} \mathbf{k}_{j,t}^\top}{\sqrt{d_k}} + \mathcal{M}_{ij}$$

where $\mathcal{M}_{ij} = 0$ if $\mathbf{A}_{ij} = 1$ and $-\infty$ otherwise.

Standard graph attention normalizes solely over the topological neighbors. In **SR-GAT**, we modulate incoming message weights by the source node reliability $r_{j,t}$:

$$\tilde{\alpha}_{ij,t} = \frac{\exp(S_{ij,t}) \cdot (\delta + (1-\delta) r_{j,t})}{\sum_{k \in \mathcal{N}(i)} \exp(S_{ik,t}) \cdot (\delta + (1-\delta) r_{k,t}) + \epsilon}$$

where $\delta = 0.05$ guarantees a minimum gradient path.

The updated joint representation incorporates an explicit linear self-loop to preserve identity when all neighbors are occluded:

$$\mathbf{h}_{i,t}^{(l+1)} = \text{LayerNorm}\left( \mathbf{W}_{self}^{(l)} \mathbf{h}_{i,t}^{(l)} + \sum_{j \in \mathcal{N}(i)} \tilde{\alpha}_{ij,t} \mathbf{v}_{j,t}^{(l)} \right)$$

---

### Module 3: Spatial Aggregation & Reliability-Aware Temporal Attention (RATA)

To compress the spatial graph into a temporal sequence of frame tokens $\mathbf{z}_t$, we compute a reliability-weighted spatial pooling:

$$\mathbf{z}_t = \sum_{v=1}^V \left( \frac{r_{v,t}}{\sum_{u=1}^V r_{u,t} + \epsilon} \right) \mathbf{h}_{v,t}^{(L_{spatial})}$$

In the temporal attention module, we compute multi-head self-attention across frames $t, \tau \in \{1, \dots, T\}$.

To prevent completely corrupted or occluded frames (e.g. child obscured by a prop) from polluting temporal dynamics, the attention logits are modulated by the logarithmic reliability of the source frame $\tau$:

$$\tilde{A}_{t,\tau}^{temp} = \text{softmax}\left( \frac{\mathbf{q}_t^{temp} (\mathbf{k}_\tau^{temp})^\top}{\sqrt{d_k}} + \log(R_\tau + \epsilon) \right)$$

When $R_\tau \to 0$, $\log(R_\tau + \epsilon) \to -\infty$, which naturally zeroes out attention to frame $\tau$ while preserving sequence continuity.

The temporal sequence is pooled to a final clip descriptor $\mathbf{u} \in \mathbb{R}^D$:

$$\mathbf{u} = \sum_{t=1}^T \left( \frac{R_t}{\sum_{s=1}^T R_s + \epsilon} \right) \mathbf{z}_t^{(L_{temporal})}$$

---

### Module 4: Uncertainty Estimation & Calibrated Selective Prediction

The classification logits $\mathbf{z} \in \mathbb{R}^K$ are mapped to probabilities via temperature-calibrated softmax:

$$\hat{p}_k(T) = \frac{\exp(z_k / T)}{\sum_{j=1}^K \exp(z_j / T)}$$

where the scalar parameter $T^* > 0$ is learned on the inner validation set by minimizing the negative log-likelihood:

$$T^* = \arg\min_{T > 0} - \frac{1}{N_{val}} \sum_{i=1}^{N_{val}} \log \hat{p}_{y_i}(T)$$

#### Selective Classification Rule
In high-stakes clinical therapy assessment, making no prediction is preferable to making a confident error on a severely corrupted observation. We define the selective decision function with threshold $\tau \in (0, 1)$:

$$\hat{y}(x) = \begin{cases} \arg\max_k \hat{p}_k(T^*), & \text{if } \max_k \hat{p}_k(T^*) \geq \tau \\ \text{\textbf{ABSTAIN}}, & \text{otherwise} \end{cases}$$

This allows clinicians to review abstained sessions manually while guaranteeing high precision on retained automated assessments.
