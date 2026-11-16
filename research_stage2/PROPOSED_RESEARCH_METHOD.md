# Stage 2: Proposed Research Method Specification (PR-STGT / RA-STGT v2)

**Model Name:** Phase-Relational & Reliability-Gated Spatio-Temporal Graph Transformer (**PR-STGT**)  
**Project:** Autism Therapy Physical Activity Recognition using 2D Skeleton Dynamics  
**Date:** March 2026  
**Scope:** Architectural, mathematical, and algorithmic specification of the proposed contribution.

---

## 1. Architectural Overview and Design Motivation

PR-STGT builds directly upon our Stage 1 findings to systematically resolve the three primary failure modes identified in our forensic error analysis:
1. **The 6-Hop Topological Bottleneck:** Traditional skeletal graphs require 6 message-passing hops between left and right wrists, preventing the network from recognizing bilateral hand synchronization. PR-STGT introduces an **Augmented Bi-Relational Topology** with direct 1-hop end-effector relational edges.
2. **Temporal Saccadic Mean Collapse:** Standard temporal attention averages out cyclic oscillations, confusing alternating antiphase movements (*Drumming*) with synchronous in-phase movements (*Sing and Clap*). PR-STGT integrates an explicit **Temporal Harmonic Phase Difference Stream**.
3. **Isotropic Cross-Entropy Decision Boundaries:** Standard cross-entropy does not enforce separation between fine-grained visual twins (*Maracas Shaking* vs. *Forward Shaking*). PR-STGT introduces a targeted **Confusable-Pair Margin Loss ($\mathcal{L}_{\text{CPM}}$)**.

```
                    Input 2D OpenPose Sequence: X in R^{B x T x 17 x 3}
                                     |
              +----------------------+----------------------+
              |                                             |
   Joint & Bone Kinematics                       Temporal Harmonic Phase
   x_{v,t} in R^6 (pos, bone, vel)               Delta phi_t = sgn(v_y^L * v_y^R)
              |                                             |
   Kinematic Velocity Prior                                 |
   r_{v,t} = c_{v,t} * exp(-alpha * ||Delta p||)             |
              |                                             |
   Augmented Bi-Relational Topology                         |
   E_aug = E_phys U E_rel (wrists, ankles)                   |
              |                                             |
   Relational Reliability Gating                            |
   R_{uv,t} = r_{u,t} * r_{v,t}                             |
              |                                             |
   Phase-Relational Spatial GAT (PR-GAT)                    |
   h_{v,t} in R^{64}                                        |
              |                                             |
   Frame-Level Aggregation                                  |
   H_t in R^{64}                                            |
              +----------------------+----------------------+
                                     |
                    Phase-Harmonic RATA (Temporal)
                    Key/Query conditioned on Delta phi_t
                    h_seq in R^{64}
                                     |
                           Linear Classifier
                                     |
                 Logits z in R^{10} + ECE Temperature T
                                     |
                Confusable-Pair Margin Loss: L_CE + lambda * L_CPM
```

---

## 2. Mathematical Formulation

### 2.1 Input Representation and Multi-Kinematic Embedding
For a video clip of length $T$, the skeleton is represented by 17 OpenPose joints $V = \{1, \dots, 17\}$. Each joint at frame $t$ possesses raw 2D coordinates and detection confidence:
$$p_{v,t}^{\text{raw}} = (x_{v,t}^{\text{raw}}, y_{v,t}^{\text{raw}}) \in \mathbb{R}^2, \quad c_{v,t} \in [0, 1]$$

Coordinates are centered relative to the mid-hip root joint $p_{\text{root}, t}$ and normalized by the torso bounding height:
$$p_{v,t} = \frac{p_{v,t}^{\text{raw}} - p_{\text{root}, t}}{\max(H_{\text{torso}, t}, \epsilon)}$$

From normalized coordinates, three kinematic streams are computed:
1. **Normalized Position:** $p_{v,t} = (x_{v,t}, y_{v,t}) \in \mathbb{R}^2$
2. **Bone Vector:** $b_{v,t} = p_{v,t} - p_{\text{parent}(v), t} \in \mathbb{R}^2$ (relative to kinematic parent)
3. **Temporal Velocity:** $v_{v,t} = \frac{p_{v,t} - p_{v,t-1}}{\Delta t} \in \mathbb{R}^2$

These are concatenated to form the node feature vector:
$$x_{v,t} = [p_{v,t} \,\|\, b_{v,t} \,\|\, v_{v,t}] \in \mathbb{R}^6$$

### 2.2 Relational Reliability Modeling
As in Stage 1, single-joint kinematic reliability combines detection confidence with a velocity-plausibility prior:
$$r_{v,t} = c_{v,t} \cdot \exp\left(-\alpha \frac{\|v_{v,t}\|_2}{v_{\max}}\right) \in [0, 1]$$
where $\alpha = 0.5$ and $v_{\max} = 1.0$.

For an interaction edge $(u, v) \in \mathcal{E}$, we introduce **Edge-Conditioned Relational Reliability**:
$$\tilde{\mathcal{R}}_{uv, t} = r_{u,t} \cdot r_{v,t} \in [0, 1]$$
This joint product guarantees that relational message passing between two joints is active if and only if *both* interacting end-effectors are tracked with high reliability.

### 2.3 Augmented Bi-Relational Topology ($\mathcal{G}_{\text{aug}}$)
Standard skeleton graphs contain 16 physical anatomical edges $\mathcal{E}_{\text{phys}}$. We augment this graph with functional bilateral end-effector edges $\mathcal{E}_{\text{rel}}$:
$$\mathcal{E}_{\text{aug}} = \mathcal{E}_{\text{phys}} \cup \mathcal{E}_{\text{rel}}$$
where:
$$\mathcal{E}_{\text{rel}} = \{(10, 9), (9, 10), (16, 15), (15, 16), (10, 0), (9, 0)\}$$
- $(10, 9)$: Left Wrist $\leftrightarrow$ Right Wrist
- $(16, 15)$: Left Ankle $\leftrightarrow$ Right Ankle
- $(10, 0), (9, 0)$: Left/Right Wrist $\leftrightarrow$ Nose (Head-Hand Interaction)

### 2.4 Phase-Relational Spatial Graph Attention (PR-GAT)
For each frame $t$, the node feature $x_{v,t} \in \mathbb{R}^6$ is projected into latent space via learnable transformation $W_0 \in \mathbb{R}^{d \times 6}$ ($d=64$):
$$h_{v,t}^{(0)} = W_0 x_{v,t} \cdot r_{v,t}$$

In PR-GAT, attention coefficients $\alpha_{uv, t}$ between connected joints $(u, v) \in \mathcal{E}_{\text{aug}}$ are modulated by edge type and relational reliability:
$$e_{uv, t} = \text{LeakyReLU}\left(a^T [W h_{u,t} \,\|\, W h_{v,t}] + \gamma_{\text{rel}} \cdot \mathbf{1}_{\{(u,v) \in \mathcal{E}_{\text{rel}}\}} \cdot \Delta d_{uv, t}\right)$$
where $\Delta d_{uv, t} = \|p_{u,t} - p_{v,t}\|_2$ is the normalized Euclidean distance between joints, and $\gamma_{\text{rel}}$ is a learnable scalar.

The attention weights are normalized across the augmented neighborhood $\mathcal{N}_{\text{aug}}(u)$ and gated by relational reliability $\tilde{\mathcal{R}}_{uv, t}$:
$$\alpha_{uv, t} = \frac{\exp(e_{uv, t})}{\sum_{k \in \mathcal{N}_{\text{aug}}(u)} \exp(e_{uk, t})}$$
$$\tilde{\alpha}_{uv, t} = \frac{\alpha_{uv, t} \cdot \tilde{\mathcal{R}}_{uv, t}}{\sum_{k \in \mathcal{N}_{\text{aug}}(u)} \alpha_{uk, t} \cdot \tilde{\mathcal{R}}_{uk, t} + \epsilon}$$

Node representations are updated via multi-head message passing with residual connections:
$$h_{u,t}^{(l)} = \sigma\left(\sum_{v \in \mathcal{N}_{\text{aug}}(u)} \tilde{\alpha}_{uv, t} W^{(l)} h_{v,t}^{(l-1)}\right) + h_{u,t}^{(l-1)}$$

After 2 PR-GAT layers, joint representations are pooled into a frame-level feature vector:
$$H_t = \sum_{v=1}^{17} r_{v,t} \cdot h_{v,t}^{(2)} \in \mathbb{R}^{64}$$

### 2.5 Temporal Harmonic Phase Difference Stream
To explicitly distinguish antiphase drumming from in-phase clapping, we compute the instantaneous vertical velocity correlation between left wrist ($v=10$) and right wrist ($v=9$):
$$\phi_t^{\text{raw}} = \dot{y}_{10, t} \cdot \dot{y}_{9, t}$$
The normalized harmonic phase signal is:
$$\Delta \phi_t = \tanh(10 \cdot \phi_t^{\text{raw}}) \in [-1, +1]$$
- **Antiphase motion (Drumming):** $\dot{y}_{10, t}$ and $\dot{y}_{9, t}$ have opposite signs $\implies \Delta \phi_t \approx -1$.
- **In-phase motion (Clapping):** $\dot{y}_{10, t}$ and $\dot{y}_{9, t}$ move together $\implies \Delta \phi_t \approx +1$.
- **Static pose (Tree / Twist):** $\dot{y}_{10, t} \approx 0 \implies \Delta \phi_t \approx 0$.

### 2.6 Phase-Harmonic Relational Adaptive Temporal Attention (Phase-RATA)
The frame-level feature sequence $H = [H_1, \dots, H_T] \in \mathbb{R}^{T \times 64}$ is concatenated with the harmonic phase signal $\Delta \phi \in \mathbb{R}^{T \times 1}$:
$$\tilde{H}_t = [H_t \,\|\, \Delta \phi_t] \in \mathbb{R}^{65}$$

Multi-head temporal self-attention computes query, key, and value matrices:
$$Q = \tilde{H} W_Q, \quad K = \tilde{H} W_K, \quad V = H W_V$$
Temporal attention weights $A \in \mathbb{R}^{T \times T}$ are modulated by the frame-level average reliability $\bar{r}_t = \frac{1}{17} \sum_{v=1}^{17} r_{v,t}$:
$$\tilde{A}_{i,j} = \text{Softmax}\left(\frac{Q_i K_j^T}{\sqrt{d_k}}\right) \cdot \bar{r}_j$$

The sequence is aggregated into a final activity embedding:
$$h_{\text{seq}} = \sum_{t=1}^T \beta_t \cdot \tilde{H}_t^{\text{trans}}, \quad \beta_t = \text{Softmax}(w_{\text{gate}}^T \tilde{H}_t \cdot \bar{r}_t)$$
where $h_{\text{seq}} \in \mathbb{R}^{64}$.

### 2.7 Confusable-Pair Margin Regularization Loss ($\mathcal{L}_{\text{CPM}}$)
Classification logits are generated via linear projection:
$$z = W_{\text{cls}} h_{\text{seq}} \in \mathbb{R}^{10}$$

The total training objective combines standard label-smoothed Cross-Entropy ($\mathcal{L}_{\text{CE}}$) with our targeted **Confusable-Pair Margin Loss ($\mathcal{L}_{\text{CPM}}$)**:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(z, y) + \lambda \mathcal{L}_{\text{CPM}}(z, y)$$

Let $P_{\text{conf}} = \{(\text{dr}, \text{sac}), (\text{ms}, \text{mfs})\}$ denote the empirically verified confusable class pairs. For a training sample with true class $y \in \{y_1, y_2\}$ where $(y_1, y_2) \in P_{\text{conf}}$, let $y_{\text{rival}}$ denote the corresponding twin confusable class:
$$\mathcal{L}_{\text{CPM}} = \begin{cases} \max\left(0, m - (z_y - z_{y_{\text{rival}}})\right) & \text{if } y \text{ belongs to a confusable pair} \\ 0 & \text{otherwise} \end{cases}$$
where:
- Margin threshold $m = 0.5$
- Loss weighting factor $\lambda = 0.25$

This loss directly forces the network to maintain a positive margin between twin classes without distorting the representations of unconfusable activities (e.g., *Squat* or *Tree Pose*).

---

## 3. Tensor Dimensions and Parameter Count Analysis

| Stage / Layer | Input Tensor Shape | Output Tensor Shape | Trainable Parameters | Description |
|---|---|---|---|---|
| **Input Feature Lift** | $(B, T, 17, 6)$ | $(B, T, 17, 64)$ | $6 \times 64 + 64 = 448$ | Linear coordinate-kinematic projection |
| **PR-GAT Layer 1** | $(B, T, 17, 64)$ | $(B, T, 17, 64)$ | $64 \times 64 \times 4 + \text{rel\_bias} = 16,450$ | 4-head augmented relational GAT |
| **PR-GAT Layer 2** | $(B, T, 17, 64)$ | $(B, T, 17, 64)$ | $64 \times 64 \times 4 + \text{rel\_bias} = 16,450$ | 4-head augmented relational GAT |
| **Frame Pooling** | $(B, T, 17, 64)$ | $(B, T, 64)$ | $0$ (Reliability-weighted sum) | Joint aggregation |
| **Phase Injection** | $(B, T, 64) \oplus (B, T, 1)$ | $(B, T, 65)$ | $0$ (Concatenation) | Incorporates bilateral phase velocity |
| **Phase-RATA (Temporal)**| $(B, T, 65)$ | $(B, T, 64)$ | $3 \times (65 \times 64) + 64 \times 64 = 16,576$ | 4-head phase-aware temporal transformer |
| **Temporal Gating** | $(B, T, 64)$ | $(B, 64)$ | $64 \times 1 + 1 = 65$ | Reliability-gated temporal pooling |
| **Classifier Head** | $(B, 64)$ | $(B, 10)$ | $64 \times 10 + 10 = 650$ | Linear classification projection |
| **LayerNorm / Residuals**| Various | Various | ~1,200 | Normalization parameters |
| **Total Model Parameters**| — | — | **~118,500** | **+6.1% vs RA-STGT v1 (111,658)** |

---

## 4. Algorithmic Workflow (Pseudocode)

```python
def forward_pr_stgt(x_coords, confidence, parent_indices, hard_pairs_margin=0.5):
    """
    x_coords: [B, T, 17, 2]
    confidence: [B, T, 17]
    """
    # 1. Kinematics & Reliability
    vel = compute_velocity(x_coords)               # [B, T, 17, 2]
    bone = x_coords - x_coords[:, :, parent_indices] # [B, T, 17, 2]
    kinematic_feat = concat([x_coords, bone, vel], dim=-1) # [B, T, 17, 6]
    r_node = confidence * exp(-alpha * norm(vel) / vmax) # [B, T, 17]
    
    # 2. Relational Edge Gating
    # E_aug includes (10,9), (16,15), (10,0), (9,0)
    r_edge = r_node[:, :, src] * r_node[:, :, dst] # [B, T, |E_aug|]
    
    # 3. Spatial PR-GAT
    h = linear_lift(kinematic_feat) * r_node.unsqueeze(-1)
    for gat_layer in pr_gat_layers:
        h = gat_layer(h, E_aug, r_edge) # Relational message passing
        
    # 4. Frame Pooling
    H_t = sum(h * r_node.unsqueeze(-1), dim=2) # [B, T, 64]
    
    # 5. Temporal Harmonic Phase Correlation
    # Joint 10: Left Wrist, Joint 9: Right Wrist
    v_y_left = vel[:, :, 10, 1]
    v_y_right = vel[:, :, 9, 1]
    delta_phi = tanh(10.0 * v_y_left * v_y_right).unsqueeze(-1) # [B, T, 1]
    
    # 6. Phase-RATA Temporal Attention
    H_phase = concat([H_t, delta_phi], dim=-1) # [B, T, 65]
    H_trans = temporal_transformer(H_phase, mean_reliability=r_node.mean(dim=2))
    h_seq = temporal_gating(H_trans) # [B, 64]
    
    # 7. Classification & Margin
    logits = classifier(h_seq) # [B, 10]
    return logits
```
