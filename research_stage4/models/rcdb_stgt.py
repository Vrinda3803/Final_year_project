"""
RCDB-STGT: Reliability-Conditioned Dynamic Bilateral Spatio-Temporal Graph Transformer
Stage 4 Model Architecture for Autism Therapy Physical Activity Recognition.

Features:
- Anatomical skeleton graph message passing on BODY_25 (24 bone edges).
- Learned dynamic bilateral edge strength a_uv(t) between contralateral limb pairs.
- Multiplicative edge-conditioned reliability gating g_uv(t) = q_uv(t) * a_uv(t).
- Temporal edge dynamics via causal/smooth 1D depthwise convolution over edge states.
- Supports all Stage 4 ablation configurations (A0 to A9).
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path("d:/Documents/Autism_Project")
sys.path.insert(0, str(ROOT / ".codex-deps"))
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from novel_contribution.reliability_estimation.velocity_prior import ReliabilityEstimator
from novel_contribution.data_processing.dataset_loader import BODY_25_EDGES
from research_stage2.models.phase_rata import compute_harmonic_phase_difference

# Standard OpenPose BODY_25 Bilateral Limb Homology Pairs
PAIR_DEFINITIONS = {
    "wrists": [(4, 7)],                 # RWrist <-> LWrist
    "ankles": [(11, 14)],               # RAnkle <-> LAnkle
    "elbows": [(3, 6)],                 # RElbow <-> LElbow
    "knees":  [(10, 13)],               # RKnee <-> LKnee
    "hips":   [(9, 12)],                # RHip <-> LHip
    "shoulders": [(2, 5)]               # RShoulder <-> LShoulder
}

# Standard E2 relational pairs: Wrists, Ankles, and Nose-to-wrist anchors
E2_DEFAULT_PAIRS = [(4, 7), (11, 14)]


def get_anatomical_adjacency(num_joints: int = 25) -> torch.Tensor:
    """Returns normalized physical adjacency matrix for standard OpenPose BODY_25."""
    adj = torch.eye(num_joints, dtype=torch.float32)
    for i, j in BODY_25_EDGES:
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    deg = adj.sum(dim=1).clamp_min(1.0).rsqrt()
    norm_adj = deg[:, None] * adj * deg[None, :]
    return norm_adj


class DynamicBilateralEdgeLearner(nn.Module):
    """
    Learns time-varying, symmetric bilateral edge strength a_uv(t) between paired joints.
    Conditioned on joint embeddings, interaction representations, and detection reliabilities.
    """
    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 32,
        use_temporal_smoothing: bool = True,
        temporal_kernel_size: int = 5,
        dropout: float = 0.10
    ):
        super().__init__()
        self.d_model = d_model
        self.use_temporal_smoothing = use_temporal_smoothing
        
        # Input features: [h_u, h_v, |h_u - h_v|, h_u * h_v, r_u, r_v] -> 4 * d_model + 2
        in_features = 4 * d_model + 2
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        if use_temporal_smoothing:
            # 1D Depthwise convolution to smooth bilateral dynamics across time
            self.temporal_conv = nn.Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=temporal_kernel_size,
                padding=temporal_kernel_size // 2,
                groups=1,
                bias=True
            )
        else:
            self.temporal_conv = None

    def forward(
        self,
        h_u: torch.Tensor,
        h_v: torch.Tensor,
        r_u: torch.Tensor,
        r_v: torch.Tensor
    ) -> torch.Tensor:
        """
        h_u, h_v: (B, T, d_model)
        r_u, r_v: (B, T, 1)
        Returns: a_uv: (B, T, 1) in [0, 1]
        """
        b, t, d = h_u.shape
        
        # Symmetric feature construction: invariant to order of u and v
        diff = torch.abs(h_u - h_v)
        prod = h_u * h_v
        
        feat_uv = torch.cat([h_u, h_v, diff, prod, r_u, r_v], dim=-1)
        feat_vu = torch.cat([h_v, h_u, diff, prod, r_v, r_u], dim=-1)
        feat_sym = 0.5 * (feat_uv + feat_vu)  # (B, T, 4*d + 2)
        
        # Raw edge logits
        logits = self.edge_mlp(feat_sym)  # (B, T, 1)
        
        # Optional temporal edge smoothing across time
        if self.temporal_conv is not None:
            # Conv1d expects (Batch, Channels, Time) -> (B, 1, T)
            logits_t = logits.transpose(1, 2)
            logits_smoothed = self.temporal_conv(logits_t)
            logits = logits_smoothed.transpose(1, 2)
            
        edge_strength = torch.sigmoid(logits)  # (B, T, 1)
        return edge_strength


class RCDB_SpatialAttention(nn.Module):
    """
    Reliability-Conditioned Dynamic Bilateral Spatial Graph Attention Layer.
    Combines:
      1. Anatomical physical graph attention over standard skeleton bones.
      2. Learned dynamic bilateral message passing over homologous contralateral joints.
      3. Multiplicative edge-conditioned reliability gating.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int = 2,
        dropout: float = 0.25,
        bilateral_pairs: list[tuple[int, int]] | None = None,
        use_dynamic_edges: bool = True,
        use_reliability_gate: bool = True,
        use_temporal_smoothing: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = float(self.head_dim) ** -0.5
        
        self.use_dynamic_edges = use_dynamic_edges
        self.use_reliability_gate = use_reliability_gate
        
        if bilateral_pairs is None:
            self.bilateral_pairs = E2_DEFAULT_PAIRS
        else:
            self.bilateral_pairs = bilateral_pairs
            
        # Anatomical Adjacency buffer (fixed physical bones)
        self.register_buffer("anat_adj", get_anatomical_adjacency(num_joints=25))
        
        # Anatomical multi-head attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.anat_out = nn.Linear(d_model, d_model)
        
        # Dynamic Bilateral edge learners (one per bilateral pair)
        if self.use_dynamic_edges:
            self.edge_learners = nn.ModuleList([
                DynamicBilateralEdgeLearner(
                    d_model=d_model,
                    hidden_dim=32,
                    use_temporal_smoothing=use_temporal_smoothing,
                    dropout=0.10
                )
                for _ in range(len(self.bilateral_pairs))
            ])
            # Bilateral message projection
            self.w_bi = nn.Linear(d_model, d_model)
        else:
            self.edge_learners = None
            self.w_bi = None
            
        # Integration parameters
        self.gamma_bi = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Buffer to store last computed bilateral edge strengths for interpretability analysis
        self.last_edge_weights: dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor, r_joint: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, V, d_model)
        r_joint: (B, T, V) - calibrated joint reliability
        Returns: (B, T, V, d_model)
        """
        b, t, v, d = x.shape
        h = self.num_heads
        hd = self.head_dim
        
        # -------------------------------------------------------------
        # 1. Anatomical Physical Graph Attention
        # -------------------------------------------------------------
        q = self.q_proj(x).view(b, t, v, h, hd).permute(0, 1, 3, 2, 4)  # (B, T, H, V, Dk)
        k = self.k_proj(x).view(b, t, v, h, hd).permute(0, 1, 3, 2, 4)
        val = self.v_proj(x).view(b, t, v, h, hd).permute(0, 1, 3, 2, 4)
        
        attn_logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, T, H, V, V)
        
        # Mask strictly by physical anatomical skeleton graph
        mask = self.anat_adj.view(1, 1, 1, v, v) <= 0.0
        attn_logits = attn_logits.masked_fill(mask, -1e9)
        attn = F.softmax(attn_logits, dim=-1)  # (B, T, H, V, V)
        
        # Scale anatomical attention by joint reliability
        if r_joint is not None:
            r_i = r_joint.unsqueeze(2).unsqueeze(4)  # (B, T, 1, V, 1)
            r_j = r_joint.unsqueeze(2).unsqueeze(3)  # (B, T, 1, 1, V)
            r_edge = torch.sqrt(torch.clamp(r_i * r_j, min=1e-6))
            gated_attn = attn * (0.05 + 0.95 * r_edge)
            attn = gated_attn / gated_attn.sum(dim=-1, keepdim=True).clamp_min(1e-5)
            
        attn = self.dropout(attn)
        anat_out = torch.matmul(attn, val).permute(0, 1, 3, 2, 4).contiguous().view(b, t, v, d)
        anat_out = self.anat_out(anat_out)
        
        # -------------------------------------------------------------
        # 2. Dynamic Bilateral Message Passing
        # -------------------------------------------------------------
        bi_messages = torch.zeros_like(x)
        self.last_edge_weights.clear()
        
        if self.use_dynamic_edges and self.edge_learners is not None:
            val_bi = self.w_bi(x)  # (B, T, V, d_model)
            
            for idx, (u, v_node) in enumerate(self.bilateral_pairs):
                h_u = x[:, :, u, :]         # (B, T, d)
                h_v = x[:, :, v_node, :]    # (B, T, d)
                r_u = r_joint[:, :, u:u+1]  # (B, T, 1)
                r_v = r_joint[:, :, v_node:v_node+1]
                
                # Learned dynamic relationship strength a_uv(t) in [0, 1]
                a_uv = self.edge_learners[idx](h_u, h_v, r_u, r_v)  # (B, T, 1)
                
                # Multiplicative edge-conditioned reliability gate q_uv(t)
                if self.use_reliability_gate:
                    q_uv = torch.sqrt(torch.clamp(r_u * r_v, min=1e-6))  # (B, T, 1)
                    g_uv = q_uv * a_uv
                else:
                    g_uv = a_uv
                    
                # Cache weights for interpretability and visualization
                pair_key = f"{u}_{v_node}"
                self.last_edge_weights[pair_key] = g_uv.detach()
                
                # Bidirectional bilateral messages: u <- v and v <- u
                bi_messages[:, :, u, :] += g_uv * val_bi[:, :, v_node, :]
                bi_messages[:, :, v_node, :] += g_uv * val_bi[:, :, u, :]
                
        # -------------------------------------------------------------
        # 3. Residual Integration & Normalization
        # -------------------------------------------------------------
        total_out = anat_out + self.gamma_bi * bi_messages
        out = self.norm(x + self.dropout(total_out))
        return F.relu(out)


class RCDB_TemporalAttention(nn.Module):
    """Temporal self-attention with causal/global attention over sequence length T."""
    def __init__(self, d_model: int, num_heads: int = 2, dropout: float = 0.25):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model)"""
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm_ffn(x + self.dropout(ffn_out))
        return x


class RCDB_STGT(nn.Module):
    """
    Reliability-Conditioned Dynamic Bilateral Spatio-Temporal Graph Transformer.
    Stage 4 Master Architecture for Autism Physical Therapy Recognition.
    """
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 10,
        d_model: int = 64,
        num_spatial_heads: int = 2,
        num_temporal_heads: int = 2,
        num_spatial_layers: int = 2,
        num_temporal_layers: int = 2,
        dropout: float = 0.25,
        bilateral_pairs: list[tuple[int, int]] | None = None,
        use_dynamic_edges: bool = True,
        use_reliability_gate: bool = True,
        use_temporal_smoothing: bool = True,
        use_phase_stream: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.use_dynamic_edges = use_dynamic_edges
        self.use_phase_stream = use_phase_stream
        
        # 1. Kinematic Velocity Reliability Estimator
        self.reliability_estimator = ReliabilityEstimator(num_joints=25)
        
        # 2. Input Coordinate Projection
        self.input_proj = nn.Linear(in_channels, d_model)
        
        # 3. Spatial Layers: RCDB-GAT
        self.spatial_layers = nn.ModuleList([
            RCDB_SpatialAttention(
                d_model=d_model,
                num_heads=num_spatial_heads,
                dropout=dropout,
                bilateral_pairs=bilateral_pairs,
                use_dynamic_edges=use_dynamic_edges,
                use_reliability_gate=use_reliability_gate,
                use_temporal_smoothing=use_temporal_smoothing
            )
            for _ in range(num_spatial_layers)
        ])
        
        # 4. Optional Harmonic Phase Stream Projection
        if self.use_phase_stream:
            self.phase_proj = nn.Linear(1, d_model)
        else:
            self.phase_proj = None
            
        # 5. Temporal Attention Layers
        self.temporal_layers = nn.ModuleList([
            RCDB_TemporalAttention(d_model=d_model, num_heads=num_temporal_heads, dropout=dropout)
            for _ in range(num_temporal_layers)
        ])
        
        # 6. Classifier & Temperature Parameter
        self.classifier = nn.Linear(d_model, num_classes)
        self.temperature = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=False)

    def forward(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, V, 4) - coordinates and velocities
        q: (B, T, V)    - detector confidence
        Returns: logits: (B, num_classes)
        """
        b, t, v, _ = x.shape
        
        # Calculate joint reliability r(t) in [0, 1]
        r_joint, _ = self.reliability_estimator(x, q)  # (B, T, V)
        
        # Lift input coordinates to d_model
        h = self.input_proj(x)  # (B, T, V, d_model)
        
        # Pass through Spatial Graph Layers
        for layer in self.spatial_layers:
            h = layer(h, r_joint)
            
        # Spatial pooling across 25 joints weighted by joint reliability
        r_weights = r_joint.unsqueeze(-1)  # (B, T, V, 1)
        h_pooled = (h * r_weights).sum(dim=2) / r_weights.sum(dim=2).clamp_min(1e-5)  # (B, T, d_model)
        
        # Optional: Add Phase Stream
        if self.use_phase_stream and self.phase_proj is not None:
            delta_phi = compute_harmonic_phase_difference(x)  # (B, T, 1)
            h_pooled = h_pooled + self.phase_proj(delta_phi)
            
        # Temporal Attention Layers
        for t_layer in self.temporal_layers:
            h_pooled = t_layer(h_pooled)
            
        # Sequence-level temporal pooling (mean over frames)
        z = h_pooled.mean(dim=1)  # (B, d_model)
        logits = self.classifier(z)  # (B, num_classes)
        return logits

    def get_last_bilateral_weights(self) -> dict[str, torch.Tensor]:
        """Returns the dictionary of bilateral edge strengths from the last spatial layer."""
        if len(self.spatial_layers) > 0 and self.spatial_layers[-1].last_edge_weights:
            return self.spatial_layers[-1].last_edge_weights
        return {}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device("cpu")
    print("Testing RCDB_STGT model initialization...")
    model = RCDB_STGT(
        in_channels=4,
        num_classes=10,
        d_model=64,
        use_dynamic_edges=True,
        use_reliability_gate=True,
        use_temporal_smoothing=True,
        use_phase_stream=False
    ).to(device)
    
    n_params = count_parameters(model)
    print(f"RCDB_STGT Parameter Count: {n_params:,}")
    
    x = torch.randn(2, 64, 25, 4, device=device)
    q = torch.rand(2, 64, 25, device=device)
    
    out = model(x, q)
    print(f"Forward pass output shape: {out.shape}")
    edge_weights = model.get_last_bilateral_weights()
    print(f"Captured Bilateral Edges: {list(edge_weights.keys())}")
    for k, v in edge_weights.items():
        print(f"  Edge {k} shape: {v.shape}, Mean Strength: {v.mean().item():.4f}")
    print("RCDB_STGT test passed successfully!")
