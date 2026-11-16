"""
PR-GAT: Phase-Relational and Reliability-Gated Spatial Graph Attention.
Extends SR-GAT with an Augmented Bi-Relational Topology and Edge-Conditioned Relational Reliability.
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".codex-deps"))
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from novel_contribution.data_processing.dataset_loader import BODY_25_EDGES

# OpenPose BODY_25 Key End-Effector Indices:
# Joint 4: RWrist, Joint 7: LWrist
# Joint 11: RAnkle, Joint 14: LAnkle
# Joint 0: Nose (Head Anchor)
RELATIONAL_EDGES = [
    (7, 4),   # LWrist <-> RWrist (Inter-hand bilateral coordination)
    (4, 7),
    (14, 11), # LAnkle <-> RAnkle (Bilateral leg coordination)
    (11, 14),
    (7, 0),   # LWrist <-> Nose (Hand-to-face spatial distance)
    (0, 7),
    (4, 0),   # RWrist <-> Nose
    (0, 4)
]


def get_augmented_adjacency_matrix(use_relational_edges: bool = True) -> torch.Tensor:
    """Returns normalized adjacency matrix for BODY_25, optionally augmented with bilateral relational edges."""
    adj = torch.eye(25, dtype=torch.float32)
    for i, j in BODY_25_EDGES:
        adj[i, j] = 1.0
        adj[j, i] = 1.0
        
    if use_relational_edges:
        for i, j in RELATIONAL_EDGES:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
            
    deg = adj.sum(dim=1).clamp_min(1.0).rsqrt()
    norm_adj = deg[:, None] * adj * deg[None, :]
    return norm_adj


class PhaseRelationalGatedAttention(nn.Module):
    """
    PR-GAT: Phase-Relational Spatial Graph Attention.
    Supports physical + augmented relational edges, with edge-conditioned relational reliability gating.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 2,
        dropout: float = 0.25,
        use_reliability: bool = True,
        use_relational_edges: bool = True,
        use_relational_reliability: bool = True
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.use_reliability = use_reliability
        self.use_relational_edges = use_relational_edges
        self.use_relational_reliability = use_relational_reliability
        
        # Augmented adjacency buffer
        self.register_buffer("adj", get_augmented_adjacency_matrix(use_relational_edges))
        
        # Linear projections
        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)
        self.self_proj = nn.Linear(in_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)
        
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = float(self.head_dim) ** -0.5

    def forward(self, x: torch.Tensor, r_joint: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, T, V, in_dim)
        r_joint: (B, T, V) - joint reliability in [0, 1]
        Returns: (B, T, V, out_dim)
        """
        b, t, v, d = x.shape
        h = self.num_heads
        hd = self.head_dim
        
        self_feat = self.self_proj(x)
        
        q = self.q_proj(x).view(b, t, v, h, hd).permute(0, 1, 3, 2, 4)  # (B, T, H, V, Dk)
        k = self.k_proj(x).view(b, t, v, h, hd).permute(0, 1, 3, 2, 4)  # (B, T, H, V, Dk)
        val = self.v_proj(x).view(b, t, v, h, hd).permute(0, 1, 3, 2, 4)  # (B, T, H, V, Dk)
        
        attn_logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, T, H, V, V)
        
        # Mask using augmented skeleton graph
        mask = self.adj.view(1, 1, 1, v, v) <= 0.0
        attn_logits = attn_logits.masked_fill(mask, -1e9)
        attn = F.softmax(attn_logits, dim=-1)  # (B, T, H, V_i, V_j)
        
        if self.use_reliability and r_joint is not None:
            if self.use_relational_reliability:
                # Edge-conditioned relational reliability: r_ij = r_i * r_j
                # r_joint is (B, T, V)
                r_i = r_joint.unsqueeze(2).unsqueeze(4)  # (B, T, 1, V, 1)
                r_j = r_joint.unsqueeze(2).unsqueeze(3)  # (B, T, 1, 1, V)
                r_edge = torch.sqrt(torch.clamp(r_i * r_j, min=1e-6))  # Geometric mean edge reliability
                gated_attn = attn * (0.05 + 0.95 * r_edge)
            else:
                # Single-node reliability gating
                r_source = r_joint.unsqueeze(2).unsqueeze(3)  # (B, T, 1, 1, V)
                gated_attn = attn * (0.05 + 0.95 * r_source)
                
            denom = gated_attn.sum(dim=-1, keepdim=True).clamp_min(1e-5)
            attn = gated_attn / denom
            
        attn = self.dropout(attn)
        out = torch.matmul(attn, val)  # (B, T, H, V, Dk)
        out = out.permute(0, 1, 3, 2, 4).contiguous().view(b, t, v, self.out_dim)
        out = self.out_proj(out)
        
        res = self.norm(self_feat + out)
        return F.relu(res)
