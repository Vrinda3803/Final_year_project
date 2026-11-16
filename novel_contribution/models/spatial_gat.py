"""
Spatial Graph Attention modules: Standard GAT and proposed Spatial Reliability-Gated GAT (SR-GAT).
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
from novel_contribution.data_processing.dataset_loader import get_adjacency_matrix


class SpatialReliabilityGatedAttention(nn.Module):
    """
    SR-GAT: Attenuates spatial message passing from low-reliability joints.
    Prevents occluded or noisy limb nodes from corrupting torso/adjacent features.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 2,
        dropout: float = 0.25,
        use_reliability: bool = True
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.use_reliability = use_reliability
        
        # Physical skeleton adjacency buffer
        self.register_buffer("adj", get_adjacency_matrix())
        
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
        
        # Self-loop transformation
        self_feat = self.self_proj(x)  # (B, T, V, out_dim)
        
        # Project queries, keys, values
        q = self.q_proj(x).view(b, t, v, h, hd).permute(0, 1, 3, 2, 4)  # (B, T, H, V, Dk)
        k = self.k_proj(x).view(b, t, v, h, hd).permute(0, 1, 3, 2, 4)  # (B, T, H, V, Dk)
        val = self.v_proj(x).view(b, t, v, h, hd).permute(0, 1, 3, 2, 4)  # (B, T, H, V, Dk)
        
        # Compute spatial attention matrix: (B, T, H, V, V)
        attn_logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # Mask with physical skeleton graph
        mask = self.adj.view(1, 1, 1, v, v) <= 0.0
        attn_logits = attn_logits.masked_fill(mask, -1e9)
        attn = F.softmax(attn_logits, dim=-1)  # (B, T, H, V_i, V_j)
        
        if self.use_reliability and r_joint is not None:
            # Gate message passing by source node reliability r_j
            # r_joint is (B, T, V), expand to (B, T, 1, 1, V_j)
            r_source = r_joint.unsqueeze(2).unsqueeze(3)  # (B, T, 1, 1, V)
            gated_attn = attn * (0.05 + 0.95 * r_source)  # Keep small epsilon path
            # Re-normalize across incoming neighbors
            denom = gated_attn.sum(dim=-1, keepdim=True).clamp_min(1e-5)
            attn = gated_attn / denom
            
        attn = self.dropout(attn)
        out = torch.matmul(attn, val)  # (B, T, H, V, Dk)
        out = out.permute(0, 1, 3, 2, 4).contiguous().view(b, t, v, self.out_dim)
        out = self.out_proj(out)
        
        # Residual connection with self-loop feature
        res = self.norm(self_feat + out)
        return F.relu(res)
