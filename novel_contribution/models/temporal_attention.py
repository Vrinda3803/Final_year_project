"""
Temporal Attention modules: Standard Transformer Temporal Attention and proposed RATA.
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".codex-deps"))

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReliabilityAwareTemporalAttention(nn.Module):
    """
    RATA: Multi-head temporal self-attention modulated by frame-level reliability R_t.
    Downweights corrupted or occluded frames without breaking sequential temporal order.
    """
    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 2,
        dropout: float = 0.25,
        use_reliability: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_reliability = use_reliability
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = float(self.head_dim) ** -0.5

    def forward(self, x: torch.Tensor, r_frame: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, T, d_model) - sequence of frame embeddings
        r_frame: (B, T) - frame reliability scalar in [0, 1]
        Returns: (B, T, d_model)
        """
        b, t, d = x.shape
        h = self.num_heads
        hd = self.head_dim
        
        q = self.q_proj(x).view(b, t, h, hd).transpose(1, 2)  # (B, H, T, Dk)
        k = self.k_proj(x).view(b, t, h, hd).transpose(1, 2)  # (B, H, T, Dk)
        val = self.v_proj(x).view(b, t, h, hd).transpose(1, 2)  # (B, H, T, Dk)
        
        attn_logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, H, T, T_tau)
        
        if self.use_reliability and r_frame is not None:
            # Modulate attention logits by target frame log-reliability
            # r_frame is (B, T), expand to (B, 1, 1, T_tau)
            r_tau = r_frame.unsqueeze(1).unsqueeze(2).clamp_min(1e-4)  # (B, 1, 1, T)
            attn_logits = attn_logits + torch.log(r_tau)
            
        attn = F.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, val)  # (B, H, T, Dk)
        out = out.transpose(1, 2).contiguous().view(b, t, d)
        out = self.out_proj(out)
        
        # Residual + Norm
        x = self.norm1(x + self.dropout(out))
        # FFN + Norm
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x
