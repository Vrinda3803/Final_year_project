"""
Phase-RATA: Temporal Harmonic Phase Difference Stream & Relational Adaptive Temporal Attention.
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


def compute_harmonic_phase_difference(x: torch.Tensor) -> torch.Tensor:
    """
    Computes bilateral instantaneous phase difference between Left Wrist and Right Wrist.
    In OpenPose BODY_25:
      Joint 4: Right Wrist
      Joint 7: Left Wrist
    x: (B, T, V, 4), where channel 3 is normalized vertical velocity (vy).
    Returns:
      delta_phi: (B, T, 1) in [-1.0, +1.0]
        -1.0: Antiphase movement (Drumming: one hand up, other down)
        +1.0: In-phase movement (Sing and Clap: hands moving synchronously)
         0.0: Stationary pose (Tree pose, Twist pose)
    """
    vy_rwrist = x[:, :, 4, 3]  # (B, T)
    vy_lwrist = x[:, :, 7, 3]  # (B, T)
    
    # Velocity product: positive if moving in same direction, negative if opposite
    phase_product = vy_rwrist * vy_lwrist  # (B, T)
    delta_phi = torch.tanh(10.0 * phase_product).unsqueeze(-1)  # (B, T, 1)
    return delta_phi


class PhaseReliabilityAwareTemporalAttention(nn.Module):
    """
    Phase-RATA: Integrates bilateral harmonic phase differences with frame-level reliability-weighted
    temporal self-attention.
    """
    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 2,
        dropout: float = 0.25,
        use_reliability: bool = True,
        use_phase_stream: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_reliability = use_reliability
        self.use_phase_stream = use_phase_stream
        
        # Optional projection for harmonic phase stream
        if use_phase_stream:
            self.phase_proj = nn.Linear(1, d_model)
        else:
            self.phase_proj = None
            
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = float(self.head_dim) ** -0.5

    def forward(
        self, 
        z: torch.Tensor, 
        r_frame: torch.Tensor | None = None,
        phase_signal: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        z: (B, T, d_model) - sequence of frame tokens
        r_frame: (B, T) - mean frame reliability in [0, 1]
        phase_signal: (B, T, 1) - harmonic phase difference in [-1, +1]
        Returns: (B, T, d_model)
        """
        b, t, d = z.shape
        h = self.num_heads
        hd = self.head_dim
        
        # Inject harmonic phase signal if enabled
        if self.use_phase_stream and self.phase_proj is not None and phase_signal is not None:
            phase_emb = self.phase_proj(phase_signal)  # (B, T, d_model)
            z_in = z + phase_emb
        else:
            z_in = z
            
        q = self.q_proj(z_in).view(b, t, h, hd).transpose(1, 2)  # (B, H, T, Dk)
        k = self.k_proj(z_in).view(b, t, h, hd).transpose(1, 2)  # (B, H, T, Dk)
        val = self.v_proj(z).view(b, t, h, hd).transpose(1, 2)    # (B, H, T, Dk)
        
        attn_logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, H, T, T)
        attn = F.softmax(attn_logits, dim=-1)  # (B, H, T_i, T_j)
        
        if self.use_reliability and r_frame is not None:
            # Gate attention by key frame reliability r_j
            r_key = r_frame.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            gated_attn = attn * (0.05 + 0.95 * r_key)
            denom = gated_attn.sum(dim=-1, keepdim=True).clamp_min(1e-5)
            attn = gated_attn / denom
            
        attn = self.dropout(attn)
        out = torch.matmul(attn, val)  # (B, H, T, Dk)
        out = out.transpose(1, 2).contiguous().view(b, t, d)
        out = self.out_proj(out)
        
        res = self.norm(z + out)
        return F.relu(res)
