"""
PR-STGT: Phase-Relational & Reliability-Gated Spatio-Temporal Graph Transformer (RA-STGT v2).
Unified architecture supporting controlled single-factor ablation variants (E1 through E5).
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

from novel_contribution.reliability_estimation.velocity_prior import ReliabilityEstimator
from research_stage2.models.pr_gat import PhaseRelationalGatedAttention
from research_stage2.models.phase_rata import PhaseReliabilityAwareTemporalAttention, compute_harmonic_phase_difference


class PRSTGT(nn.Module):
    """
    PR-STGT (RA-STGT v2):
    Integrates:
      1. Single-joint kinematic reliability priors (r_v)
      2. Augmented Bi-Relational Topology & Edge-Conditioned Relational Reliability (r_uv = r_u * r_v)
      3. Instantaneous Bilateral Harmonic Phase Difference Stream (delta_phi)
      4. Phase-conditioned Relational Adaptive Temporal Attention (Phase-RATA)
      5. Selective prediction and ECE temperature scaling
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
        variant: str = "e5_full_pr_stgt"  # 'e1_ra_stgt', 'e2_topology_only', 'e3_phase_only', 'e4_margin_only', 'e5_full_pr_stgt'
    ):
        super().__init__()
        self.variant = variant
        self.d_model = d_model
        self.num_classes = num_classes
        
        # Configure ablation toggles based on variant
        if variant == "e1_ra_stgt":
            self.use_relational_edges = False
            self.use_relational_reliability = False
            self.use_phase_stream = False
        elif variant == "e2_topology_only":
            self.use_relational_edges = True
            self.use_relational_reliability = True
            self.use_phase_stream = False
        elif variant == "e3_phase_only":
            self.use_relational_edges = False
            self.use_relational_reliability = False
            self.use_phase_stream = True
        elif variant == "e4_margin_only":
            self.use_relational_edges = False
            self.use_relational_reliability = False
            self.use_phase_stream = False
        elif variant == "e5_full_pr_stgt":
            self.use_relational_edges = True
            self.use_relational_reliability = True
            self.use_phase_stream = True
        else:
            raise ValueError(f"Unknown PRSTGT variant: {variant}")
            
        # 1. Kinematic Reliability Estimator
        self.reliability_estimator = ReliabilityEstimator(num_joints=25)
        
        # 2. Coordinate Lift
        self.input_proj = nn.Linear(in_channels, d_model)
        
        # 3. Spatial PR-GAT layers
        self.spatial_layers = nn.ModuleList([
            PhaseRelationalGatedAttention(
                in_dim=d_model,
                out_dim=d_model,
                num_heads=num_spatial_heads,
                dropout=dropout,
                use_reliability=True,
                use_relational_edges=self.use_relational_edges,
                use_relational_reliability=self.use_relational_reliability
            )
            for _ in range(num_spatial_layers)
        ])
        
        # 4. Temporal Phase-RATA layers
        self.temporal_layers = nn.ModuleList([
            PhaseReliabilityAwareTemporalAttention(
                d_model=d_model,
                num_heads=num_temporal_heads,
                dropout=dropout,
                use_reliability=True,
                use_phase_stream=self.use_phase_stream
            )
            for _ in range(num_temporal_layers)
        ])
        
        # 5. Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Temperature parameter for ECE calibration
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        return_reliability: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, T, V, 4) - normalized coordinates and velocities
        q: (B, T, V) - OpenPose detection confidence
        """
        b, t, v, c = x.shape
        
        # Compute joint and frame reliabilities
        r_joint, r_frame = self.reliability_estimator(x, q)
        
        # Compute harmonic phase difference if enabled
        if self.use_phase_stream:
            phase_signal = compute_harmonic_phase_difference(x)  # (B, T, 1)
        else:
            phase_signal = None
            
        # Coordinate projection
        h = self.input_proj(x)  # (B, T, V, d_model)
        
        # Spatial PR-GAT message passing
        for layer in self.spatial_layers:
            h = layer(h, r_joint)
            
        # Spatial pooling weighted by joint reliability
        w_joint = r_joint / (r_joint.sum(dim=-1, keepdim=True).clamp_min(1e-5))
        frame_tokens = (h * w_joint.unsqueeze(-1)).sum(dim=2)  # (B, T, d_model)
        
        # Temporal Phase-RATA attention
        z = frame_tokens
        for layer in self.temporal_layers:
            z = layer(z, r_frame=r_frame, phase_signal=phase_signal)
            
        # Sequence pooling weighted by frame reliability
        w_frame = r_frame / (r_frame.sum(dim=-1, keepdim=True).clamp_min(1e-5))
        clip_rep = (z * w_frame.unsqueeze(-1)).sum(dim=1)  # (B, d_model)
        
        # Logits computation with temperature
        raw_logits = self.head(clip_rep)
        scaled_logits = raw_logits / self.temperature.clamp_min(1e-2)
        
        if return_reliability:
            return scaled_logits, r_joint, r_frame
        return scaled_logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for v in ["e1_ra_stgt", "e2_topology_only", "e3_phase_only", "e4_margin_only", "e5_full_pr_stgt"]:
        m = PRSTGT(variant=v)
        p = count_parameters(m)
        dummy_x = torch.randn(2, 60, 25, 4)
        dummy_q = torch.rand(2, 60, 25)
        out = m(dummy_x, dummy_q)
        print(f"Variant: {v:<18} | Params: {p:>7} | Output: {out.shape}")
