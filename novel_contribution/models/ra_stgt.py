"""
Full RA-STGT Architecture and controlled ablation variants.
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
from novel_contribution.models.spatial_gat import SpatialReliabilityGatedAttention
from novel_contribution.models.temporal_attention import ReliabilityAwareTemporalAttention


class RASTGT(nn.Module):
    """
    RA-STGT: Reliability-Aware Spatio-Temporal Graph Transformer.
    Integrates joint-level kinematic reliability estimation, spatial reliability-gated
    graph attention (SR-GAT), and reliability-aware temporal attention (RATA).
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
        mode: str = "full"  # 'full', 'spatial_only', 'temporal_only', 'baseline_stgt', 'naive_concat'
    ):
        super().__init__()
        self.mode = mode
        self.d_model = d_model
        self.num_classes = num_classes
        
        # Reliability estimator
        self.reliability_estimator = ReliabilityEstimator(num_joints=25)
        
        # Determine reliability usage based on ablation mode
        self.use_spatial = mode in ("full", "spatial_only")
        self.use_temporal = mode in ("full", "temporal_only")
        actual_in_channels = in_channels + 1 if mode == "naive_concat" else in_channels
        
        # Initial joint feature projection
        self.input_proj = nn.Linear(actual_in_channels, d_model)
        
        # Spatial SR-GAT layers
        self.spatial_layers = nn.ModuleList([
            SpatialReliabilityGatedAttention(
                in_dim=d_model,
                out_dim=d_model,
                num_heads=num_spatial_heads,
                dropout=dropout,
                use_reliability=self.use_spatial
            )
            for _ in range(num_spatial_layers)
        ])
        
        # Temporal RATA layers
        self.temporal_layers = nn.ModuleList([
            ReliabilityAwareTemporalAttention(
                d_model=d_model,
                num_heads=num_temporal_heads,
                dropout=dropout,
                use_reliability=self.use_temporal
            )
            for _ in range(num_temporal_layers)
        ])
        
        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(
        self, 
        x: torch.Tensor, 
        q: torch.Tensor,
        return_reliability: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, T, V, 4) - normalized coordinates and velocities
        q: (B, T, V) - OpenPose detection confidence
        
        Returns:
            logits: (B, num_classes)
            (optional) r_joint: (B, T, V), r_frame: (B, T)
        """
        b, t, v, c = x.shape
        
        # Step 1: Compute reliability indicators
        r_joint, r_frame = self.reliability_estimator(x, q)
        
        # Input preparation
        if self.mode == "naive_concat":
            # Concat raw confidence q as a 5th channel
            x_in = torch.cat([x, q.unsqueeze(-1)], dim=-1)
        else:
            x_in = x
            
        h = self.input_proj(x_in)  # (B, T, V, d_model)
        
        # Step 2: Spatial graph message passing (SR-GAT)
        for layer in self.spatial_layers:
            h = layer(h, r_joint if self.use_spatial else None)
            
        # Step 3: Spatial aggregation to frame tokens
        if self.use_spatial:
            # Weighted average by joint reliability
            w = r_joint / (r_joint.sum(dim=-1, keepdim=True).clamp_min(1e-5))
            frame_tokens = (h * w.unsqueeze(-1)).sum(dim=2)  # (B, T, d_model)
        else:
            frame_tokens = h.mean(dim=2)  # (B, T, d_model)
            
        # Step 4: Temporal attention (RATA)
        z = frame_tokens
        for layer in self.temporal_layers:
            z = layer(z, r_frame if self.use_temporal else None)
            
        # Step 5: Sequence pooling to clip representation
        if self.use_temporal:
            w_t = r_frame / (r_frame.sum(dim=-1, keepdim=True).clamp_min(1e-5))
            clip_rep = (z * w_t.unsqueeze(-1)).sum(dim=1)  # (B, d_model)
        else:
            clip_rep = z.mean(dim=1)  # (B, d_model)
            
        # Step 6: Classification logits
        logits = self.head(clip_rep)
        
        if return_reliability:
            return logits, r_joint, r_frame
        return logits


if __name__ == "__main__":
    print("Testing RA-STGT architectures across all ablation modes...")
    modes = ["full", "spatial_only", "temporal_only", "baseline_stgt", "naive_concat"]
    dummy_x = torch.randn(2, 64, 25, 4)
    dummy_q = torch.rand(2, 64, 25)
    
    for m in modes:
        model = RASTGT(in_channels=4, num_classes=10, d_model=64, mode=m)
        out = model(dummy_x, dummy_q)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Mode: {m:<15} | Output shape: {out.shape} | Parameters: {n_params:,}")
        assert out.shape == (2, 10), f"Incorrect output shape for mode {m}!"
    print("All architecture tests passed.")
