"""
Reliability-Conditioned Bilateral Spatio-Temporal Graph Convolutional Network (RC-BiGCN)

Novel Contribution:
1. Invariant Skeleton Representation: Body-centered (Mid-Hip root subtracted) and torso-scale normalized.
2. Reliability-Conditioned Bilateral Attention: Learns dynamic bilateral coordination (wrists, elbows, knees)
   softly modulated by joint confidence rather than zero-gating.
3. Multi-Scale Dilated Temporal Convolution (MS-TCN): Replaces overparameterized temporal self-attention with
   multi-receptive field temporal convolutions tailored for small-data generalisation.
4. Class-Balanced & Confusable Margin Loss: Alleviates the severe class imbalance (dr=25% vs mfs=7%).
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BODY25_PAIRS = [
    (1, 0), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7), (1, 8),
    (8, 9), (9, 10), (10, 11), (11, 24), (11, 22), (22, 23),
    (8, 12), (12, 13), (13, 14), (14, 21), (14, 19), (19, 20),
    (0, 15), (15, 17), (0, 16), (16, 18)
]

BILATERAL_PAIRS = [
    (4, 7),   # Wrists (Right Wrist, Left Wrist)
    (3, 6),   # Elbows
    (10, 13), # Knees
    (11, 14)  # Ankles
]

def build_adjacency_matrix(num_joints=25):
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    for i in range(num_joints):
        A[i, i] = 1.0
    for u, v in BODY25_PAIRS:
        A[u, v] = 1.0
        A[v, u] = 1.0
    # Degree normalize
    D = np.sum(A, axis=1)
    D_inv = np.power(D, -0.5, where=D > 0)
    D_inv[D <= 0] = 0.0
    return torch.tensor(D_inv[:, None] * A * D_inv[None, :], dtype=torch.float32)


class MultiScaleTemporalConv(nn.Module):
    """Multi-Scale Temporal Convolutional Block with Residual Connection."""
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7], dilations=[1, 2], stride=1, dropout=0.2):
        super().__init__()
        num_branches = len(kernel_sizes) * len(dilations)
        branch_channels = max(out_channels // num_branches, 8)
        
        branches = []
        for k in kernel_sizes:
            for d in dilations:
                pad = (k - 1) * d // 2
                branches.append(nn.Sequential(
                    nn.Conv2d(in_channels, branch_channels, (k, 1), stride=(stride, 1), padding=(pad, 0), dilation=(d, 1), bias=False),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout)
                ))
        self.branches = nn.ModuleList(branches)
        total_dim = branch_channels * len(branches)
        self.proj = nn.Sequential(
            nn.Conv2d(total_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        if stride != 1 or in_channels != out_channels:
            self.res = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.res = nn.Identity()
            
    def forward(self, x):
        feats = [b(x) for b in self.branches]
        cat = torch.cat(feats, dim=1)
        return F.relu(self.proj(cat) + self.res(x), inplace=True)


class ReliabilityConditionedBilateralConv(nn.Module):
    """
    Learns dynamic spatial edges between anatomically connected joints
    AND bilateral coordination pairs conditioned softly on joint reliability.
    """
    def __init__(self, channels, num_joints=25):
        super().__init__()
        self.num_joints = num_joints
        self.register_buffer('A_base', build_adjacency_matrix(num_joints))
        
        # Adaptive edge parameter
        self.A_learnable = nn.Parameter(torch.zeros(num_joints, num_joints))
        
        # Bilateral dynamic interaction MLP
        self.bilat_mlp = nn.Sequential(
            nn.Linear(channels * 2 + 2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )
        
    def forward(self, x, q):
        # x: (B, C, T, V), q: (B, T, V)
        B, C, T, V = x.shape
        A_fixed = self.A_base + self.A_learnable # (V, V)
        
        # Dynamic bilateral edges
        # Reshape to (B*T, V, C)
        x_perm = x.permute(0, 2, 3, 1).reshape(B * T, V, C)
        # Adapt q to the current temporal resolution T
        if q.shape[1] != T:
            # Interpolate or pool q along temporal dimension: (B, 1, T_orig, V) -> (B, 1, T, V)
            q_resampled = F.adaptive_avg_pool2d(q.unsqueeze(1), (T, V)).squeeze(1)
        else:
            q_resampled = q
        q_perm = q_resampled.reshape(B * T, V, 1)
        
        # Bilateral edge tensor
        A_bilat = torch.zeros(B * T, V, V, device=x.device, dtype=x.dtype)
        for u, v in BILATERAL_PAIRS:
            hu, hv = x_perm[:, u], x_perm[:, v]
            qu, qv = q_perm[:, u], q_perm[:, v]
            feat = torch.cat([hu, hv, qu, qv], dim=-1)
            weight = self.bilat_mlp(feat) # (B*T, 1)
            A_bilat[:, u, v] = weight.squeeze(-1)
            A_bilat[:, v, u] = weight.squeeze(-1)
            
        A_total = A_fixed.unsqueeze(0) + A_bilat # (B*T, V, V)
        
        # Message passing: (B*T, V, C)
        out = torch.bmm(A_total, x_perm) # (B*T, V, C)
        out = out.reshape(B, T, V, C).permute(0, 3, 1, 2) # (B, C, T, V)
        return F.relu(self.proj(out) + x, inplace=True)


class RC_BiGCN(nn.Module):
    """Complete Model with Invariant Preprocessing, Spatial Graph, MS-TCN, and Classifier."""
    def __init__(self, in_channels=5, num_classes=10, base_channels=64, num_joints=25, dropout=0.3):
        super().__init__()
        self.num_joints = num_joints
        
        # Input embedding: in_channels typically (x, y, vx, vy, q)
        self.input_embed = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Spatio-Temporal Blocks
        self.st1_spatial = ReliabilityConditionedBilateralConv(base_channels, num_joints)
        self.st1_temporal = MultiScaleTemporalConv(base_channels, base_channels, stride=1, dropout=dropout)
        
        self.st2_spatial = ReliabilityConditionedBilateralConv(base_channels, num_joints)
        self.st2_temporal = MultiScaleTemporalConv(base_channels, base_channels * 2, stride=2, dropout=dropout)
        
        self.st3_spatial = ReliabilityConditionedBilateralConv(base_channels * 2, num_joints)
        self.st3_temporal = MultiScaleTemporalConv(base_channels * 2, base_channels * 2, stride=2, dropout=dropout)
        
        # Global Pooling and Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_channels * 2, base_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(base_channels, num_classes)
        )
        
    def forward(self, x, q):
        # x: (B, T, V, C), q: (B, T, V)
        # 1. Invariant skeleton normalization
        coords = x[:, :, :, :2]
        root = coords[:, :, 8:9, :] # mid-hip (B, T, 1, 2)
        neck = coords[:, :, 1:2, :] # neck (B, T, 1, 2)
        torso = torch.norm(neck - root, dim=-1, keepdim=True)
        torso = torch.clamp(torso, min=1e-4)
        
        coords_norm = (coords - root) / torso
        vel_norm = torch.zeros_like(coords_norm)
        vel_norm[:, 1:] = coords_norm[:, 1:] - coords_norm[:, :-1]
        
        # Pack input tensor (B, Cin, T, V)
        inp = torch.cat([coords_norm, vel_norm, q.unsqueeze(-1)], dim=-1) # (B, T, V, 5)
        inp = inp.permute(0, 3, 1, 2).contiguous() # (B, 5, T, V)
        
        h = self.input_embed(inp)
        
        # Stage 1
        h = self.st1_spatial(h, q)
        h = self.st1_temporal(h)
        
        # Stage 2
        h = self.st2_spatial(h, q)
        h = self.st2_temporal(h)
        
        # Stage 3
        h = self.st3_spatial(h, q)
        h = self.st3_temporal(h)
        
        # Global Spatio-Temporal Average Pooling: (B, C)
        out = h.mean(dim=[-1, -2])
        return self.classifier(out)
