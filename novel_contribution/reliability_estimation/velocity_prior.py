"""
Biomechanical velocity consistency prior and combined joint/frame reliability computation.
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".codex-deps"))

import torch
import torch.nn as nn
import numpy as np


class ReliabilityEstimator(nn.Module):
    """
    Computes joint-level reliability r_{v,t} and frame-level reliability R_t
    by combining OpenPose PAF confidence with kinematic displacement plausibility.
    """
    def __init__(self, num_joints: int = 25, default_sigma: float = 0.5):
        super().__init__()
        self.num_joints = num_joints
        # Log-variance per joint, initialized to log(default_sigma^2)
        init_log_var = torch.full((num_joints,), float(np.log(default_sigma**2)), dtype=torch.float32)
        self.register_buffer("log_var", init_log_var)

    def fit_velocity_priors(self, X_train: torch.Tensor | np.ndarray, Q_train: torch.Tensor | np.ndarray):
        """
        Estimates empirical per-joint velocity variance sigma_v^2 on clean training detections only.
        X_train: (N, T, V, 4), Q_train: (N, T, V)
        """
        if isinstance(X_train, np.ndarray):
            X_train = torch.tensor(X_train, dtype=torch.float32)
        if isinstance(Q_train, np.ndarray):
            Q_train = torch.tensor(Q_train, dtype=torch.float32)
            
        # Velocities are in channels 2:4 (dx, dy)
        vel = X_train[..., 2:4]  # (N, T, V, 2)
        disp_sq = (vel ** 2).sum(dim=-1)  # (N, T, V)
        
        # Filter for confident detections (Q > 0.5) to avoid fitting on tracking noise
        mask = Q_train > 0.5
        variances = []
        for v in range(self.num_joints):
            v_mask = mask[..., v]
            if v_mask.sum() > 10:
                v_disp = disp_sq[..., v][v_mask]
                var = torch.var(v_disp, unbiased=True).clamp_min(1e-4)
            else:
                var = torch.tensor(0.25, dtype=torch.float32)
            variances.append(var)
            
        var_tensor = torch.stack(variances)
        self.log_var.copy_(torch.log(var_tensor))

    def forward(self, x: torch.Tensor, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes joint- and frame-level reliability.
        x: (B, T, V, 4) - normalized coordinates and velocities
        q: (B, T, V) - measured OpenPose detection confidence in [0, 1]
        
        Returns:
            r_joint: (B, T, V) - joint reliability in [0, 1]
            r_frame: (B, T) - frame reliability in [0, 1]
        """
        b, t, v, c = x.shape
        # Velocity displacement squared
        dx_dy = x[..., 2:4]
        disp_sq = (dx_dy ** 2).sum(dim=-1)  # (B, T, V)
        
        # Velocity plausibility: exp(-disp_sq / (2 * var_v))
        var = torch.exp(self.log_var).view(1, 1, v)  # (1, 1, V)
        v_prior = torch.exp(-disp_sq / (2.0 * var + 1e-6))  # (B, T, V)
        
        # Combined joint reliability
        c_conf = torch.clamp(q, min=0.0, max=1.0)
        r_joint = c_conf * v_prior  # (B, T, V)
        
        # Frame-level reliability summary
        r_frame = r_joint.mean(dim=-1)  # (B, T)
        
        return r_joint, r_frame


if __name__ == "__main__":
    print("Testing ReliabilityEstimator...")
    estimator = ReliabilityEstimator(num_joints=25)
    dummy_x = torch.randn(4, 64, 25, 4)
    dummy_q = torch.rand(4, 64, 25)
    r_j, r_f = estimator(dummy_x, dummy_q)
    print("Joint reliability shape:", r_j.shape, "min:", r_j.min().item(), "max:", r_j.max().item())
    print("Frame reliability shape:", r_f.shape, "min:", r_f.min().item(), "max:", r_f.max().item())
    assert (r_j >= 0.0).all() and (r_j <= 1.0).all(), "r_j out of bounds!"
    assert (r_f >= 0.0).all() and (r_f <= 1.0).all(), "r_f out of bounds!"
    print("ReliabilityEstimator unit test passed.")
