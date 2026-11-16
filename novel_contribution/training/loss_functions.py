"""
Loss functions: Class-balanced cross-entropy and focal loss for MMASD.
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".codex-deps"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ClassBalancedFocalLoss(nn.Module):
    """
    Focal loss with inverse-frequency class weights to counteract class imbalance.
    """
    def __init__(self, class_counts: np.ndarray | torch.Tensor, gamma: float = 1.5):
        super().__init__()
        if isinstance(class_counts, np.ndarray):
            counts = torch.tensor(class_counts, dtype=torch.float32)
        else:
            counts = class_counts.float()
            
        # Inverse class frequency weighting
        total = counts.sum()
        num_classes = len(counts)
        weights = total / (num_classes * counts.clamp_min(1.0))
        weights = weights / weights.mean()  # Normalize
        self.register_buffer("weights", weights)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, num_classes)
        targets: (B,)
        """
        ce_loss = F.cross_entropy(logits, targets, weight=self.weights, reduction="none")
        p = torch.exp(-ce_loss)
        focal_loss = ((1.0 - p) ** self.gamma) * ce_loss
        return focal_loss.mean()
