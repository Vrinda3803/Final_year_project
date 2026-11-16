"""
Loss functions for PR-STGT: Class-Balanced Focal Loss combined with Confusable-Pair Margin Loss (L_CPM).
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
import numpy as np

# Class mapping:
# 0: as, 1: bs, 2: ce, 3: dr, 4: mfs, 5: ms, 6: sac, 7: sq, 8: tr, 9: tw
CONFUSABLE_TWINS = {
    3: 6,  # dr (Drumming) <-> sac (Sing and Clap)
    6: 3,  # sac <-> dr
    4: 5,  # mfs (Maracas Forward Shaking) <-> ms (Maracas Shaking)
    5: 4   # ms <-> mfs
}


class ConfusablePairMarginLoss(nn.Module):
    """
    Penalizes insufficient margin separation between empirically confusable therapy exercise twins:
    1. (Drumming, Sing-and-Clap)
    2. (Maracas Shaking, Maracas Forward Shaking)
    """
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, num_classes)
        targets: (B,) class indices
        """
        batch_size = logits.size(0)
        losses = []
        
        for i in range(batch_size):
            y = int(targets[i].item())
            if y in CONFUSABLE_TWINS:
                y_twin = CONFUSABLE_TWINS[y]
                # Margin condition: logits[i, y] - logits[i, y_twin] >= margin
                logit_diff = logits[i, y] - logits[i, y_twin]
                loss_i = F.relu(self.margin - logit_diff)
                losses.append(loss_i)
                
        if len(losses) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
        return torch.stack(losses).mean()


class PRSTGTLoss(nn.Module):
    """
    Combined objective for PR-STGT:
    L_total = L_CBFL(logits, y; gamma=1.2) + lambda_margin * L_CPM(logits, y; margin=0.5)
    """
    def __init__(
        self,
        class_counts: np.ndarray | torch.Tensor | None = None,
        gamma: float = 1.2,
        use_margin_loss: bool = True,
        margin: float = 0.5,
        lambda_margin: float = 0.25
    ):
        super().__init__()
        self.gamma = gamma
        self.use_margin_loss = use_margin_loss
        self.lambda_margin = lambda_margin
        
        if class_counts is not None:
            if isinstance(class_counts, np.ndarray):
                counts = torch.tensor(class_counts, dtype=torch.float32)
            else:
                counts = class_counts.float()
            total = counts.sum()
            num_classes = len(counts)
            weights = total / (num_classes * counts.clamp_min(1.0))
            weights = weights / weights.mean()
            self.register_buffer("weights", weights)
        else:
            self.weights = None
            
        if use_margin_loss:
            self.cpm_loss = ConfusablePairMarginLoss(margin=margin)
        else:
            self.cpm_loss = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        # 1. Base focal loss
        if self.weights is not None:
            ce_loss = F.cross_entropy(logits, targets, weight=self.weights, reduction="none")
        else:
            ce_loss = F.cross_entropy(logits, targets, reduction="none")
            
        p = torch.exp(-ce_loss)
        focal_loss = (((1.0 - p) ** self.gamma) * ce_loss).mean()
        loss_dict = {"base_loss": float(focal_loss.item())}
        
        # 2. Confusable-pair margin loss
        if self.use_margin_loss and self.cpm_loss is not None:
            cpm = self.cpm_loss(logits, targets)
            total_loss = focal_loss + self.lambda_margin * cpm
            loss_dict["cpm_loss"] = float(cpm.item())
            loss_dict["total_loss"] = float(total_loss.item())
        else:
            total_loss = focal_loss
            loss_dict["cpm_loss"] = 0.0
            loss_dict["total_loss"] = float(focal_loss.item())
            
        return total_loss, loss_dict
