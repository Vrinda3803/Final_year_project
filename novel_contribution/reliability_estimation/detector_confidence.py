"""
Extractor and validator for measured OpenPose detection confidence.
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".codex-deps"))

import torch
import numpy as np

# Anatomical body part groupings for OpenPose BODY_25
LIMB_GROUPS = {
    "torso": [1, 8],
    "right_arm": [2, 3, 4],
    "left_arm": [5, 6, 7],
    "right_leg": [9, 10, 11, 22, 23, 24],
    "left_leg": [12, 13, 14, 19, 20, 21],
    "head": [0, 15, 16, 17, 18]
}

def extract_detector_confidence(q: torch.Tensor | np.ndarray) -> torch.Tensor:
    """
    Validates and clips OpenPose PAF confidence scores into [0, 1].
    q: shape (B, T, V) or (T, V)
    """
    if isinstance(q, np.ndarray):
        q = torch.tensor(q, dtype=torch.float32)
    return torch.clamp(q, min=0.0, max=1.0)


def compute_limb_reliability(q: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Computes average confidence per anatomical limb group.
    q: shape (B, T, V)
    Returns: dict mapping limb_name -> Tensor of shape (B, T)
    """
    limb_scores = {}
    for name, joint_ids in LIMB_GROUPS.items():
        limb_scores[name] = q[..., joint_ids].mean(dim=-1)
    return limb_scores
