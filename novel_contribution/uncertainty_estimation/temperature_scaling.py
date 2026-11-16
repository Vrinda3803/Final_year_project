"""
Temperature scaling for probability calibration of activity predictions.
Fitted strictly on held-out validation subjects to prevent test leakage.
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".codex-deps"))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar


class TemperatureScaler(nn.Module):
    """
    Learns a single temperature parameter T > 0 on validation fold logits
    to minimize Negative Log-Likelihood (NLL).
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def fit(self, val_logits: torch.Tensor | np.ndarray, val_labels: torch.Tensor | np.ndarray) -> float:
        """
        Optimizes T on validation logits.
        val_logits: (N, num_classes)
        val_labels: (N,)
        """
        if isinstance(val_logits, torch.Tensor):
            logits_np = val_logits.detach().cpu().numpy()
        else:
            logits_np = val_logits.copy()
            
        if isinstance(val_labels, torch.Tensor):
            labels_np = val_labels.detach().cpu().numpy()
        else:
            labels_np = val_labels.copy()

        def nll_objective(t_val: float) -> float:
            if t_val <= 0:
                return 1e9
            scaled = logits_np / t_val
            # Stable log-softmax
            max_l = np.max(scaled, axis=1, keepdims=True)
            log_sum_exp = max_l + np.log(np.sum(np.exp(scaled - max_l), axis=1, keepdims=True))
            log_probs = scaled - log_sum_exp
            nll = -np.mean(log_probs[np.arange(len(labels_np)), labels_np])
            return float(nll)

        res = minimize_scalar(nll_objective, bounds=(0.05, 10.0), method="bounded")
        opt_t = float(res.x)
        self.temperature.data = torch.tensor([opt_t], dtype=torch.float32)
        return opt_t

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scales logits by optimal temperature."""
        return logits / self.temperature.clamp_min(1e-4)

    def predict_proba(self, logits: torch.Tensor | np.ndarray) -> np.ndarray:
        """Returns calibrated probabilities."""
        if isinstance(logits, np.ndarray):
            logits_t = torch.tensor(logits, dtype=torch.float32)
        else:
            logits_t = logits
        scaled = self.forward(logits_t)
        probs = torch.softmax(scaled, dim=-1).detach().cpu().numpy()
        return probs
