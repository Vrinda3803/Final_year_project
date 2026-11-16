"""
Calibration metrics: Expected Calibration Error (ECE), MCE, Brier Score, and reliability diagram stats.
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".codex-deps"))

import numpy as np
from typing import Dict, Any


def compute_calibration_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Computes ECE, MCE, Brier Score, and bin statistics.
    probs: (N, num_classes) - predicted probabilities
    labels: (N,) - true class indices
    """
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    accuracies = (preds == labels).astype(float)
    n = len(labels)
    
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_stats = []
    
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (confs >= low) & (confs < high if i < n_bins - 1 else confs <= high)
        count = int(np.sum(mask))
        if count > 0:
            bin_acc = float(np.mean(accuracies[mask]))
            bin_conf = float(np.mean(confs[mask]))
            gap = abs(bin_acc - bin_conf)
            ece += (count / n) * gap
            mce = max(mce, gap)
            bin_stats.append({
                "bin_idx": i,
                "low": float(low),
                "high": float(high),
                "center": float((low + high) / 2.0),
                "count": count,
                "accuracy": bin_acc,
                "confidence": bin_conf,
                "gap": gap
            })
        else:
            bin_stats.append({
                "bin_idx": i,
                "low": float(low),
                "high": float(high),
                "center": float((low + high) / 2.0),
                "count": 0,
                "accuracy": 0.0,
                "confidence": 0.0,
                "gap": 0.0
            })
            
    # Multiclass Brier score: mean squared difference from one-hot target
    one_hot = np.eye(probs.shape[1])[labels]
    brier_score = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    
    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier_score": brier_score,
        "mean_confidence": float(np.mean(confs)),
        "mean_accuracy": float(np.mean(accuracies)),
        "bin_stats": bin_stats
    }
