"""
Selective prediction and risk-coverage curve computation for clinical abstention.
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".codex-deps"))

import numpy as np
from typing import Dict, Any, List


def evaluate_selective_prediction(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: List[float] | None = None
) -> Dict[str, Any]:
    """
    Evaluates selective classification with abstention across confidence thresholds.
    probs: (N, num_classes)
    labels: (N,)
    """
    if thresholds is None:
        thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    n_total = len(labels)
    num_classes = probs.shape[1]
    
    curve = []
    for tau in thresholds:
        accepted_mask = confs >= tau
        n_accepted = int(np.sum(accepted_mask))
        coverage = n_accepted / n_total
        
        if n_accepted > 0:
            acc = float(np.mean(preds[accepted_mask] == labels[accepted_mask]))
            # Compute macro-F1 on accepted
            f1_list = []
            for c in range(num_classes):
                tp = np.sum((labels[accepted_mask] == c) & (preds[accepted_mask] == c))
                fp = np.sum((labels[accepted_mask] != c) & (preds[accepted_mask] == c))
                fn = np.sum((labels[accepted_mask] == c) & (preds[accepted_mask] != c))
                denom = 2 * tp + fp + fn
                f1_list.append(float(2 * tp / denom) if denom > 0 else 0.0)
            macro_f1 = float(np.mean(f1_list))
        else:
            acc = 1.0
            macro_f1 = 1.0
            
        curve.append({
            "threshold": float(tau),
            "coverage": float(coverage),
            "n_accepted": n_accepted,
            "n_abstained": n_total - n_accepted,
            "selective_accuracy": acc,
            "selective_macro_f1": macro_f1,
            "risk": 1.0 - acc
        })
        
    return {
        "n_total": n_total,
        "curve": curve
    }
