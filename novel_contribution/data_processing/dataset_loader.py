"""
Data loader and zero-leakage subject-independent partitioning for MMASD OpenPose skeletons.
"""
from __future__ import annotations
import sys, os
from pathlib import Path
from typing import Tuple, List, Dict, Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".codex-deps"))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# OpenPose BODY_25 anatomical skeleton edges (24 connections)
BODY_25_EDGES = [
    (1, 0),   # Neck -> Nose
    (1, 2),   # Neck -> RShoulder
    (2, 3),   # RShoulder -> RElbow
    (3, 4),   # RElbow -> RWrist
    (1, 5),   # Neck -> LShoulder
    (5, 6),   # LShoulder -> LElbow
    (6, 7),   # LElbow -> LWrist
    (1, 8),   # Neck -> MidHip
    (8, 9),   # MidHip -> RHip
    (9, 10),  # RHip -> RKnee
    (10, 11), # RKnee -> RAnkle
    (8, 12),  # MidHip -> LHip
    (12, 13), # LHip -> LKnee
    (13, 14), # LKnee -> LAnkle
    (0, 15),  # Nose -> REye
    (15, 17), # REye -> REar
    (0, 16),  # Nose -> LEye
    (16, 18), # LEye -> LEar
    (14, 19), # LAnkle -> LBigToe
    (19, 20), # LBigToe -> LSmallToe
    (14, 21), # LAnkle -> LHeel
    (11, 22), # RAnkle -> RBigToe
    (22, 23), # RBigToe -> RSmallToe
    (11, 24)  # RAnkle -> RHeel
]

CLASSES = ["as", "bs", "ce", "dr", "mfs", "ms", "sac", "sq", "tr", "tw"]


class SkeletonDataset(Dataset):
    """PyTorch Dataset yielding normalized skeleton sequence, confidence, and target."""
    def __init__(self, X: np.ndarray, Q: np.ndarray, y: np.ndarray, groups: np.ndarray, clips: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Q = torch.tensor(Q, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.groups = groups
        self.clips = clips

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str]:
        return self.X[idx], self.Q[idx], self.y[idx], str(self.groups[idx]), str(self.clips[idx])


def load_raw_mmasd(npz_path: str | Path | None = None) -> Dict[str, np.ndarray]:
    """Load cached MMASD OpenPose 64-frame array."""
    if npz_path is None:
        npz_path = ROOT / "data/processed/mmasd_openpose/mmasd_openpose_64frames.npz"
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Processed MMASD array not found at: {npz_path}")
    
    with np.load(npz_path, allow_pickle=True) as d:
        data = {
            "X": d["X"].astype(np.float32),      # (1374, 64, 25, 4)
            "Q": d["Q"].astype(np.float32),      # (1374, 64, 25)
            "y": d["y"].astype(np.int64),        # (1374,)
            "groups": d["groups"].astype(str),   # (1374,)
            "clips": d["clips"].astype(str),     # (1374,)
            "classes": d["classes"].astype(str)  # (10,)
        }
    return data


def create_subject_independent_folds(
    groups: np.ndarray, 
    n_folds: int = 5, 
    seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Greedy bin-packing to split unique subjects into n balanced folds.
    Guarantees no subject appears in more than one test fold.
    """
    rng = np.random.default_rng(seed)
    unique_subjects = np.unique(groups)
    tie_breaker = {s: rng.random() for s in unique_subjects}
    
    # Sort subjects descending by clip count with random tie-break
    sorted_subjects = sorted(unique_subjects, key=lambda s: (-int(np.sum(groups == s)), tie_breaker[s]))
    
    bins: List[List[str]] = [[] for _ in range(n_folds)]
    bin_clip_counts = np.zeros(n_folds, dtype=int)
    
    for subj in sorted_subjects:
        k = int(np.argmin(bin_clip_counts))
        bins[k].append(subj)
        bin_clip_counts[k] += int(np.sum(groups == subj))
        
    folds = []
    for test_subjs in bins:
        train_idx = np.flatnonzero(~np.isin(groups, test_subjs))
        test_idx = np.flatnonzero(np.isin(groups, test_subjs))
        folds.append((train_idx, test_idx))
    return folds


def partition_train_val_test(
    data: Dict[str, np.ndarray],
    outer_train_idx: np.ndarray,
    test_idx: np.ndarray,
    val_subj_fraction: float = 0.18,
    seed: int = 42
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Partitions outer training data into fit and validation partitions grouped by subject.
    Applies strict zero-leakage assertions and normalizes using fit partition statistics only.
    """
    groups = data["groups"]
    train_subjects = np.unique(groups[outer_train_idx])
    
    rng = np.random.default_rng(seed)
    shuffled_subjects = train_subjects.copy()
    rng.shuffle(shuffled_subjects)
    
    n_val = max(1, round(val_subj_fraction * len(shuffled_subjects)))
    val_subjects = set(shuffled_subjects[:n_val])
    fit_subjects = set(shuffled_subjects[n_val:])
    
    fit_idx = outer_train_idx[np.isin(groups[outer_train_idx], list(fit_subjects))]
    val_idx = outer_train_idx[np.isin(groups[outer_train_idx], list(val_subjects))]
    
    # STRICT ZERO-LEAKAGE ASSERTIONS
    assert len(set(groups[fit_idx]).intersection(set(groups[val_idx]))) == 0, "DATA LEAKAGE: Fit and Val share subjects!"
    assert len(set(groups[fit_idx]).intersection(set(groups[test_idx]))) == 0, "DATA LEAKAGE: Fit and Test share subjects!"
    assert len(set(groups[val_idx]).intersection(set(groups[test_idx]))) == 0, "DATA LEAKAGE: Val and Test share subjects!"
    assert len(fit_idx) + len(val_idx) == len(outer_train_idx), "Partition sample count mismatch!"
    
    # Compute normalization statistics STRICTLY on fit partition
    X_raw = data["X"]
    fit_X = X_raw[fit_idx]
    mu = fit_X.mean(axis=(0, 1, 2), keepdims=True)  # (1, 1, 1, 4)
    sd = fit_X.std(axis=(0, 1, 2), keepdims=True)   # (1, 1, 1, 4)
    sd[sd < 1e-5] = 1.0
    
    # Normalize with clipping to eliminate outliers
    X_norm = np.clip((X_raw - mu) / sd, -8.0, 8.0).astype(np.float32)
    
    fit_data = {
        "dataset": SkeletonDataset(X_norm[fit_idx], data["Q"][fit_idx], data["y"][fit_idx], groups[fit_idx], data["clips"][fit_idx]),
        "indices": fit_idx,
        "subjects": fit_subjects
    }
    val_data = {
        "dataset": SkeletonDataset(X_norm[val_idx], data["Q"][val_idx], data["y"][val_idx], groups[val_idx], data["clips"][val_idx]),
        "indices": val_idx,
        "subjects": val_subjects
    }
    test_data = {
        "dataset": SkeletonDataset(X_norm[test_idx], data["Q"][test_idx], data["y"][test_idx], groups[test_idx], data["clips"][test_idx]),
        "indices": test_idx,
        "subjects": set(groups[test_idx])
    }
    
    norm_stats = {"mu": mu, "sd": sd}
    return fit_data, val_data, test_data, norm_stats


def get_adjacency_matrix() -> torch.Tensor:
    """Returns normalized physical adjacency matrix for OpenPose BODY_25."""
    adj = torch.eye(25, dtype=torch.float32)
    for i, j in BODY_25_EDGES:
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    deg = adj.sum(dim=1).clamp_min(1.0).rsqrt()
    norm_adj = deg[:, None] * adj * deg[None, :]
    return norm_adj


if __name__ == "__main__":
    print("Testing MMASD data loader and subject partitioning...")
    data = load_raw_mmasd()
    print(f"Loaded X: {data['X'].shape}, Q: {data['Q'].shape}, y: {data['y'].shape}")
    folds = create_subject_independent_folds(data["groups"], n_folds=5, seed=42)
    for f, (tr, te) in enumerate(folds, 1):
        fit_d, val_d, test_d, _ = partition_train_val_test(data, tr, te, seed=42+f)
        print(f"Fold {f}: Fit={len(fit_d['indices'])} clips ({len(fit_d['subjects'])} subjs) | "
              f"Val={len(val_d['indices'])} clips ({len(val_d['subjects'])} subjs) | "
              f"Test={len(test_d['indices'])} clips ({len(test_d['subjects'])} subjs)")
    print("All zero-leakage assertions passed successfully.")
