"""Closer, pose/kinematic-only adaptation of the 2026 MMASD base paper.

This trains a denoising temporal Transformer autoencoder on local (T, 8, 8)
body-group kinematic sequences. It follows the paper's 8-frame/4-stride windows,
128-d latent state, 2-layer/4-head Transformer, SmoothL1 reconstruction loss,
Adam lr=1e-3, ten epochs, and six-cluster discovery. It is not an exact
reproduction because the local files do not include the paper's 33-joint
BlazePose + optical-flow inputs or its session-aligned joint-attention values.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".codex-deps"))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mmasd_dynamics_replication import (
    N_CLUSTERS,
    SEED,
    adjusted_rand,
    best_kmeans,
    calinski_harabasz,
    davies_bouldin,
    normalized_mutual_information,
    predict,
    silhouette,
)


INPUT = ROOT / "data" / "processed" / "mmasd_features" / "sessions_features.npz"
OUT_JSON = ROOT / "results" / "mmasd_temporal_autoencoder_2026_11_12.json"
OUT_CSV = ROOT / "results" / "mmasd_temporal_autoencoder_folds_2026_11_12.csv"
WINDOW = 8
STRIDE = 4
LATENT_DIM = 128
EPOCHS = 10
BATCH_SIZE = 32
N_TRAIN_SEEDS = 3
N_PERMUTATIONS = 199


class TemporalDenoisingAE(nn.Module):
    def __init__(self, input_dim: int = 64) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, LATENT_DIM)
        self.position = nn.Parameter(torch.zeros(1, WINDOW, LATENT_DIM))
        nn.init.normal_(self.position, mean=0.0, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=LATENT_DIM,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(LATENT_DIM)
        self.output_proj = nn.Linear(LATENT_DIM, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x) + self.position
        return self.norm(self.encoder(h))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(x)
        return self.output_proj(h), h.mean(dim=1)


def make_windows() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    z = np.load(INPUT, allow_pickle=True)
    seqs = z["features"]
    y = z["labels"].astype(np.int64)
    groups = z["subjects"].astype(str)
    sessions = z["sessions"].astype(str)
    windows, labels, window_groups, window_sessions, order = [], [], [], [], []
    used = 0
    for i, seq in enumerate(seqs):
        seq = np.asarray(seq, dtype=np.float32).reshape(len(seq), -1)
        starts = list(range(0, len(seq) - WINDOW + 1, STRIDE))
        if starts:
            used += 1
        for j, start in enumerate(starts):
            windows.append(seq[start : start + WINDOW])
            labels.append(y[i])
            window_groups.append(groups[i])
            window_sessions.append(sessions[i])
            order.append(j)
    info = {
        "n_recordings": int(len(seqs)),
        "n_subjects": int(len(np.unique(groups))),
        "n_activity_labels": int(len(np.unique(y))),
        "n_windows": int(len(windows)),
        "sessions_with_windows": int(used),
        "window_length": WINDOW,
        "stride": STRIDE,
        "features_per_frame": int(windows[0].shape[1]),
    }
    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(labels),
        np.asarray(window_groups),
        np.asarray(window_sessions),
        np.asarray(order),
        info,
    )


def embed(model: TemporalDenoisingAE, x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    scaled = (x - mean) / std
    result = []
    model.eval()
    with torch.no_grad():
        for batch in torch.from_numpy(scaled.astype(np.float32)).split(256):
            result.append(model.encode(batch).mean(dim=1).numpy())
    return np.concatenate(result, axis=0)


def fit_model(train_x: np.ndarray, seed: int) -> tuple[TemporalDenoisingAE, np.ndarray, np.ndarray, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    mean = train_x.mean(axis=(0, 1), keepdims=True)
    std = train_x.std(axis=(0, 1), keepdims=True)
    std[std < 1e-6] = 1.0
    train_scaled = (train_x - mean) / std
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_scaled.astype(np.float32))),
        batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(seed), drop_last=False,
    )
    model = TemporalDenoisingAE(input_dim=train_x.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.SmoothL1Loss()
    last_loss = float("nan")
    model.train()
    for _ in range(EPOCHS):
        losses = []
        for (target,) in loader:
            corrupted = target + torch.randn_like(target) * 0.05
            frame_mask = torch.rand((len(target), WINDOW, 1)) < 0.05
            corrupted = corrupted.masked_fill(frame_mask, 0.0)
            prediction, _ = model(corrupted)
            loss = loss_fn(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        last_loss = float(np.mean(losses))
    return model, mean, std, last_loss


def reconstruction_loss(model: TemporalDenoisingAE, x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    scaled = (x - mean) / std
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in torch.from_numpy(scaled.astype(np.float32)).split(256):
            prediction, _ = model(batch)
            losses.append(float(nn.functional.smooth_l1_loss(prediction, batch)))
    return float(np.mean(losses))


def main() -> None:
    x, y, groups, sessions, window_order, data_info = make_windows()
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(SEED)
    fold_groups = np.array_split(rng.permutation(unique_groups), 5)
    fold_rows, test_labels_by_fold = [], []

    for fold, heldout in enumerate(fold_groups, 1):
        test_mask = np.isin(groups, heldout)
        tr, te = np.flatnonzero(~test_mask), np.flatnonzero(test_mask)
        models_test_labels = []
        run_rows = []
        first_embeddings = None
        for run in range(N_TRAIN_SEEDS):
            seed = SEED + fold * 10007 + run * 5003
            model, mean, std, train_loss = fit_model(x[tr], seed)
            ztr = embed(model, x[tr], mean, std)
            zte = embed(model, x[te], mean, std)
            centers, _ = best_kmeans(ztr, N_CLUSTERS, seed + 211)
            labels = predict(zte, centers)
            models_test_labels.append(labels)
            run_rows.append({
                "run": run + 1,
                "train_reconstruction_loss_last_epoch": train_loss,
                "test_reconstruction_loss": reconstruction_loss(model, x[te], mean, std),
            })
            if run == 0:
                first_embeddings = zte

        stability = [adjusted_rand(models_test_labels[a], models_test_labels[b])
                     for a, b in combinations(range(N_TRAIN_SEEDS), 2)]
        labs = models_test_labels[0]
        fold_rows.append({
            "fold": fold,
            "train_subjects": int(len(unique_groups) - len(heldout)),
            "test_subjects": int(len(heldout)),
            "test_windows": int(len(te)),
            "test_clusters_used": int(len(np.unique(labs))),
            "test_reconstruction_loss_mean": float(np.mean([r["test_reconstruction_loss"] for r in run_rows])),
            "silhouette_sampled": silhouette(first_embeddings, labs, SEED + fold),
            "davies_bouldin": davies_bouldin(first_embeddings, labs),
            "calinski_harabasz": calinski_harabasz(first_embeddings, labs),
            "activity_ARI": adjusted_rand(labs, y[te]),
            "activity_NMI": normalized_mutual_information(labs, y[te]),
            "seed_stability_ARI_mean": float(np.mean(stability)) if stability else 1.0,
        })
        test_labels_by_fold.append((te, labs))
        print(f"fold {fold}/5 complete: held-out subjects={len(heldout)}, windows={len(te)}, test_loss={fold_rows[-1]['test_reconstruction_loss_mean']:.5f}", flush=True)

    oof = np.full(len(x), -1, dtype=np.int64)
    for inds, labs in test_labels_by_fold:
        oof[inds] = labs
    transition_seqs, session_kl = [], []
    total_trans = same_trans = 0
    for sid in np.unique(sessions):
        inds = np.flatnonzero(sessions == sid)
        inds = inds[np.argsort(window_order[inds])]
        labs = oof[inds]
        if len(labs) >= 2:
            transition_seqs.append(labs.copy())
            same_trans += int(np.sum(labs[:-1] == labs[1:]))
            total_trans += len(labs) - 1
        if len(labs) >= 4:
            mid = len(labs) // 2
            a = np.bincount(labs[:mid], minlength=N_CLUSTERS).astype(float)
            b = np.bincount(labs[mid:], minlength=N_CLUSTERS).astype(float)
            pa = (a + 0.5) / (a.sum() + 0.5 * N_CLUSTERS)
            pb = (b + 0.5) / (b.sum() + 0.5 * N_CLUSTERS)
            session_kl.append(float(np.sum(pa * np.log(pa / pb))))

    observed_stay = same_trans / total_trans if total_trans else float("nan")
    rng = np.random.default_rng(SEED + 991)
    null = np.empty(N_PERMUTATIONS, dtype=float)
    for i in range(N_PERMUTATIONS):
        same = total = 0
        for seq in transition_seqs:
            perm = rng.permutation(seq)
            same += int(np.sum(perm[:-1] == perm[1:]))
            total += len(perm) - 1
        null[i] = same / total if total else np.nan
    pval = float((1 + np.sum(null >= observed_stay)) / (N_PERMUTATIONS + 1))

    def aggregate(key: str) -> dict[str, float]:
        vals = np.asarray([r[key] for r in fold_rows], dtype=float)
        return {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1))}

    summary = {
        "study": "MMASD base-paper temporal autoencoder adaptation",
        "date": "2026-11-12",
        "base_paper": "Akinwumi, Qian, and Ojo (2026), doi:10.3389/fnins.2026.1737098",
        "status": "closer adaptation, not exact reproduction",
        "reproduction_limits": [
            "Local inputs are 8-body-group x 8-kinematic-channel sequences, not 33-joint BlazePose pairwise distances plus optical flow.",
            "The local workspace contains no source video/optical-flow inputs for the paper's selected 15-session subset.",
            "Session-aligned joint-attention duration values used by the paper are not available locally, so the reported KL-attention correlation cannot be recomputed.",
        ],
        "model": {
            "input_channels_per_frame": int(x.shape[-1]),
            "latent_dimension": LATENT_DIM,
            "transformer_encoder_layers": 2,
            "attention_heads": 4,
            "feedforward_dimension": 256,
            "dropout": 0.1,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "optimizer": "Adam",
            "learning_rate": 1e-3,
            "loss": "SmoothL1",
            "noise_std": 0.05,
            "temporal_frame_dropout": 0.05,
            "clusters": N_CLUSTERS,
            "seed_runs_per_outer_fold": N_TRAIN_SEEDS,
        },
        "data": data_info,
        "evaluation": "5-fold held-out stored-subject groups; preprocessing/model/clusters fitted on the training subjects; six-cluster assignments evaluated on held-out windows",
        "metrics_mean_sd_across_folds": {
            key: aggregate(key) for key in [
                "test_reconstruction_loss_mean", "silhouette_sampled", "davies_bouldin",
                "calinski_harabasz", "activity_ARI", "activity_NMI", "seed_stability_ARI_mean",
            ]
        },
        "heldout_dynamics": {
            "n_transitions": int(total_trans),
            "observed_self_transition_rate": float(observed_stay),
            "null_self_transition_rate_mean": float(np.mean(null)),
            "null_self_transition_rate_sd": float(np.std(null, ddof=1)),
            "one_sided_shuffle_permutation_p_value": pval,
            "sessions_with_early_late_kl": int(len(session_kl)),
            "early_late_kl_mean": float(np.mean(session_kl)) if session_kl else None,
            "early_late_kl_median": float(np.median(session_kl)) if session_kl else None,
            "joint_attention_correlation": None,
        },
        "fold_results": fold_rows,
        "interpretation": "The results evaluate a pose/kinematic-only temporal Transformer adaptation. They do not reproduce the paper's multimodal input, exact selected sessions, saliency analysis, or joint-attention association, and they do not constitute clinical outcome evaluation.",
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fold_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fold_rows)
    print(json.dumps({
        "json": str(OUT_JSON), "csv": str(OUT_CSV), "data": data_info,
        "metrics": summary["metrics_mean_sd_across_folds"],
        "dynamics": summary["heldout_dynamics"], "folds": fold_rows,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
