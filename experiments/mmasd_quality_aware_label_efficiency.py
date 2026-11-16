"""MMASD quality-aware self-supervised spatiotemporal graph experiment.

Research question: can masked-node temporal pretraining and an explicit
observed-body-group quality signal improve label efficiency and robustness to
missing kinematic groups under held-out stored-group evaluation?

Important data limitation: the local cache contains 8 body-group x 8
kinematic summaries, not original joint confidence scores or missing-joint
masks. Reliability is therefore evaluated with controlled synthetic
body-group masking; this is not real detector-confidence modelling.

Run from project root with the bundled Python 3.12 runtime:
    python experiments/mmasd_quality_aware_label_efficiency.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
import copy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".codex-deps"))
sys.path.insert(0, str(ROOT / "models"))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from quality_aware_stgcn import QualityAwareSTGCN

INPUT = ROOT / "data" / "processed" / "mmasd_features" / "sessions_features.npz"
OUT_JSON = ROOT / "results" / "mmasd_quality_aware_label_efficiency_2026_11_12.json"
OUT_CSV = ROOT / "results" / "mmasd_quality_aware_label_efficiency_folds_2026_11_12.csv"
OUT_SUMMARY_CSV = ROOT / "results" / "mmasd_quality_aware_label_efficiency_summary_2026_11_12.csv"
SEED = 42
N_FOLDS = 5
LABEL_BUDGETS = (0.25, 0.50, 1.00)
VARIANTS = ("supervised_no_quality", "supervised_quality", "ssl_quality")
CORRUPTION_RATES = (0.0, 0.20, 0.40)
CORRUPTION_REPEATS = 3
PRETRAIN_EPOCHS = 8
MAX_FINETUNE_EPOCHS = 20
PATIENCE = 5
BATCH_SIZE = 64
MASK_AUGMENTATION_RATE = 0.15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_CLASSES = 11
FEATURE_DIM = 8
N_GROUPS = 8

BODY_EDGES = (
    (0, 1), (1, 0), (1, 2), (2, 1), (1, 3), (3, 1),
    (1, 4), (4, 1), (1, 5), (5, 1), (2, 6), (6, 2),
    (3, 7), (7, 3), (2, 3), (3, 2), (4, 5), (5, 4),
)


def make_adjacency() -> torch.Tensor:
    a = torch.eye(N_GROUPS, dtype=torch.float32)
    for i, j in BODY_EDGES:
        a[i, j] = 1.0
    deg = a.sum(dim=1).clamp_min(1.0)
    d = torch.diag(deg.rsqrt())
    return d @ a @ d


ADJACENCY = make_adjacency().to(DEVICE)


def metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    pred = probs.argmax(axis=1)
    acc = float(np.mean(pred == y_true))
    f1s = []
    for cls in range(N_CLASSES):
        tp = int(np.sum((y_true == cls) & (pred == cls)))
        fp = int(np.sum((y_true != cls) & (pred == cls)))
        fn = int(np.sum((y_true == cls) & (pred != cls)))
        den = 2 * tp + fp + fn
        f1s.append((2 * tp / den) if den else 0.0)
    conf = probs.max(axis=1)
    correct = (pred == y_true).astype(np.float32)
    ece = 0.0
    for lo in np.linspace(0.0, 1.0, 11)[:-1]:
        hi = lo + 0.1
        sel = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if np.any(sel):
            ece += float(sel.mean()) * abs(float(conf[sel].mean()) - float(correct[sel].mean()))
    return {"accuracy": acc, "macro_f1": float(np.mean(f1s)), "ece_10bin": float(ece)}


def greedy_group_folds(groups: np.ndarray, y: np.ndarray, n_splits: int,
                       seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique, counts = np.unique(groups, return_counts=True)
    rng = np.random.default_rng(seed)
    tie = {g: rng.random() for g in unique}
    order = sorted(unique.tolist(), key=lambda g: (-int(np.sum(groups == g)), tie[g]))
    bins: list[list[str]] = [[] for _ in range(n_splits)]
    sizes = np.zeros(n_splits, dtype=int)
    for g in order:
        target = int(np.argmin(sizes))
        bins[target].append(str(g))
        sizes[target] += int(np.sum(groups == g))
    folds = []
    for heldout in bins:
        te = np.flatnonzero(np.isin(groups, heldout))
        tr = np.flatnonzero(~np.isin(groups, heldout))
        folds.append((tr, te))
    return folds


def choose_validation_groups(train_idx: np.ndarray, groups: np.ndarray,
                             y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    local_groups = np.unique(groups[train_idx])
    local_y = y[train_idx]
    local_g = groups[train_idx]
    folds = greedy_group_folds(local_g, local_y, 5, seed)
    fit_local, val_local = folds[0]
    return train_idx[fit_local], train_idx[val_local]


def choose_labeled_subset(indices: np.ndarray, y: np.ndarray, fraction: float,
                          seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    picked = []
    for cls in np.unique(y[indices]):
        cls_idx = indices[y[indices] == cls]
        n = max(1, int(round(len(cls_idx) * fraction)))
        picked.extend(rng.choice(cls_idx, size=min(n, len(cls_idx)), replace=False).tolist())
    return np.asarray(sorted(picked), dtype=int)


def fit_normalizer(sequences: list[np.ndarray], indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frames = np.concatenate([sequences[int(i)].reshape(-1, FEATURE_DIM) for i in indices], axis=0)
    mean = np.nanmean(frames, axis=0).astype(np.float32)
    std = np.nanstd(frames, axis=0).astype(np.float32)
    std[~np.isfinite(std) | (std < 1e-5)] = 1.0
    mean[~np.isfinite(mean)] = 0.0
    return mean, std


def make_batch(sequences: list[np.ndarray], indices: np.ndarray,
               mean: np.ndarray, std: np.ndarray, mask_rate: float,
               rng: np.random.Generator) -> tuple[torch.Tensor, ...]:
    max_t = max(len(sequences[int(i)]) for i in indices)
    b = len(indices)
    target = np.zeros((b, max_t, N_GROUPS, FEATURE_DIM), dtype=np.float32)
    quality = np.zeros((b, max_t, N_GROUPS), dtype=np.float32)
    valid_time = np.zeros((b, max_t), dtype=bool)
    for row, idx in enumerate(indices):
        seq = np.asarray(sequences[int(idx)], dtype=np.float32)
        seq = np.nan_to_num((seq - mean[None, None, :]) / std[None, None, :],
                            nan=0.0, posinf=0.0, neginf=0.0)
        t = len(seq)
        target[row, :t] = seq
        quality[row, :t] = 1.0
        valid_time[row, :t] = True
    if mask_rate > 0:
        corruption = rng.random(quality.shape) < mask_rate
        corruption &= valid_time[:, :, None]
        # Keep at least one body group visible in each non-empty time step.
        all_masked = corruption.all(axis=2) & valid_time
        for bi, ti in np.argwhere(all_masked):
            corruption[bi, ti, int(rng.integers(0, N_GROUPS))] = False
        quality *= (~corruption).astype(np.float32)
    x = target * quality[..., None]
    return tuple(torch.as_tensor(v, device=DEVICE) for v in (x, quality, valid_time, target))


def iter_batches(indices: np.ndarray, batch_size: int, rng: np.random.Generator,
                 shuffle: bool = True):
    order = rng.permutation(indices) if shuffle else np.asarray(indices)
    for start in range(0, len(order), batch_size):
        yield order[start:start + batch_size]


def pretrain_ssl(model: QualityAwareSTGCN, sequences: list[np.ndarray],
                 train_idx: np.ndarray, mean: np.ndarray, std: np.ndarray,
                 seed: int) -> list[float]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    losses_by_epoch = []
    for _epoch in range(PRETRAIN_EPOCHS):
        model.train()
        total, n_batches = 0.0, 0
        for ids in iter_batches(train_idx, BATCH_SIZE, rng):
            x, quality, valid, target = make_batch(
                sequences, ids, mean, std, mask_rate=0.25, rng=rng)
            _, reconstruction, _ = model(x, quality, valid, ADJACENCY)
            hidden = (quality == 0) & valid[:, :, None]
            weights = hidden.unsqueeze(-1).float()
            if float(weights.sum()) == 0:
                weights = valid[:, :, None, None].float().expand_as(target)
            loss = (((reconstruction - target) ** 2) * weights).sum() / (weights.sum() * FEATURE_DIM).clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
            n_batches += 1
        losses_by_epoch.append(total / max(1, n_batches))
    return losses_by_epoch


def run_supervised(model: QualityAwareSTGCN, sequences: list[np.ndarray],
                   train_idx: np.ndarray, val_idx: np.ndarray, y: np.ndarray,
                   mean: np.ndarray, std: np.ndarray, seed: int) -> tuple[dict, int]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    labels = torch.as_tensor(y, dtype=torch.long, device=DEVICE)
    counts = np.bincount(y[train_idx], minlength=N_CLASSES).astype(np.float32)
    weights = np.zeros(N_CLASSES, dtype=np.float32)
    present = counts > 0
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    class_weights = torch.as_tensor(weights, device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_f1, best_state, best_epoch, patience = -1.0, None, 0, 0
    for epoch in range(1, MAX_FINETUNE_EPOCHS + 1):
        model.train()
        for ids in iter_batches(train_idx, BATCH_SIZE, rng):
            x, quality, valid, _target = make_batch(
                sequences, ids, mean, std, MASK_AUGMENTATION_RATE, rng)
            if not model.use_quality:
                # The no-quality ablation receives masked inputs but not the
                # observed-node indicator, isolating the value of that signal.
                quality_for_model = torch.ones_like(quality)
            else:
                quality_for_model = quality
            logits, _recon, _embedding = model(x, quality_for_model, valid, ADJACENCY)
            loss = F.cross_entropy(logits, labels[ids], weight=class_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_metrics, _ = evaluate(model, sequences, val_idx, y, mean, std,
                                  corruption=0.0, seed=seed + epoch)
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_macro_f1": float(best_f1)}, best_epoch


@torch.no_grad()
def evaluate(model: QualityAwareSTGCN, sequences: list[np.ndarray],
             indices: np.ndarray, y: np.ndarray, mean: np.ndarray, std: np.ndarray,
             corruption: float, seed: int, repeats: int = 1
             ) -> tuple[dict[str, float], np.ndarray]:
    model.eval()
    all_probs = []
    all_y = []
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + repeat * 13007)
        probs = []
        for ids in iter_batches(indices, BATCH_SIZE, rng, shuffle=False):
            x, quality, valid, _target = make_batch(
                sequences, ids, mean, std, mask_rate=corruption, rng=rng)
            qmodel = quality if model.use_quality else torch.ones_like(quality)
            logits, _rec, _emb = model(x, qmodel, valid, ADJACENCY)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        cur_probs = np.concatenate(probs)
        all_probs.append(cur_probs)
    mean_probs = np.mean(all_probs, axis=0)
    all_y = y[indices]
    return metrics(all_y, mean_probs), mean_probs


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.use_deterministic_algorithms(False)
    data = np.load(INPUT, allow_pickle=True)
    sequences = [np.asarray(s, dtype=np.float32) for s in data["features"]]
    y = data["labels"].astype(np.int64)
    groups = data["subjects"].astype(str)
    if len(np.unique(y)) != N_CLASSES:
        raise ValueError(f"Expected {N_CLASSES} classes; found {len(np.unique(y))}")

    outer = greedy_group_folds(groups, y, N_FOLDS, SEED)
    fold_rows: list[dict[str, Any]] = []
    pretrain_rows = []
    for fold, (outer_train, test_idx) in enumerate(outer, 1):
        fit_idx, val_idx = choose_validation_groups(outer_train, groups, y, SEED + fold)
        mean, std = fit_normalizer(sequences, fit_idx)
        torch.manual_seed(SEED + fold)
        pre_model = QualityAwareSTGCN(n_classes=N_CLASSES, use_quality=True).to(DEVICE)
        ssl_losses = pretrain_ssl(pre_model, sequences, fit_idx, mean, std, SEED + fold)
        pretrain_state = copy.deepcopy(pre_model.state_dict())
        pretrain_rows.append({"fold": fold, "ssl_loss_first": ssl_losses[0],
                              "ssl_loss_last": ssl_losses[-1]})
        print(f"Fold {fold}/{N_FOLDS}: pretraining loss {ssl_losses[0]:.4f} -> {ssl_losses[-1]:.4f}", flush=True)

        for budget in LABEL_BUDGETS:
            labeled = choose_labeled_subset(fit_idx, y, budget, SEED + fold + int(budget * 1000))
            for variant in VARIANTS:
                use_quality = variant != "supervised_no_quality"
                run_seed = SEED + fold * 101 + int(budget * 100) + VARIANTS.index(variant)
                torch.manual_seed(run_seed)
                model = QualityAwareSTGCN(n_classes=N_CLASSES,
                                          use_quality=use_quality).to(DEVICE)
                if variant == "ssl_quality":
                    model.load_state_dict(pretrain_state)
                val_info, best_epoch = run_supervised(
                    model, sequences, labeled, val_idx, y, mean, std,
                    run_seed)
                for rate in CORRUPTION_RATES:
                    test_metrics, _ = evaluate(
                        model, sequences, test_idx, y, mean, std,
                        corruption=rate, seed=SEED + fold * 1009 + int(rate * 100),
                        repeats=CORRUPTION_REPEATS if rate else 1)
                    row = {
                        "fold": fold,
                        "variant": variant,
                        "label_fraction": budget,
                        "labeled_sessions": int(len(labeled)),
                        "test_sessions": int(len(test_idx)),
                        "corruption_rate": rate,
                        "best_epoch": best_epoch,
                        "val_macro_f1": val_info["best_val_macro_f1"],
                        **test_metrics,
                    }
                    fold_rows.append(row)
                    print(f"  {variant} labels={budget:.0%} mask={rate:.0%}: "
                          f"acc={test_metrics['accuracy']:.3f} "
                          f"macroF1={test_metrics['macro_f1']:.3f} "
                          f"ECE={test_metrics['ece_10bin']:.3f}", flush=True)

    aggregate = {}
    keys = sorted({(r["variant"], r["label_fraction"], r["corruption_rate"])
                   for r in fold_rows})
    for variant, budget, rate in keys:
        rows = [r for r in fold_rows if r["variant"] == variant and
                r["label_fraction"] == budget and r["corruption_rate"] == rate]
        aggregate[f"{variant}|{budget:.2f}|{rate:.2f}"] = {
            f"{metric}_mean": float(np.mean([r[metric] for r in rows]))
            for metric in ("accuracy", "macro_f1", "ece_10bin")
        } | {
            f"{metric}_sd": float(np.std([r[metric] for r in rows], ddof=1))
            for metric in ("accuracy", "macro_f1", "ece_10bin")
        }

    summary = {
        "study": "MMASD quality-aware self-supervised spatiotemporal graph learning",
        "status": "initial experiment completed; results are preliminary",
        "data": {"recordings": int(len(y)), "groups": int(len(np.unique(groups))),
                 "classes": int(len(np.unique(y))), "input_shape": "variable time x 8 body groups x 8 kinematic features"},
        "method": {"spatial_graph": "anatomical graph over eight body groups plus self loops",
                   "temporal_encoder": "bidirectional GRU over spatially pooled graph features",
                   "self_supervised_objective": "reconstruct artificially masked body-group feature vectors",
                   "quality_signal": "observed body-group mask; controlled synthetic missingness only",
                   "label_budgets": list(LABEL_BUDGETS),
                   "test_corruption_rates": list(CORRUPTION_RATES),
                   "corruption_repeats": CORRUPTION_REPEATS,
                   "early_stopping_metric": "inner validation macro-F1",
                   "device": str(DEVICE)},
        "evaluation": "five outer folds grouped by stored subject identifiers; validation groups and all fitted normalization are separated from outer test groups",
        "group_identity_caveat": "Stored filename-derived group identifiers have not been independently verified as participant identities.",
        "data_limitations": [
            "The cached features do not include per-joint detector confidence or true missing-joint masks.",
            "Synthetic body-group masking tests controlled robustness; it is not a substitute for real pose-quality metadata.",
            "This result is a ten/eleven-class activity-recognition task (the local cache has eleven classes), not autism diagnosis.",
        ],
        "pretraining_fold_losses": pretrain_rows,
        "fold_results": fold_rows,
        "aggregate_mean_sd_across_folds": aggregate,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fold_rows[0]))
        writer.writeheader()
        writer.writerows(fold_rows)
    summary_rows = []
    for key, values in aggregate.items():
        variant, budget, rate = key.split("|")
        summary_rows.append({"variant": variant, "label_fraction": float(budget),
                             "corruption_rate": float(rate), **values})
    with OUT_SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved {OUT_JSON}")
    print(f"Saved {OUT_CSV}")
    print(f"Saved {OUT_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
