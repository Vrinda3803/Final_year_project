"""Subject-grouped, pose/kinematic-only reproduction attempt of MMASD dynamics.

This is an adaptation, not an exact reproduction of Akinwumi et al. (2026): the
local processed sequences contain 8 body-group x 8 kinematic channels, while the
paper uses 2D pose plus optical flow and a denoising temporal Transformer AE.
We use 8-step overlapping windows, train-fold standardization/PCA, and k-means
with six clusters. Evaluation is grouped by subject and reports descriptive
unsupervised metrics; it does not estimate clinical or therapeutic outcomes.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "mmasd_features" / "sessions_features.npz"
OUT_JSON = ROOT / "results" / "mmasd_dynamics_replication_2026_11_12.json"
OUT_CSV = ROOT / "results" / "mmasd_dynamics_replication_folds_2026_11_12.csv"
SEED = 42
WINDOW = 8
STRIDE = 4
N_COMPONENTS = 128
N_CLUSTERS = 6
N_RESTARTS = 5
N_INIT = 3
MAX_ITER = 60
SILHOUETTE_SAMPLE = 300
N_PERMUTATIONS = 199


def comb2(n: np.ndarray | float) -> np.ndarray | float:
    return n * (n - 1) / 2


def adjusted_rand(a: np.ndarray, b: np.ndarray) -> float:
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table = np.zeros((ai.max() + 1, bi.max() + 1), dtype=np.int64)
    np.add.at(table, (ai, bi), 1)
    sum_cells = comb2(table).sum()
    sum_rows = comb2(table.sum(axis=1)).sum()
    sum_cols = comb2(table.sum(axis=0)).sum()
    total = comb2(len(a))
    expected = sum_rows * sum_cols / total if total else 0.0
    maximum = (sum_rows + sum_cols) / 2
    return float((sum_cells - expected) / (maximum - expected)) if maximum != expected else 1.0


def normalized_mutual_information(a: np.ndarray, b: np.ndarray) -> float:
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table = np.zeros((ai.max() + 1, bi.max() + 1), dtype=np.float64)
    np.add.at(table, (ai, bi), 1)
    pxy = table / table.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    nz = pxy > 0
    ix, iy = np.nonzero(nz)
    mi = np.sum(pxy[nz] * np.log(pxy[nz] / (px[ix] * py[iy])))
    hx = -np.sum(px[px > 0] * np.log(px[px > 0]))
    hy = -np.sum(py[py > 0] * np.log(py[py > 0]))
    den = (hx + hy) / 2
    return float(mi / den) if den else 1.0


def kmeans_fit(x: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(x)
    centers = np.empty((k, x.shape[1]), dtype=np.float64)
    centers[0] = x[rng.integers(n)]
    closest = np.sum((x - centers[0]) ** 2, axis=1)
    for j in range(1, k):
        weights = closest / max(closest.sum(), 1e-12)
        centers[j] = x[rng.choice(n, p=weights)]
        closest = np.minimum(closest, np.sum((x - centers[j]) ** 2, axis=1))
    for _ in range(MAX_ITER):
        d2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(d2, axis=1)
        new = centers.copy()
        for j in range(k):
            members = x[labels == j]
            if len(members):
                new[j] = members.mean(axis=0)
            else:
                new[j] = x[rng.integers(n)]
        if np.allclose(new, centers, rtol=1e-5, atol=1e-6):
            centers = new
            break
        centers = new
    d2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return centers, np.argmin(d2, axis=1)


def predict(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.argmin(((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)


def best_kmeans(x: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Select the lowest-inertia initialization using training data only."""
    best_centers, best_labels, best_inertia = None, None, float("inf")
    for init in range(N_INIT):
        centers, labels = kmeans_fit(x, k, seed + init * 1009)
        inertia = float(((x - centers[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_centers, best_labels, best_inertia = centers, labels, inertia
    return best_centers, best_labels


def silhouette(x: np.ndarray, labels: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    unique = np.unique(labels)
    if len(unique) < 2:
        return float("nan")
    indices = np.arange(len(x))
    if len(indices) > SILHOUETTE_SAMPLE:
        indices = np.sort(rng.choice(indices, SILHOUETTE_SAMPLE, replace=False))
    clusters = [np.flatnonzero(labels == lab) for lab in unique]
    vals = []
    for sample_idx in indices:
        row, lab = x[sample_idx], labels[sample_idx]
        d = np.sqrt(np.maximum(((x - row) ** 2).sum(axis=1), 0.0))
        own = clusters[np.where(unique == lab)[0][0]]
        own = own[own != sample_idx]
        a = float(d[own].mean()) if len(own) else 0.0
        b = min(float(d[c].mean()) for c, other in zip(clusters, unique) if other != lab and len(c))
        vals.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(vals))


def davies_bouldin(x: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if len(unique) < 2:
        return float("nan")
    centers, scatters = [], []
    for lab in unique:
        points = x[labels == lab]
        center = points.mean(axis=0)
        centers.append(center)
        scatters.append(float(np.sqrt(((points - center) ** 2).sum(axis=1)).mean()))
    centers = np.asarray(centers)
    scatters = np.asarray(scatters)
    distances = np.sqrt(np.maximum(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), 1e-12))
    ratios = (scatters[:, None] + scatters[None, :]) / distances
    np.fill_diagonal(ratios, -np.inf)
    return float(np.mean(np.max(ratios, axis=1)))


def calinski_harabasz(x: np.ndarray, labels: np.ndarray) -> float:
    n, k = len(x), len(np.unique(labels))
    if k < 2 or n <= k:
        return float("nan")
    grand = x.mean(axis=0)
    between = 0.0
    within = 0.0
    for lab in np.unique(labels):
        pts = x[labels == lab]
        center = pts.mean(axis=0)
        between += len(pts) * float(((center - grand) ** 2).sum())
        within += float(((pts - center) ** 2).sum())
    return float((between / (k - 1)) / (within / (n - k))) if within else float("inf")


def main() -> None:
    z = np.load(INPUT, allow_pickle=True)
    sequences = z["features"]
    activity = z["labels"].astype(int)
    subjects = z["subjects"].astype(str)
    sessions = z["sessions"].astype(str)

    windows, win_activity, win_subject, win_session, win_order = [], [], [], [], []
    usable_sessions = 0
    for i, seq in enumerate(sequences):
        seq = np.asarray(seq, dtype=np.float32).reshape(len(seq), -1)
        starts = list(range(0, len(seq) - WINDOW + 1, STRIDE))
        if starts:
            usable_sessions += 1
        for order, start in enumerate(starts):
            windows.append(seq[start : start + WINDOW].reshape(-1))
            win_activity.append(activity[i])
            win_subject.append(subjects[i])
            win_session.append(sessions[i])
            win_order.append(order)
    xraw = np.asarray(windows, dtype=np.float32)
    y = np.asarray(win_activity, dtype=np.int64)
    groups = np.asarray(win_subject)
    session_ids = np.asarray(win_session)
    order = np.asarray(win_order)

    unique_groups = np.unique(groups)
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(unique_groups)
    group_folds = np.array_split(shuffled, 5)
    fold_rows = []
    fold_predictions = []
    fold_test_indices = []
    pairwise_seed_stabilities = []

    for fold_id, heldout in enumerate(group_folds, 1):
        test_mask = np.isin(groups, heldout)
        train, test = np.flatnonzero(~test_mask), np.flatnonzero(test_mask)
        mu = xraw[train].mean(axis=0)
        sd = xraw[train].std(axis=0)
        sd[sd < 1e-6] = 1.0
        xt = (xraw[train] - mu) / sd
        xv = (xraw[test] - mu) / sd
        pca_mu = xt.mean(axis=0)
        centered = xt - pca_mu
        cov = centered.T @ centered / max(1, len(centered) - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        components = eigvecs[:, np.argsort(eigvals)[::-1][:N_COMPONENTS]]
        ztrain = centered @ components
        ztest = (xv - pca_mu) @ components

        run_labels = []
        for rep in range(N_RESTARTS):
            centers_r, _ = best_kmeans(ztrain, N_CLUSTERS, SEED + fold_id * 10007 + rep * 5003)
            run_labels.append(predict(ztest, centers_r))
        labels_test = run_labels[0]
        stability = [adjusted_rand(run_labels[a], run_labels[b])
                     for a in range(len(run_labels)) for b in range(a + 1, len(run_labels))]
        pairwise_seed_stabilities.extend(stability)

        fold_y = y[test]
        fold_rows.append({
            "fold": fold_id,
            "train_subjects": int(len(np.unique(groups[train]))),
            "test_subjects": int(len(heldout)),
            "test_windows": int(len(test)),
            "n_test_clusters": int(len(np.unique(labels_test))),
            "silhouette_sampled": silhouette(ztest, labels_test, SEED + fold_id),
            "davies_bouldin": davies_bouldin(ztest, labels_test),
            "calinski_harabasz": calinski_harabasz(ztest, labels_test),
            "activity_ARI": adjusted_rand(labels_test, fold_y),
            "activity_NMI": normalized_mutual_information(labels_test, fold_y),
            "seed_stability_ARI_mean": float(np.mean(stability)) if stability else 1.0,
        })
        fold_predictions.append((test, labels_test))
        fold_test_indices.append(test)

    # Aggregate held-out labels, retaining original window order within each session.
    oof = np.full(len(xraw), -1, dtype=np.int64)
    for inds, labs in fold_predictions:
        oof[inds] = labs
    transitions = []
    session_rows = []
    for sid in np.unique(session_ids):
        inds = np.flatnonzero(session_ids == sid)
        inds = inds[np.argsort(order[inds])]
        labs = oof[inds]
        if len(labs) >= 2:
            transitions.extend(zip(labs[:-1], labs[1:]))
        if len(labs) >= 4:
            mid = len(labs) // 2
            counts_a = np.bincount(labs[:mid], minlength=N_CLUSTERS).astype(float)
            counts_b = np.bincount(labs[mid:], minlength=N_CLUSTERS).astype(float)
            pa = (counts_a + 0.5) / (counts_a.sum() + 0.5 * N_CLUSTERS)
            pb = (counts_b + 0.5) / (counts_b.sum() + 0.5 * N_CLUSTERS)
            kl = float(np.sum(pa * np.log(pa / pb)))
            session_rows.append({"session": str(sid), "n_windows": int(len(labs)), "early_late_kl": kl})
    transitions = np.asarray(transitions, dtype=np.int64)
    observed_stay = float(np.mean(transitions[:, 0] == transitions[:, 1])) if len(transitions) else float("nan")

    # Null comparison: independently shuffle each session's primitive sequence,
    # preserving its marginal cluster counts and session length.
    rng = np.random.default_rng(SEED + 991)
    null_stay = np.empty(N_PERMUTATIONS, dtype=np.float64)
    session_sequences = []
    for sid in np.unique(session_ids):
        inds = np.flatnonzero(session_ids == sid)
        inds = inds[np.argsort(order[inds])]
        if len(inds) >= 2:
            session_sequences.append(oof[inds].copy())
    for p in range(N_PERMUTATIONS):
        same = total = 0
        for labs in session_sequences:
            perm = rng.permutation(labs)
            same += int(np.sum(perm[:-1] == perm[1:]))
            total += len(perm) - 1
        null_stay[p] = same / total if total else np.nan
    p_value = float((1 + np.sum(null_stay >= observed_stay)) / (N_PERMUTATIONS + 1))

    def mean_sd(key: str) -> dict[str, float]:
        vals = np.asarray([row[key] for row in fold_rows], dtype=float)
        return {"mean": float(np.nanmean(vals)), "sd": float(np.nanstd(vals, ddof=1))}

    summary = {
        "study": "MMASD pose/kinematic-only learning-dynamics reproduction attempt",
        "date": "2026-11-12",
        "base_paper": "Akinwumi, Qian, and Ojo (2026), doi:10.3389/fnins.2026.1737098",
        "reproduction_status": "adapted, not exact",
        "reason_not_exact": [
            "Paper uses 2D BlazePose pose and optical flow; the local processed input is an 8-body-group by 8-kinematic-channel sequence.",
            "No local optical-flow or source-video inputs were found.",
            "The paper uses a denoising temporal Transformer autoencoder; this run uses PCA plus k-means because PyTorch and scikit-learn are unavailable in the active runtime.",
            "The paper's session subset and joint-attention measurements are not present in the local result artifacts.",
        ],
        "protocol": {
            "source_file": str(INPUT.relative_to(ROOT)),
            "n_recordings": int(len(sequences)),
            "n_subjects": int(len(unique_groups)),
            "n_activity_labels": int(len(np.unique(activity))),
            "n_windows": int(len(xraw)),
            "n_sessions_with_at_least_one_window": int(usable_sessions),
            "window_length": WINDOW,
            "stride": STRIDE,
            "raw_window_dimension": int(xraw.shape[1]),
            "pca_dimensions": N_COMPONENTS,
            "clusters": N_CLUSTERS,
            "outer_evaluation": "5-fold grouped holdout by stored subject identifier",
            "seed": SEED,
            "kmeans_restarts_per_fold": N_RESTARTS,
            "best_inertia_initializations_per_run": N_INIT,
            "transition_null_permutations": N_PERMUTATIONS,
        },
        "metrics_mean_sd_across_subject_folds": {
            key: mean_sd(key) for key in ["silhouette_sampled", "davies_bouldin", "calinski_harabasz", "activity_ARI", "activity_NMI", "seed_stability_ARI_mean"]
        },
        "heldout_transition_analysis": {
            "n_transitions": int(len(transitions)),
            "observed_self_transition_rate": observed_stay,
            "null_self_transition_rate_mean": float(np.mean(null_stay)),
            "null_self_transition_rate_sd": float(np.std(null_stay, ddof=1)),
            "one_sided_permutation_p_value": p_value,
            "n_sessions_with_early_late_kl": int(len(session_rows)),
            "early_late_kl_mean": float(np.mean([r["early_late_kl"] for r in session_rows])) if session_rows else None,
            "early_late_kl_median": float(np.median([r["early_late_kl"] for r in session_rows])) if session_rows else None,
            "joint_attention_correlation": None,
            "joint_attention_note": "Local session-aligned joint-attention duration values were unavailable; paper correlation was not re-estimated.",
        },
        "fold_results": fold_rows,
        "interpretation_limit": "The results evaluate an adapted kinematic-only clustering baseline, not the paper's multimodal Transformer or a clinical learning outcome. The 11 activity labels are used only for external cluster association metrics.",
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fold_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fold_rows)
    print(json.dumps({
        "json": str(OUT_JSON), "csv": str(OUT_CSV),
        "protocol": summary["protocol"],
        "summary": summary["metrics_mean_sd_across_subject_folds"],
        "transitions": summary["heldout_transition_analysis"],
        "fold_results": fold_rows,
    }, indent=2))


if __name__ == "__main__":
    main()
