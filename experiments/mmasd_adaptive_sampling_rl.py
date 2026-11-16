"""PPO-based, fixed-budget adaptive window sampling for MMASD.

This is an experimental implementation of the proposed contribution, not a
clinical decision system. It learns a policy on training-subject sessions to
choose temporal gaps between low-cost kinematic descriptors. The selected
windows are summarized and classified. All fold normalization, policy training,
and classifier fitting are restricted to training subjects.

Run from the project root with NumPy, scikit-learn, and PyTorch installed:
    python experiments/mmasd_adaptive_sampling_rl.py

The experiment writes dated fold-level and summary outputs under results/.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".codex-deps"))
import numpy as np
import torch
from torch import nn


INPUT = ROOT / "data" / "processed" / "mmasd_features" / "sessions_features.npz"
OUT_JSON = ROOT / "results" / "mmasd_adaptive_sampling_rl_2026_11_12.json"
OUT_CSV = ROOT / "results" / "mmasd_adaptive_sampling_rl_folds_2026_11_12.csv"
SEED = 42
N_FOLDS = 5
WINDOW = 8
SAMPLE_BUDGET = 8
STRIDES = (1, 2, 4, 8)
PPO_EPISODES = 48
PPO_BATCH_SESSIONS = 16
PPO_UPDATES = 4
PPO_LR = 3e-4
GAMMA = 0.98
CLIP_EPS = 0.2
ENTROPY_WEIGHT = 0.015
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    accuracy = float(np.mean(y_true == y_pred))
    class_f1 = []
    for c in np.unique(np.concatenate([y_true, y_pred])):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        denom = 2 * tp + fp + fn
        class_f1.append(2 * tp / denom if denom else 0.0)
    return accuracy, float(np.mean(class_f1))


def _window_descriptors(seq: np.ndarray) -> np.ndarray:
    """Cheap per-window summaries: position, spread, and motion magnitude."""
    seq = np.asarray(seq, dtype=np.float32).reshape(len(seq), -1)
    if len(seq) < WINDOW:
        seq = np.pad(seq, ((0, WINDOW - len(seq)), (0, 0)), mode="edge")
    starts = np.arange(max(1, len(seq) - WINDOW + 1), dtype=np.int64)
    windows = np.stack([seq[s:s + WINDOW] for s in starts])
    mean = windows.mean(axis=1)
    spread = windows.std(axis=1)
    velocity = np.abs(np.diff(windows, axis=1)).mean(axis=1)
    return np.nan_to_num(np.concatenate([mean, spread, velocity], axis=1),
                         nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _group_folds(groups: np.ndarray, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic five-fold split that keeps every stored group intact."""
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    unique = rng.permutation(unique)
    chunks = np.array_split(unique, N_FOLDS)
    folds = []
    for heldout in chunks:
        test = np.flatnonzero(np.isin(groups, heldout))
        train = np.flatnonzero(~np.isin(groups, heldout))
        folds.append((train, test))
    return folds


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, n_actions: int) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(state_dim, 128), nn.Tanh(),
                                  nn.Linear(128, 128), nn.Tanh())
        self.actor = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(state)
        return self.actor(h), self.critic(h).squeeze(-1)


def _state(desc: np.ndarray, chosen: list[np.ndarray], position: int,
           total: int, step: int) -> np.ndarray:
    """Current cheap descriptor plus coverage and selected-set novelty."""
    if chosen:
        center = np.mean(chosen, axis=0)
        novelty = float(np.linalg.norm(desc - center) / np.sqrt(desc.size))
    else:
        novelty = 0.0
    return np.concatenate([desc, np.asarray([
        position / max(1, total - 1),
        step / SAMPLE_BUDGET,
        novelty,
    ], dtype=np.float32)]).astype(np.float32)


def _mask_unreachable_actions(logits: torch.Tensor, position: int, total: int,
                              step: int, episode_budget: int) -> torch.Tensor:
    """Keep enough candidate windows available to finish without duplicates."""
    future_picks = episode_budget - step - 1
    latest_next_position = total - 1 - future_picks + 1
    max_stride = latest_next_position - position
    invalid = torch.as_tensor([stride > max_stride for stride in STRIDES],
                               dtype=torch.bool, device=logits.device)
    return logits.masked_fill(invalid.unsqueeze(0), -1e9)


def _rollout(model: ActorCritic, sessions: list[np.ndarray], train: bool,
             rng: np.random.Generator) -> dict[str, Any]:
    states: list[np.ndarray] = []
    actions: list[int] = []
    log_probs: list[float] = []
    rewards: list[float] = []
    values: list[float] = []
    chosen_sets: list[list[int]] = []
    for descs in sessions:
        pos, chosen, selected_ids = 0, [], []
        episode_budget = min(SAMPLE_BUDGET, len(descs))
        episode_rewards = []
        for step in range(episode_budget):
            s = _state(descs[pos], chosen, pos, len(descs), step)
            st = torch.as_tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.set_grad_enabled(train):
                logits, value = model(st)
                logits = _mask_unreachable_actions(logits, pos, len(descs), step, episode_budget)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample() if train else torch.argmax(logits, dim=-1)
                lp = dist.log_prob(action)
            a = int(action.item())
            selected = descs[pos]
            if chosen:
                novelty = float(np.linalg.norm(selected - np.mean(chosen, axis=0)) /
                                (np.sqrt(selected.size) + 1e-8))
            else:
                novelty = 0.0
            # Reward favors informative new motion while discouraging repeated,
            # adjacent views. This is an information-preservation proxy, not a
            # therapy or clinical outcome reward.
            reward = min(novelty, 3.0) - 0.04
            if selected_ids and pos - selected_ids[-1] <= 1:
                reward -= 0.03
            states.append(s)
            actions.append(a)
            log_probs.append(float(lp.detach().cpu()))
            values.append(float(value.detach().cpu()))
            episode_rewards.append(reward)
            chosen.append(selected)
            selected_ids.append(pos)
            pos = min(len(descs) - 1, pos + STRIDES[a])
        chosen_sets.append(selected_ids)
        rewards.extend(episode_rewards)
    return {"states": states, "actions": actions, "log_probs": log_probs,
            "rewards": rewards, "values": values, "chosen": chosen_sets}


def _train_policy(train_descs: list[np.ndarray], seed: int) -> ActorCritic:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    state_dim = train_descs[0].shape[1] + 3
    model = ActorCritic(state_dim, len(STRIDES)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=PPO_LR)
    for _ in range(PPO_EPISODES):
        chosen_sessions = rng.choice(len(train_descs), size=min(PPO_BATCH_SESSIONS, len(train_descs)),
                                     replace=len(train_descs) < PPO_BATCH_SESSIONS)
        batch_sessions = [train_descs[int(i)] for i in chosen_sessions]
        batch = _rollout(model, batch_sessions, train=True, rng=rng)
        r = np.asarray(batch["rewards"], dtype=np.float32)
        v = np.asarray(batch["values"], dtype=np.float32)
        # Episodes terminate at the fixed observation budget; use discounted
        # returns independently for each session to avoid crossing boundaries.
        returns = np.empty_like(r)
        cursor = 0
        for _session in batch_sessions:
            running = 0.0
            episode_budget = min(SAMPLE_BUDGET, len(_session))
            for k in range(episode_budget - 1, -1, -1):
                idx = cursor + k
                running = r[idx] + GAMMA * running
                returns[idx] = running
            cursor += episode_budget
        adv = returns - v
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        states = torch.as_tensor(np.asarray(batch["states"]), dtype=torch.float32, device=DEVICE)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=DEVICE)
        old_lp = torch.as_tensor(batch["log_probs"], dtype=torch.float32, device=DEVICE)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=DEVICE)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=DEVICE)
        for _ in range(PPO_UPDATES):
            logits, value = model(states)
            dist = torch.distributions.Categorical(logits=logits)
            new_lp = dist.log_prob(actions)
            ratio = torch.exp(new_lp - old_lp)
            s1 = ratio * adv_t
            s2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_t
            policy_loss = -torch.minimum(s1, s2).mean()
            value_loss = torch.nn.functional.mse_loss(value, ret_t)
            entropy = dist.entropy().mean()
            loss = policy_loss + 0.5 * value_loss - ENTROPY_WEIGHT * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
    return model


def _policy_select(model: ActorCritic, descs: np.ndarray) -> list[int]:
    model.eval()
    pos, chosen, chosen_ids = 0, [], []
    episode_budget = min(SAMPLE_BUDGET, len(descs))
    with torch.no_grad():
        for step in range(episode_budget):
            s = _state(descs[pos], chosen, pos, len(descs), step)
            logits, _ = model(torch.as_tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0))
            logits = _mask_unreachable_actions(logits, pos, len(descs), step, episode_budget)
            action = int(torch.argmax(logits, dim=-1).item())
            chosen.append(descs[pos])
            chosen_ids.append(pos)
            pos = min(len(descs) - 1, pos + STRIDES[action])
    return chosen_ids


def _baseline_select(descs: np.ndarray, kind: str, seed: int) -> list[int]:
    n = len(descs)
    budget = min(SAMPLE_BUDGET, n)
    if kind == "uniform":
        return np.linspace(0, n - 1, budget, dtype=int).tolist()
    if kind == "random":
        return sorted(np.random.default_rng(seed).choice(n, budget, replace=False).tolist())
    if kind == "heuristic":
        # Deterministic novelty greedy selection over cheap motion descriptors.
        salience = np.linalg.norm(descs, axis=1)
        chosen = [int(np.argmax(salience))]
        while len(chosen) < budget:
            d = np.min(np.linalg.norm(descs[:, None, :] - descs[np.asarray(chosen)][None, :, :], axis=2), axis=1)
            d[np.asarray(chosen)] = -np.inf
            chosen.append(int(np.argmax(d)))
        return sorted(chosen)
    raise ValueError(kind)


def _aggregate(descs: np.ndarray, ids: list[int]) -> np.ndarray:
    selected = descs[np.asarray(ids, dtype=int)]
    return np.concatenate([selected.mean(axis=0), selected.std(axis=0)]).astype(np.float32)


class LinearSoftmax:
    """Small NumPy-free-dependency classifier implemented with PyTorch."""
    def __init__(self, in_dim: int, n_classes: int, seed: int) -> None:
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.15),
                                 nn.Linear(128, n_classes)).to(DEVICE)

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 100) -> None:
        xt = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
        yt = torch.as_tensor(y, dtype=torch.long, device=DEVICE)
        counts = torch.bincount(yt).float().clamp_min(1)
        weights = (counts.sum() / (len(counts) * counts)).to(DEVICE)
        opt = torch.optim.AdamW(self.net.parameters(), lr=1e-3, weight_decay=1e-4)
        self.net.train()
        for _ in range(epochs):
            logits = self.net(xt)
            loss = nn.functional.cross_entropy(logits, yt, weight=weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            return self.net(torch.as_tensor(x, dtype=torch.float32, device=DEVICE)).argmax(1).cpu().numpy()


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"MMASD feature file is missing: {INPUT}")
    data = np.load(INPUT, allow_pickle=True)
    sequences = data["features"]
    labels = data["labels"].astype(int)
    groups = data["subjects"].astype(str)
    descs = [_window_descriptors(s) for s in sequences]
    methods = ("uniform", "random", "heuristic", "ppo")
    all_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for fold, (tr, te) in enumerate(_group_folds(groups, SEED), 1):
        # Standardize descriptors using only training-subject windows.
        mean = np.concatenate([descs[i] for i in tr]).mean(axis=0)
        std = np.concatenate([descs[i] for i in tr]).std(axis=0)
        std[std < 1e-6] = 1.0
        fold_descs = [(d - mean) / std for d in descs]
        policy = _train_policy([fold_descs[i] for i in tr], SEED + fold)
        for method in methods:
            xtr, xte = [], []
            for i in tr:
                ids = (_policy_select(policy, fold_descs[i]) if method == "ppo"
                       else _baseline_select(fold_descs[i], method, SEED + fold + i))
                xtr.append(_aggregate(fold_descs[i], ids))
            for i in te:
                ids = (_policy_select(policy, fold_descs[i]) if method == "ppo"
                       else _baseline_select(fold_descs[i], method, SEED + fold + i))
                xte.append(_aggregate(fold_descs[i], ids))
            xtr, xte = np.asarray(xtr), np.asarray(xte)
            # Fit classifier standardization only on the outer training fold.
            xm, xs = xtr.mean(0), xtr.std(0)
            xs[xs < 1e-6] = 1.0
            xtr, xte = (xtr - xm) / xs, (xte - xm) / xs
            classes = np.unique(labels[tr])
            cls_map = {int(c): j for j, c in enumerate(classes)}
            ytr = np.asarray([cls_map[int(v)] for v in labels[tr]])
            clf = LinearSoftmax(xtr.shape[1], len(classes), SEED + fold)
            clf.fit(xtr, ytr)
            pred = classes[clf.predict(xte)]
            acc, mf1 = _metrics(labels[te], pred)
            row = {"fold": fold, "method": method, "train_sessions": int(len(tr)),
                   "test_sessions": int(len(te)), "sample_budget_windows": SAMPLE_BUDGET,
                   "accuracy": acc, "macro_f1": mf1}
            fold_rows.append(row)
            all_rows.append({"true": labels[te].tolist(), "pred": pred.tolist(), **row})
        print(f"Completed outer fold {fold}/{N_FOLDS} (held-out groups={len(np.unique(groups[te]))})", flush=True)

    summary: dict[str, Any] = {
        "study": "MMASD PPO adaptive temporal window sampling",
        "status": "preliminary execution completed; independent replication and review are still required",
        "seed": SEED,
        "device": str(DEVICE),
        "n_sessions": int(len(labels)),
        "n_groups": int(len(np.unique(groups))),
        "n_classes": int(len(np.unique(labels))),
        "outer_evaluation": "5-fold grouped by stored subject identifier",
        "group_identity_caveat": "Stored filename-derived group identifiers have not been independently verified as participant identities.",
        "sampling": {"window_length": WINDOW, "candidate_stride": 1,
                     "budget_windows": SAMPLE_BUDGET, "allowed_next_strides": list(STRIDES)},
        "policy": {"algorithm": "PPO clipped policy gradient", "training_batches": PPO_EPISODES,
                   "sessions_per_batch": PPO_BATCH_SESSIONS,
                   "updates_per_batch": PPO_UPDATES, "learning_rate": PPO_LR,
                   "reward": "training-session descriptor novelty minus small selection cost; proxy for information preservation, not a clinical outcome"},
        "interpretation_limit": "This is a first implementation of the sampler concept. It does not reproduce the cited graph model, the base paper's optical-flow representation, or their published scores.",
        "fold_results": fold_rows,
        "aggregate": {},
    }
    for method in methods:
        rows = [r for r in fold_rows if r["method"] == method]
        summary["aggregate"][method] = {
            "accuracy_mean": float(np.mean([r["accuracy"] for r in rows])),
            "accuracy_sd": float(np.std([r["accuracy"] for r in rows], ddof=1)),
            "macro_f1_mean": float(np.mean([r["macro_f1"] for r in rows])),
            "macro_f1_sd": float(np.std([r["macro_f1"] for r in rows], ddof=1)),
        }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fold_rows[0]))
        writer.writeheader()
        writer.writerows(fold_rows)
    print(f"Saved {OUT_JSON}")
    print(f"Saved {OUT_CSV}")
    for method, values in summary["aggregate"].items():
        print(f"{method}: accuracy={values['accuracy_mean']:.4f}±{values['accuracy_sd']:.4f}; "
              f"macro-F1={values['macro_f1_mean']:.4f}±{values['macro_f1_sd']:.4f}")


if __name__ == "__main__":
    main()
