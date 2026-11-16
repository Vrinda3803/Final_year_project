# ============================================================
# SPARSE adaptive graphs + strong 4-class ablation
# Goal: clean signal, fewer nodes, temporal edges only
# ============================================================
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from collections import Counter
from tqdm.auto import tqdm
import random
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(r"/content/drive/MyDrive/Autism-Behaviour-Graph-XAI/Autism-Behaviour-Graph-XAI")
KPTS_DIR = BASE_DIR / "data" / "processed" / "keypoints" / "av_asd"
LABELS_CSV = BASE_DIR / "data" / "processed" / "behaviours" / "avasd_labels.csv"
FEAT_CSV = BASE_DIR / "data" / "processed" / "behaviours" / "avasd_features.csv"
GRAPH_DIR = BASE_DIR / "data" / "graphs" / "av_asd_sparse"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# ---------- 1. Keep only the 4 largest motor classes ----------
KEEP = [
    "Aggressive Behavior",
    "Hyper- or Hyporeactivity to Sensory Input",
    "Upper Limb Stereotypies",
    "Self-Spinning or Spinning Objects",
]
labels = pd.read_csv(LABELS_CSV)
labels = labels[labels["primary_label"].isin(KEEP)].reset_index(drop=True)
print("Clips:", len(labels))
print(labels["primary_label"].value_counts())

# ---------- 2. Sparse graph builder ----------
# Only upper-body groups (most relevant for stimming)
JOINT_GROUPS = {
    "head":         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "upper_limb_L": [11, 13, 15],
    "upper_limb_R": [12, 14, 16],
    "torso":        [11, 12, 23, 24],
}

def group_feats(traj, fps=30.0):
    if len(traj) < 3:
        return np.zeros(8, dtype=np.float32)
    speed = np.linalg.norm(np.diff(traj, axis=0) * fps, axis=1)
    centred = traj - traj.mean(0)
    ang = np.arctan2(centred[:, 1], centred[:, 0])
    ang_vel = np.abs(np.diff(np.unwrap(ang))) * fps
    return np.array([
        speed.mean(), speed.std(), speed.max(),
        ang_vel.mean() if len(ang_vel) else 0.0,
        ang_vel.max() if len(ang_vel) else 0.0,
        traj[:, 0].std(), traj[:, 1].std(),
        len(traj) / fps,
    ], dtype=np.float32)

def build_sparse_graph(kpts, fps=30.0, window_sec=3.0, stride_sec=1.5, max_gap=2.0, max_nodes=24):
    if kpts.ndim == 2:
        L = kpts.shape[1] // 3
        kpts = kpts.reshape(kpts.shape[0], L, 3)
    body = kpts[:, :33, :2]
    T = body.shape[0]
    win = max(8, int(window_sec * fps))
    stride = max(4, int(stride_sec * fps))

    feats, ts, gids = [], [], []
    for start in range(0, max(1, T - win + 1), stride):
        end = min(start + win, T)
        t_c = (start + end) / 2 / fps
        for gi, (name, idxs) in enumerate(JOINT_GROUPS.items()):
            valid = [i for i in idxs if i < body.shape[1]]
            traj = body[start:end, valid].mean(1)
            feats.append(group_feats(traj, fps))
            ts.append(t_c)
            gids.append(gi)

    if not feats:
        return None

    # Cap nodes: keep evenly spaced
    if len(feats) > max_nodes:
        idx = np.linspace(0, len(feats) - 1, max_nodes).astype(int)
        feats = [feats[i] for i in idx]
        ts = [ts[i] for i in idx]
        gids = [gids[i] for i in idx]

    x = torch.tensor(np.stack(feats), dtype=torch.float)
    ts = np.array(ts)

    # TEMPORAL EDGES ONLY (sparse)
    src, dst, attr = [], [], []
    for i in range(len(ts)):
        for j in range(i + 1, len(ts)):
            dt = abs(ts[i] - ts[j])
            if dt <= max_gap:
                w = float(1.0 / (1.0 + dt))
                src.extend([i, j]); dst.extend([j, i]); attr.extend([w, w])

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(attr, dtype=torch.float).unsqueeze(1)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1))

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.timestamps = torch.tensor(ts, dtype=torch.float)
    data.group = torch.tensor(gids, dtype=torch.long)
    data.num_nodes = x.size(0)
    return data

# Build sparse graphs
rows = []
graphs_by_id = {}
for _, row in tqdm(labels.iterrows(), total=len(labels), desc="Sparse graphs"):
    vid = row["Video_ID"]
    npy = KPTS_DIR / f"{vid}.npy"
    if not npy.exists():
        continue
    g = build_sparse_graph(np.load(npy))
    if g is None or g.num_nodes < 4:
        continue
    torch.save(g, GRAPH_DIR / f"{vid}_graph.pt")
    graphs_by_id[vid] = g
    n_edges = g.edge_index.size(1) // 2
    rows.append({
        "Video_ID": vid,
        "primary_label": row["primary_label"],
        "subject_id": row["subject_id"],
        "num_nodes": g.num_nodes,
        "num_edges": n_edges,
        "avg_degree": 2 * n_edges / g.num_nodes,
    })

diag = pd.DataFrame(rows)
diag.to_csv(GRAPH_DIR / "diagnostics.csv", index=False)
print("\nSparse graphs:", len(diag))
print(diag[["num_nodes", "num_edges", "avg_degree"]].describe().round(2))

# ---------- 3. Strong RF baseline on same 4 classes (for comparison) ----------
feat_df = pd.read_csv(FEAT_CSV)
feat_df = feat_df[feat_df["primary_label"].isin(KEEP)].reset_index(drop=True)
fcols = [c for c in feat_df.columns if c not in
         ("Video_ID", "subject_id", "primary_label", "pose_centric", "n_frames", "duration_sec")]
Xrf = feat_df[fcols].fillna(0).values
yrf = feat_df["primary_label"].values
grf = feat_df["subject_id"].values
le_rf = LabelEncoder()
yrf_enc = le_rf.fit_transform(yrf)
clf = RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_leaf=2,
                             class_weight="balanced", random_state=42, n_jobs=-1)
pred_rf = cross_val_predict(clf, Xrf, yrf_enc, groups=grf, cv=GroupKFold(5))
print("\n===== RF 4-class baseline =====")
print(f"Acc={accuracy_score(yrf_enc, pred_rf):.4f}  Macro-F1={f1_score(yrf_enc, pred_rf, average='macro'):.4f}")
print(classification_report(yrf_enc, pred_rf, target_names=le_rf.classes_, zero_division=0))

# ---------- 4. GNN models ----------
class GCNClf(nn.Module):
    def __init__(self, in_dim, hidden=64, n_classes=4, dropout=0.4):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.fc = nn.Linear(hidden, n_classes)
        self.dropout = dropout
    def forward(self, data):
        x, ei = data.x, data.edge_index
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = F.relu(self.conv1(x, ei))
        x = F.dropout(x, self.dropout, self.training)
        x = F.relu(self.conv2(x, ei))
        x = global_mean_pool(x, batch)
        return self.fc(x)

class SeqOnly(nn.Module):
    def __init__(self, in_dim, hidden=64, n_classes=4):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True, num_layers=1)
        self.fc = nn.Linear(hidden, n_classes)
    def forward(self, data):
        x = data.x.unsqueeze(0)
        _, h = self.gru(x)
        return self.fc(h[-1])

def no_edges(data):
    d = Data(x=data.x, edge_index=torch.zeros((2, 0), dtype=torch.long))
    d.num_nodes = data.num_nodes
    return d

# Align graphs with labels
graphs, y_list, subj_list = [], [], []
for _, row in diag.iterrows():
    g = graphs_by_id[row["Video_ID"]]
    graphs.append(g)
    y_list.append(row["primary_label"])
    subj_list.append(row["subject_id"])

le = LabelEncoder()
y = le.fit_transform(y_list)
subjects = np.array(subj_list)
n_classes = len(le.classes_)
in_dim = graphs[0].x.size(1)
print("GNN samples:", len(graphs), "| classes:", list(le.classes_))

# Class weights
cnt = Counter(y)
weights = torch.tensor([len(y) / (n_classes * cnt[i]) for i in range(n_classes)],
                       dtype=torch.float, device=DEVICE)

def train_model(model, tr_g, tr_y, epochs=60, lr=5e-4):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    crit = nn.CrossEntropyLoss(weight=weights)
    model.train()
    for ep in range(epochs):
        order = list(range(len(tr_g)))
        random.shuffle(order)
        for i in order:
            g = tr_g[i].to(DEVICE)
            opt.zero_grad()
            out = model(g)
            loss = crit(out, torch.tensor([tr_y[i]], device=DEVICE))
            loss.backward()
            opt.step()
    return model

@torch.no_grad()
def predict(model, gs):
    model.eval()
    return np.array([model(g.to(DEVICE)).argmax(1).item() for g in gs])

def run_ablation(epochs=60):
    gkf = GroupKFold(5)
    res = {k: {"yt": [], "yp": []} for k in ["gcn_sparse", "gcn_no_edges", "sequence_only"]}
    for fold, (tr, te) in enumerate(gkf.split(graphs, y, groups=subjects)):
        print(f"Fold {fold+1}/5  n_train={len(tr)} n_test={len(te)}")
        tr_g, te_g = [graphs[i] for i in tr], [graphs[i] for i in te]
        tr_y, te_y = y[tr], y[te]

        # Sparse GCN (temporal edges)
        m = train_model(GCNClf(in_dim, n_classes=n_classes), tr_g, tr_y, epochs=epochs)
        pred = predict(m, te_g)
        res["gcn_sparse"]["yt"].extend(te_y); res["gcn_sparse"]["yp"].extend(pred)

        # No edges (structure ablation)
        tr_ne = [no_edges(g) for g in tr_g]
        te_ne = [no_edges(g) for g in te_g]
        m = train_model(GCNClf(in_dim, n_classes=n_classes), tr_ne, tr_y, epochs=epochs)
        pred = predict(m, te_ne)
        res["gcn_no_edges"]["yt"].extend(te_y); res["gcn_no_edges"]["yp"].extend(pred)

        # Sequence-only
        m = train_model(SeqOnly(in_dim, n_classes=n_classes), tr_g, tr_y, epochs=epochs)
        pred = predict(m, te_g)
        res["sequence_only"]["yt"].extend(te_y); res["sequence_only"]["yp"].extend(pred)

    print("\n========== 4-CLASS ABLATION ==========")
    for name, d in res.items():
        yt, yp = np.array(d["yt"]), np.array(d["yp"])
        acc = accuracy_score(yt, yp)
        f1 = f1_score(yt, yp, average="macro", zero_division=0)
        print(f"{name:20s}  Acc={acc:.4f}  Macro-F1={f1:.4f}")
        print(classification_report(yt, yp, target_names=le.classes_, zero_division=0))
    return res

results = run_ablation(epochs=60)