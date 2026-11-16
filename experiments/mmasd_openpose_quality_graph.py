"""Local-release MMASD 2D quality-aware graph/temporal experiment.

This is an adapted experiment, not a reproduction of RC-ProtoGCN. It uses the
available OpenPose archive, clip-level labels, filename-derived subject groups,
25 BODY_25 joints, 64 uniformly sampled frames, and a compact graph+GRU model.
The comparison isolates whether OpenPose confidence helps versus the same model
with confidence removed. Archive counts differ from the 1,202-clip configuration
in Ay et al. (2026), so scores must not be presented as a matched replication.
"""
from __future__ import annotations

import collections
import json
import math
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".codex-deps"))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ARCHIVE = ROOT / "data/raw/mmasd/Dataset_FINAL/2D skeleton/2D_openpose.zip"
DATA = ROOT / "data/processed/mmasd_openpose/mmasd_openpose_64frames.npz"
OUT_JSON = ROOT / "results/mmasd_openpose_quality_graph_2026_11_12.json"
OUT_CSV = ROOT / "results/mmasd_openpose_quality_graph_folds_2026_11_12.csv"
SEED = 42
FOLDS = 5
EPOCHS = 30
PATIENCE = 6
BATCH = 32
FRAMES = 64
CLASSES = ["as", "bs", "ce", "dr", "mfs", "ms", "sac", "sq", "tr", "tw"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BODY_25 links (child, parent), with MidHip as the normalization root.
EDGES = [(1,0),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),(1,8),(8,9),(9,10),
         (10,11),(8,12),(12,13),(13,14),(0,15),(15,17),(0,16),(16,18),
         (14,19),(19,20),(14,21),(11,22),(22,23),(11,24)]


def extract() -> dict:
    """Read each clip, track the person heuristically, normalize, and sample."""
    grouped: dict[str, list[tuple[int, str]]] = collections.defaultdict(list)
    with zipfile.ZipFile(ARCHIVE) as zf:
        for path in zf.namelist():
            if not path.endswith("_keypoints.json"):
                continue
            name = Path(path).name
            m = re.match(r"(.+?)_(\d+)_keypoints\.json$", name)
            if m:
                clip_key = path.rsplit("/", 1)[0] + "/" + m.group(1)
                grouped[clip_key].append((int(m.group(2)), path))
        xs, qs, labels, subjects, clip_names = [], [], [], [], []
        folder_counts = collections.Counter()
        skipped = collections.Counter()
        for clip_key, entries in sorted(grouped.items()):
            basename = clip_key.rsplit("/",1)[-1]
            label = basename.split("_")[0]
            if label not in CLASSES:
                skipped["unsupported_label"] += 1
                continue
            subject_match = re.search(r"_(\d{5})_", basename)
            if not subject_match:
                skipped["missing_subject_id"] += 1
                continue
            entries.sort()
            poses, scores, prev_center = [], [], None
            for _, path in entries:
                try:
                    obj = json.loads(zf.read(path))
                except Exception:
                    poses.append(np.zeros((25,2), np.float32)); scores.append(np.zeros(25,np.float32)); continue
                candidates=[]
                for person in obj.get("people", []):
                    flat=np.asarray(person.get("pose_keypoints_2d",[]),dtype=np.float32)
                    if flat.size != 75:
                        continue
                    arr=flat.reshape(25,3)
                    valid=arr[:,2] > 0
                    count=int(valid.sum())
                    conf=float(arr[valid,2].mean()) if count else 0.0
                    torso_ids=[1,2,5,8]
                    torso=arr[torso_ids]
                    tv=torso[:,2] > 0
                    center=torso[tv,:2].mean(axis=0) if tv.any() else np.zeros(2,np.float32)
                    dist=float(np.linalg.norm(center-prev_center)) if prev_center is not None and tv.any() else 0.0
                    candidates.append((count,conf,-dist,arr,center,tv.any()))
                if not candidates:
                    poses.append(np.zeros((25,2),np.float32)); scores.append(np.zeros(25,np.float32)); continue
                best=max(candidates,key=lambda a:(a[0],a[1],a[2]))
                arr=best[3].copy(); prev_center=best[4] if best[5] else prev_center
                xy=arr[:,:2]; q=arr[:,2].clip(0,1)
                # MidHip center, shoulder-width scale. Missing normalization
                # anchors invalidate the frame rather than leaking image position.
                shoulder_ok=q[2] > 0 and q[5] > 0
                if q[8] <= 0 or not shoulder_ok:
                    xy[:]=0; q[:]=0
                else:
                    center=xy[8].copy()
                    width=float(np.linalg.norm(xy[2]-xy[5]))
                    if not np.isfinite(width) or width < 1e-3:
                        xy[:]=0; q[:]=0
                    else:
                        xy=(xy-center)/width
                        xy=np.clip(xy,-5,5)
                        xy[q<=0]=0
                poses.append(np.nan_to_num(xy,nan=0,posinf=0,neginf=0).astype(np.float32))
                scores.append(np.nan_to_num(q,nan=0,posinf=0,neginf=0).astype(np.float32))
            if not poses:
                skipped["empty_clip"] += 1; continue
            p=np.asarray(poses,dtype=np.float32); q=np.asarray(scores,dtype=np.float32)
            indices=np.rint(np.linspace(0,len(p)-1,FRAMES)).astype(int)
            p=p[indices]; q=q[indices]
            # Add first-order motion as input channels; first frame has zero motion.
            motion=np.diff(p,axis=0,prepend=p[:1])
            xs.append(np.concatenate([p,motion],axis=-1))
            qs.append(q)
            labels.append(CLASSES.index(label)); subjects.append(subject_match.group(1)); clip_names.append(clip_key)
            folder_counts[label]+=1
    X=np.asarray(xs,dtype=np.float32); Q=np.asarray(qs,dtype=np.float32)
    DATA.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(DATA,X=X,Q=Q,y=np.asarray(labels,dtype=np.int64),
                        groups=np.asarray(subjects),clips=np.asarray(clip_names),
                        classes=np.asarray(CLASSES))
    return {"n_clips":len(labels),"n_subjects":len(set(subjects)),"class_counts":dict(folder_counts),
            "skipped":dict(skipped),"X_shape":list(X.shape),"Q_shape":list(Q.shape)}


def folds_for(groups: np.ndarray, y: np.ndarray) -> list[tuple[np.ndarray,np.ndarray]]:
    rng=np.random.default_rng(SEED); unique=np.unique(groups)
    tie={g:rng.random() for g in unique}
    order=sorted(unique,key=lambda g:(-int(np.sum(groups==g)),tie[g]))
    bins=[[] for _ in range(FOLDS)]; counts=np.zeros(FOLDS,dtype=int)
    for g in order:
        k=int(np.argmin(counts)); bins[k].append(g); counts[k]+=int(np.sum(groups==g))
    return [(np.flatnonzero(~np.isin(groups,b)),np.flatnonzero(np.isin(groups,b))) for b in bins]


def metric(y: np.ndarray, pred: np.ndarray) -> tuple[float,float]:
    acc=float(np.mean(y==pred)); f=[]
    for c in range(len(CLASSES)):
        tp=np.sum((y==c)&(pred==c)); fp=np.sum((y!=c)&(pred==c)); fn=np.sum((y==c)&(pred!=c))
        f.append(float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.)
    return acc,float(np.mean(f))


def adjacency() -> torch.Tensor:
    a=torch.eye(25)
    for i,j in EDGES: a[i,j]=a[j,i]=1
    d=a.sum(1).clamp_min(1).rsqrt()
    return d[:,None]*a*d[None,:]


class GraphTemporal(nn.Module):
    def __init__(self, quality: bool):
        super().__init__(); self.use_quality=quality
        self.register_buffer("adj",adjacency())
        self.in_proj=nn.Linear(4,48)
        self.g1=nn.Linear(48,64); self.g2=nn.Linear(64,64)
        self.gru=nn.GRU(128,96,batch_first=True,bidirectional=True)
        self.head=nn.Sequential(nn.Dropout(.25),nn.Linear(192,96),nn.ReLU(),nn.Dropout(.2),nn.Linear(96,len(CLASSES)))
    def forward(self,x,q):
        # x: B,T,V,4; q: B,T,V
        b,t,v,c=x.shape
        h=self.in_proj(x)
        if self.use_quality: h=h*(.25+.75*q.unsqueeze(-1))
        h=torch.einsum('vw,btwc->btvc',self.adj,h)
        h=F.relu(self.g1(h)); h=F.relu(self.g2(torch.einsum('vw,btwc->btvc',self.adj,h)))
        if self.use_quality:
            w=q/(q.sum(-1,keepdim=True).clamp_min(1e-5))
            pooled=(h*w.unsqueeze(-1)).sum(2)
            visible=q.mean(-1,keepdim=True)
            pooled=torch.cat([pooled,visible.expand(-1,-1,pooled.shape[-1])],dim=-1)
        else: pooled=torch.cat([h.mean(2),h.std(2,unbiased=False)],dim=-1)
        z,_=self.gru(pooled); return self.head(z.mean(1))


def train_fold(X,Q,y,train_idx,test_idx,fold:int,use_quality:bool):
    # Reserve a subject-grouped validation subset from outer training only.
    train_groups=np.unique(groups[train_idx]); rng=np.random.default_rng(SEED+fold)
    rng.shuffle(train_groups); nval=max(1,round(.18*len(train_groups)))
    val_groups=set(train_groups[:nval]); val_idx=train_idx[np.isin(groups[train_idx],list(val_groups))]
    fit_idx=train_idx[~np.isin(groups[train_idx],list(val_groups))]
    mu=X[fit_idx].mean(axis=(0,1,2),keepdims=True); sd=X[fit_idx].std(axis=(0,1,2),keepdims=True); sd[sd<1e-5]=1
    Z=np.clip((X-mu)/sd,-8,8).astype(np.float32)
    torch.manual_seed(SEED+fold); np.random.seed(SEED+fold)
    model=GraphTemporal(use_quality).to(DEVICE)
    counts=np.bincount(y[fit_idx],minlength=len(CLASSES)).astype(np.float32)
    weights=np.zeros(len(CLASSES),np.float32); present=counts>0
    weights[present]=counts[present].sum()/(present.sum()*counts[present])
    opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-4)
    best=-1; state=None; wait=0; best_epoch=0
    for epoch in range(1,EPOCHS+1):
        model.train(); sh=np.random.permutation(fit_idx)
        for st in range(0,len(sh),BATCH):
            ix=sh[st:st+BATCH]
            xb=torch.tensor(Z[ix],device=DEVICE); qb=torch.tensor(Q[ix],device=DEVICE)
            # confidence dropout augmentation is applied only to the model input
            if use_quality:
                drop=torch.rand_like(qb)<.05; qb=qb.masked_fill(drop,0)
            logits=model(xb,qb)
            loss=F.cross_entropy(logits,torch.tensor(y[ix],device=DEVICE),weight=torch.tensor(weights,device=DEVICE))
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        model.eval()
        with torch.no_grad():
            pred=[]
            for st in range(0,len(val_idx),BATCH):
                ix=val_idx[st:st+BATCH]
                pred.extend(model(torch.tensor(Z[ix],device=DEVICE),torch.tensor(Q[ix],device=DEVICE)).argmax(1).cpu().numpy())
        _,f1=metric(y[val_idx],np.asarray(pred))
        if f1>best+1e-5:
            best=f1; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; best_epoch=epoch; wait=0
        else:
            wait+=1
            if wait>=PATIENCE: break
    model.load_state_dict(state); model.eval()
    with torch.no_grad():
        pred=[]
        for st in range(0,len(test_idx),BATCH):
            ix=test_idx[st:st+BATCH]
            pred.extend(model(torch.tensor(Z[ix],device=DEVICE),torch.tensor(Q[ix],device=DEVICE)).argmax(1).cpu().numpy())
    acc,f1=metric(y[test_idx],np.asarray(pred))
    return {"fold":fold,"variant":"confidence_aware" if use_quality else "confidence_blind",
            "train_clips":len(fit_idx),"validation_clips":len(val_idx),"test_clips":len(test_idx),
            "test_subjects":len(np.unique(groups[test_idx])),"best_epoch":best_epoch,
            "accuracy":acc,"macro_f1":f1}


def main():
    extraction=extract() if not DATA.exists() else {"cached":True}
    global X,Q,y,groups
    with np.load(DATA,allow_pickle=False) as d:
        X=d["X"].astype(np.float32); Q=d["Q"].astype(np.float32); y=d["y"].astype(np.int64); groups=d["groups"].astype(str)
    folds=folds_for(groups,y); rows=[]
    for fi,(tr,te) in enumerate(folds,1):
        for quality in (False,True):
            print(f"fold={fi} quality={quality} ntrain={len(tr)} ntest={len(te)}",flush=True)
            rows.append(train_fold(X,Q,y,tr,te,fi,quality))
    import csv
    OUT_CSV.parent.mkdir(parents=True,exist_ok=True)
    with OUT_CSV.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary={}
    for variant in ("confidence_blind","confidence_aware"):
        r=[x for x in rows if x["variant"]==variant]
        summary[variant]={k:{"mean":float(np.mean([a[k] for a in r])),"sd":float(np.std([a[k] for a in r],ddof=1))} for k in ("accuracy","macro_f1")}
    result={"study":"MMASD local-release OpenPose quality-aware graph temporal comparison",
      "status":"completed adapted experiment; not an exact reproduction of RC-ProtoGCN",
      "dataset":{"archive":str(ARCHIVE.relative_to(ROOT)),"processed_cache":str(DATA.relative_to(ROOT)),**extraction,
       "n_clips":len(y),"n_subjects":len(np.unique(groups)),"class_counts":dict(zip(CLASSES,np.bincount(y,minlength=len(CLASSES)).tolist())),
       "archive_paper_count_note":"The local archive has a different class distribution and 1,374 clips versus the cited paper's 1,202; no undocumented filtering was applied."},
      "protocol":{"clip_level":True,"frames":FRAMES,"joints":25,"channels":"normalized xy plus first-order motion xy","outer":"five subject-grouped folds from filename-derived IDs","validation":"grouped holdout drawn only from each outer training fold","seed":SEED,"device":str(DEVICE)},
      "method":{"baseline":"same graph+BiGRU network, confidence not provided to graph/pooling","proposed":"OpenPose confidence gates node features and weights graph pooling; confidence dropout augmentation","limitations":"local archive count differs from published 10-class setup; simple graph-GRU is not ProtoGCN or RC-ProtoGCN; filename IDs not independently verified as participants"},
      "summary":summary,"fold_results":rows}
    OUT_JSON.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"dataset":result["dataset"],"summary":summary,"json":str(OUT_JSON),"csv":str(OUT_CSV)},indent=2),flush=True)


if __name__=="__main__": main()
