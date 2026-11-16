"""Temporal-summary MLP reference on the cached local MMASD OpenPose clips."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/".codex-deps"))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

DATA=ROOT/"data/processed/mmasd_openpose/mmasd_openpose_64frames.npz"
OUT_JSON=ROOT/"results/mmasd_openpose_temporal_features_2026_11_12.json"
OUT_CSV=ROOT/"results/mmasd_openpose_temporal_features_folds_2026_11_12.csv"
CLASSES=["as","bs","ce","dr","mfs","ms","sac","sq","tr","tw"]
SEED=42; DEVICE=torch.device("cpu"); BATCH=64; EPOCHS=100; PATIENCE=12

def features(X):
    # Retain broad temporal trajectory: moments over the full clip, then
    # mean and standard deviation in eight ordered temporal bins.
    b,t,v,c=X.shape
    parts=[X.mean(1).reshape(b,-1),X.std(1).reshape(b,-1),
           np.quantile(X,.1,axis=1).reshape(b,-1),np.quantile(X,.9,axis=1).reshape(b,-1)]
    for seg in np.array_split(np.arange(t),8):
        z=X[:,seg]
        parts.extend([z.mean(1).reshape(b,-1),z.std(1).reshape(b,-1)])
    return np.concatenate(parts,axis=1).astype(np.float32)

def metric(y,p):
    f=[]
    for k in range(len(CLASSES)):
        tp=np.sum((y==k)&(p==k)); fp=np.sum((y!=k)&(p==k)); fn=np.sum((y==k)&(p!=k))
        f.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return float(np.mean(y==p)),float(np.mean(f))

def grouped_folds(groups):
    rng=np.random.default_rng(SEED); ug=np.unique(groups); tie={g:rng.random() for g in ug}
    order=sorted(ug,key=lambda g:(-int(np.sum(groups==g)),tie[g])); bins=[[] for _ in range(5)]; size=np.zeros(5,int)
    for g in order:
        i=int(np.argmin(size)); bins[i].append(g); size[i]+=int(np.sum(groups==g))
    return [(np.flatnonzero(~np.isin(groups,b)),np.flatnonzero(np.isin(groups,b))) for b in bins]

class MLP(nn.Module):
    def __init__(self,d):
        super().__init__(); self.net=nn.Sequential(nn.Linear(d,512),nn.ReLU(),nn.BatchNorm1d(512),nn.Dropout(.4),
            nn.Linear(512,256),nn.ReLU(),nn.BatchNorm1d(256),nn.Dropout(.35),nn.Linear(256,len(CLASSES)))
    def forward(self,x): return self.net(x)

def run_fold(Xf,y,groups,tr,te,fold):
    rng=np.random.default_rng(SEED+fold); ug=np.unique(groups[tr]); rng.shuffle(ug)
    nval=max(1,round(.18*len(ug))); valgroups=ug[:nval]
    va=tr[np.isin(groups[tr],valgroups)]; fit=tr[~np.isin(groups[tr],valgroups)]
    mu=Xf[fit].mean(0); sd=Xf[fit].std(0); sd[sd<1e-5]=1.
    Z=np.clip((Xf-mu)/sd,-8,8).astype(np.float32)
    torch.manual_seed(SEED+fold); model=MLP(Z.shape[1]).to(DEVICE)
    counts=np.bincount(y[fit],minlength=len(CLASSES)).astype(np.float32); w=np.zeros_like(counts); p=counts>0
    w[p]=counts[p].sum()/(p.sum()*counts[p]); w=torch.tensor(w,dtype=torch.float32)
    opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-4)
    best=-1; best_state=None; wait=0; best_epoch=0
    for epoch in range(1,EPOCHS+1):
        model.train(); order=rng.permutation(fit)
        for start in range(0,len(order),BATCH):
            ix=order[start:start+BATCH]
            xb=torch.tensor(Z[ix]); xb=xb+torch.randn_like(xb)*.015
            logits=model(xb); loss=F.cross_entropy(logits,torch.tensor(y[ix]),weight=w)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        model.eval()
        with torch.no_grad(): pv=model(torch.tensor(Z[va])).argmax(1).numpy()
        _,f=metric(y[va],pv)
        if f>best+1e-5:
            best=f; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}; wait=0; best_epoch=epoch
        else:
            wait+=1
            if wait>=PATIENCE: break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad(): pred=model(torch.tensor(Z[te])).argmax(1).numpy()
    acc,f=metric(y[te],pred)
    return {"fold":fold,"train_clips":len(fit),"validation_clips":len(va),"test_clips":len(te),
        "test_subjects":len(np.unique(groups[te])),"best_epoch":best_epoch,"accuracy":acc,"macro_f1":f}

def main():
    with np.load(DATA,allow_pickle=False) as d:
        X=d['X'].astype(np.float32); y=d['y'].astype(np.int64); groups=d['groups'].astype(str)
    Xf=features(X); rows=[]
    for fold,(tr,te) in enumerate(grouped_folds(groups),1):
        print(f"fold={fold} train={len(tr)} test={len(te)}",flush=True)
        rows.append(run_fold(Xf,y,groups,tr,te,fold))
    OUT_CSV.parent.mkdir(parents=True,exist_ok=True)
    with OUT_CSV.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary={m:{'mean':float(np.mean([r[m] for r in rows])),'sd':float(np.std([r[m] for r in rows],ddof=1))} for m in ('accuracy','macro_f1')}
    result={'study':'MMASD local 2D OpenPose temporal-summary MLP reference','status':'completed adapted local-release experiment; not a matched published reproduction',
        'data':{'n_clips':len(y),'n_subjects':len(np.unique(groups)),'n_classes':len(CLASSES),'feature_dimension':int(Xf.shape[1]),
            'note':'Same cached 1,374-clip local archive and filename-derived subject folds as the graph experiment; counts differ from the cited 1,202-clip setup.'},
        'protocol':{'outer':'5-fold subject-grouped CV','validation':'subject groups held out from outer training set','seed':SEED,'feature':'global joint-coordinate/motion moments, 10th/90th quantiles, and mean/std in eight ordered temporal bins'},
        'model':{'architecture':'512-256-10 MLP','optimizer':'AdamW','learning_rate':0.0008,'weight_decay':0.0001,'class_weighted_loss':True,'early_stopping':'validation macro-F1, patience 12'},
        'summary':summary,'fold_results':rows}
    OUT_JSON.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'summary':summary,'json':str(OUT_JSON)},indent=2),flush=True)

if __name__=='__main__': main()
