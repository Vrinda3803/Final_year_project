"""Ablation of confidence-weighted temporal descriptors on local MMASD 2D data.

The experiment tests, rather than assumes, whether per-joint OpenPose confidence
improves the strongest local baseline. Evaluation uses the same outer subject
folds and inner validation split as mmasd_openpose_quality_graph.py.
"""
from __future__ import annotations
import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/".codex-deps"))
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from experiments.mmasd_openpose_temporal_features import grouped_folds,metric,MLP

DATA=ROOT/"data/processed/mmasd_openpose/mmasd_openpose_64frames.npz"
OUT_JSON=ROOT/"results/mmasd_openpose_reliability_features_2026_11_12.json"
OUT_CSV=ROOT/"results/mmasd_openpose_reliability_features_folds_2026_11_12.csv"
SEED=42; BATCH=64; EPOCHS=100; PATIENCE=12

def build_features(X,Q,aware=False):
    b,t,v,c=X.shape
    def moments(z,weights=None):
        if weights is None: return z.mean(1),z.std(1)
        w=weights[...,None]; den=w.sum(1).clip(1e-5)
        mean=(z*w).sum(1)/den
        var=(((z-mean[:,None])**2)*w).sum(1)/den
        return mean,np.sqrt(np.maximum(var,0))
    # Shared unweighted descriptor bank.
    parts=[]
    mu,sd=moments(X); parts.extend([mu.reshape(b,-1),sd.reshape(b,-1),
        np.quantile(X,.1,axis=1).reshape(b,-1),np.quantile(X,.9,axis=1).reshape(b,-1)])
    bins=np.array_split(np.arange(t),8)
    for ids in bins:
        z=X[:,ids]; parts.extend([z.mean(1).reshape(b,-1),z.std(1).reshape(b,-1)])
    if aware:
        # Confidence-weighted joint trajectories and confidence coverage are
        # kept separately so the classifier can learn to trust clean joints.
        wm,ws=moments(X,Q); parts.extend([wm.reshape(b,-1),ws.reshape(b,-1)])
        parts.extend([Q.mean(1),Q.std(1)])
        for ids in bins:
            z=X[:,ids]; q=Q[:,ids]
            bm,bs=moments(z,q); parts.extend([bm.reshape(b,-1),bs.reshape(b,-1),
                q.mean(1),q.std(1)])
    return np.concatenate(parts,axis=1).astype(np.float32)

def train_one(Xf,y,groups,tr,te,fold):
    rng=np.random.default_rng(SEED+fold); ug=np.unique(groups[tr]); rng.shuffle(ug)
    nv=max(1,round(.18*len(ug))); vg=ug[:nv]
    va=tr[np.isin(groups[tr],vg)]; fit=tr[~np.isin(groups[tr],vg)]
    mean=Xf[fit].mean(0); sd=Xf[fit].std(0); sd[sd<1e-5]=1
    Z=np.clip((Xf-mean)/sd,-8,8).astype(np.float32)
    torch.manual_seed(SEED+fold); model=MLP(Z.shape[1])
    cnt=np.bincount(y[fit],minlength=10).astype(np.float32); w=np.zeros(10,np.float32); ok=cnt>0
    w[ok]=cnt[ok].sum()/(ok.sum()*cnt[ok]); w=torch.tensor(w)
    opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-4)
    best=-1; state=None; wait=0; epoch_best=0
    for epoch in range(1,EPOCHS+1):
        model.train(); order=rng.permutation(fit)
        for st in range(0,len(order),BATCH):
            ids=order[st:st+BATCH]; xb=torch.tensor(Z[ids]); xb=xb+torch.randn_like(xb)*.015
            loss=F.cross_entropy(model(xb),torch.tensor(y[ids]),weight=w)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        model.eval()
        with torch.no_grad(): pred=model(torch.tensor(Z[va])).argmax(1).numpy()
        _,f=metric(y[va],pred)
        if f>best+1e-5:
            best=f; state={k:v.detach().clone() for k,v in model.state_dict().items()}; wait=0; epoch_best=epoch
        else:
            wait+=1
            if wait>=PATIENCE: break
    model.load_state_dict(state); model.eval()
    with torch.no_grad(): pred=model(torch.tensor(Z[te])).argmax(1).numpy()
    acc,f=metric(y[te],pred)
    return {'fold':fold,'train_clips':len(fit),'validation_clips':len(va),'test_clips':len(te),
        'best_epoch':epoch_best,'accuracy':acc,'macro_f1':f}

def main():
    with np.load(DATA,allow_pickle=False) as d:
        X=d['X'].astype(np.float32); Q=d['Q'].astype(np.float32); y=d['y'].astype(np.int64); groups=d['groups'].astype(str)
    rows=[]; summary={}
    for aware,name in ((False,'temporal_mlp'),(True,'confidence_weighted_temporal_mlp')):
        Xf=build_features(X,Q,aware)
        res=[]
        for fold,(tr,te) in enumerate(grouped_folds(groups),1):
            print(f'variant={name} fold={fold} d={Xf.shape[1]}',flush=True)
            r=train_one(Xf,y,groups,tr,te,fold); r['variant']=name; r['feature_dimension']=Xf.shape[1]
            rows.append(r); res.append(r)
        summary[name]={k:{'mean':float(np.mean([r[k] for r in res])),'sd':float(np.std([r[k] for r in res],ddof=1))} for k in ('accuracy','macro_f1')}
    OUT_CSV.parent.mkdir(parents=True,exist_ok=True)
    with OUT_CSV.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    out={'study':'MMASD confidence-weighted temporal descriptor ablation','status':'completed one-seed adapted local-release experiment',
        'data':{'clips':len(y),'subjects':len(np.unique(groups)),'classes':10,'archive_note':'Local archive count and class distribution differ from published 1,202-clip setup.'},
        'protocol':{'outer':'same 5 subject-grouped folds as graph and temporal MLP runs','inner_validation':'same 18% grouped split rule and seed by fold','seed':SEED},
        'variants':{'temporal_mlp':'Unweighted global and eight-bin coordinate/motion moments plus quantiles.',
            'confidence_weighted_temporal_mlp':'Same bank plus confidence-weighted means/variances and per-joint confidence coverage/stability descriptors.'},
        'summary':summary,'fold_results':rows}
    OUT_JSON.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({'summary':summary,'json':str(OUT_JSON)},indent=2),flush=True)

if __name__=='__main__': main()
