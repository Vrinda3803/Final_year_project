"""Joint/bone/motion temporal-summary feature experiment on local MMASD 2D."""
from __future__ import annotations
import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/".codex-deps")); sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from experiments.mmasd_openpose_temporal_features import grouped_folds,metric,MLP

DATA=ROOT/"data/processed/mmasd_openpose/mmasd_openpose_64frames.npz"
OUT_JSON=ROOT/"results/mmasd_openpose_multistream_features_2026_11_12.json"
OUT_CSV=ROOT/"results/mmasd_openpose_multistream_features_folds_2026_11_12.csv"
CLASSES=['as','bs','ce','dr','mfs','ms','sac','sq','tr','tw']
EDGES=[(1,0),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),(1,8),(8,9),(9,10),(10,11),
       (8,12),(12,13),(13,14),(0,15),(15,17),(0,16),(16,18),(14,19),(19,20),
       (14,21),(11,22),(22,23),(11,24)]
SEED=42; BATCH=64; EPOCHS=100; PATIENCE=12

def make_features(X):
    p=X[...,:2]; v=X[...,2:4]
    bones=np.stack([p[:,:,c]-p[:,:,par] for c,par in EDGES],axis=2)
    bone_motion=np.stack([v[:,:,c]-v[:,:,par] for c,par in EDGES],axis=2)
    streams=[X,np.concatenate([bones,bone_motion],axis=-1)]
    allparts=[]
    for z in streams:
        n,t,j,c=z.shape
        allparts.extend([z.mean(1).reshape(n,-1),z.std(1).reshape(n,-1),
            np.quantile(z,.1,axis=1).reshape(n,-1),np.quantile(z,.9,axis=1).reshape(n,-1)])
        # Four- and eight-bin descriptors encode complementary time scales.
        for count in (4,8):
            for ids in np.array_split(np.arange(t),count):
                seg=z[:,ids]; allparts.extend([seg.mean(1).reshape(n,-1),seg.std(1).reshape(n,-1)])
    return np.concatenate(allparts,axis=1).astype(np.float32)

def one_fold(Xf,y,groups,tr,te,fold):
    rng=np.random.default_rng(SEED+fold); ug=np.unique(groups[tr]); rng.shuffle(ug); nv=max(1,round(.18*len(ug))); vg=ug[:nv]
    va=tr[np.isin(groups[tr],vg)]; fit=tr[~np.isin(groups[tr],vg)]
    mu=Xf[fit].mean(0); sd=Xf[fit].std(0); sd[sd<1e-5]=1.; Z=np.clip((Xf-mu)/sd,-8,8).astype(np.float32)
    torch.manual_seed(SEED+fold); model=MLP(Z.shape[1]); cnt=np.bincount(y[fit],minlength=10).astype(np.float32)
    w=np.zeros(10,np.float32); ok=cnt>0; w[ok]=cnt[ok].sum()/(ok.sum()*cnt[ok]); w=torch.tensor(w)
    opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-4); best=-1; state=None; wait=0; be=0
    for epoch in range(1,EPOCHS+1):
        model.train(); order=rng.permutation(fit)
        for st in range(0,len(order),BATCH):
            ids=order[st:st+BATCH]; xb=torch.tensor(Z[ids]); xb=xb+torch.randn_like(xb)*.015
            loss=F.cross_entropy(model(xb),torch.tensor(y[ids]),weight=w)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        model.eval()
        with torch.no_grad(): pv=model(torch.tensor(Z[va])).argmax(1).numpy()
        _,f=metric(y[va],pv)
        if f>best+1e-5:
            best=f; state={k:v.detach().clone() for k,v in model.state_dict().items()}; wait=0; be=epoch
        else:
            wait+=1
            if wait>=PATIENCE: break
    model.load_state_dict(state); model.eval()
    with torch.no_grad(): pred=model(torch.tensor(Z[te])).argmax(1).numpy()
    acc,f=metric(y[te],pred)
    return {'fold':fold,'train_clips':len(fit),'validation_clips':len(va),'test_clips':len(te),'best_epoch':be,'accuracy':acc,'macro_f1':f}

def main():
    with np.load(DATA,allow_pickle=False) as d:
        X=d['X'].astype(np.float32); y=d['y'].astype(np.int64); groups=d['groups'].astype(str)
    Xf=make_features(X); rows=[]
    for fold,(tr,te) in enumerate(grouped_folds(groups),1):
        print(f'fold={fold} d={Xf.shape[1]}',flush=True); rows.append(one_fold(Xf,y,groups,tr,te,fold))
    OUT_CSV.parent.mkdir(parents=True,exist_ok=True)
    with OUT_CSV.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary={k:{'mean':float(np.mean([r[k] for r in rows])),'sd':float(np.std([r[k] for r in rows],ddof=1))} for k in ('accuracy','macro_f1')}
    result={'study':'MMASD joint/bone/motion multiscale temporal descriptor experiment','status':'completed adapted local-release experiment',
        'data':{'clips':len(y),'subjects':len(np.unique(groups)),'classes':len(CLASSES),'feature_dimension':int(Xf.shape[1]),
          'note':'Local archive differs in count and class distribution from published 1,202-clip ten-class setup.'},
        'protocol':{'outer':'same five filename-subject grouped folds','validation':'18% subject-grouped within outer training set','seed':SEED},
        'features':'Global moments and 10th/90th percentiles plus 4-bin and 8-bin means/stds for joint position/motion and directed bone/bone-motion streams.',
        'model':'Class-weighted 512-256 MLP; AdamW; validation macro-F1 early stopping; one seed.',
        'reference':{'single_joint_stream_accuracy':0.7016943452206361,'single_joint_stream_macro_f1':0.6338083599102715},
        'summary':summary,'fold_results':rows}
    OUT_JSON.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'summary':summary,'json':str(OUT_JSON)},indent=2),flush=True)

if __name__=='__main__': main()
