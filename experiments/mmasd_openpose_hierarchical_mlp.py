"""Theme-supervised hierarchical multi-task MLP on local MMASD 2D clips."""
from __future__ import annotations
import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/".codex-deps")); sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from experiments.mmasd_openpose_temporal_features import grouped_folds,metric,features

DATA=ROOT/"data/processed/mmasd_openpose/mmasd_openpose_64frames.npz"
OUT_JSON=ROOT/"results/mmasd_openpose_hierarchical_mlp_2026_11_12.json"
OUT_CSV=ROOT/"results/mmasd_openpose_hierarchical_mlp_folds_2026_11_12.csv"
CLASSES=['as','bs','ce','dr','mfs','ms','sac','sq','tr','tw']
THEME_NAMES=['robotic','music','yoga']
THEME_BY_CLASS=np.asarray([0,0,0,1,1,1,1,0,2,2],dtype=np.int64)
SEED=42; BATCH=64; EPOCHS=100; PATIENCE=12

class HierMLP(nn.Module):
    def __init__(self,d):
        super().__init__(); self.trunk=nn.Sequential(nn.Linear(d,512),nn.ReLU(),nn.BatchNorm1d(512),nn.Dropout(.4),
            nn.Linear(512,256),nn.ReLU(),nn.BatchNorm1d(256),nn.Dropout(.35))
        self.activity=nn.Linear(256,10); self.theme=nn.Linear(256,3)
    def forward(self,x):
        h=self.trunk(x); return self.activity(h),self.theme(h)

def weighted(y,classes):
    counts=np.bincount(y,minlength=classes).astype(np.float32); w=np.zeros(classes,np.float32); ok=counts>0
    w[ok]=counts[ok].sum()/(ok.sum()*counts[ok]); return torch.tensor(w)

def fitfold(Xf,y,groups,tr,te,fold):
    rng=np.random.default_rng(SEED+fold); ug=np.unique(groups[tr]); rng.shuffle(ug)
    nv=max(1,round(.18*len(ug))); vg=ug[:nv]
    va=tr[np.isin(groups[tr],vg)]; fit=tr[~np.isin(groups[tr],vg)]
    mu=Xf[fit].mean(0); sd=Xf[fit].std(0); sd[sd<1e-5]=1.; Z=np.clip((Xf-mu)/sd,-8,8).astype(np.float32)
    torch.manual_seed(SEED+fold); model=HierMLP(Z.shape[1])
    yw=weighted(y[fit],10); theme_y=THEME_BY_CLASS[y]
    tw=weighted(theme_y[fit],3); opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-4)
    best=-1; state=None; wait=0; best_epoch=0
    for epoch in range(1,EPOCHS+1):
        model.train(); order=rng.permutation(fit)
        for st in range(0,len(order),BATCH):
            ids=order[st:st+BATCH]; xb=torch.tensor(Z[ids]); xb=xb+torch.randn_like(xb)*.015
            al,tl=model(xb)
            loss=F.cross_entropy(al,torch.tensor(y[ids]),weight=yw)+.25*F.cross_entropy(tl,torch.tensor(theme_y[ids]),weight=tw)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        model.eval()
        with torch.no_grad(): ap,_=model(torch.tensor(Z[va])); pred=ap.argmax(1).numpy()
        _,f=metric(y[va],pred)
        if f>best+1e-5:
            best=f; state={k:v.detach().clone() for k,v in model.state_dict().items()}; wait=0; best_epoch=epoch
        else:
            wait+=1
            if wait>=PATIENCE: break
    model.load_state_dict(state); model.eval()
    with torch.no_grad(): ap,tp=model(torch.tensor(Z[te])); pred=ap.argmax(1).numpy(); theme_pred=tp.argmax(1).numpy()
    acc,f=metric(y[te],pred); tacc=float(np.mean(theme_y[te]==theme_pred))
    return {'fold':fold,'train_clips':len(fit),'validation_clips':len(va),'test_clips':len(te),
        'best_epoch':best_epoch,'activity_accuracy':acc,'activity_macro_f1':f,'theme_accuracy':tacc}

def main():
    with np.load(DATA,allow_pickle=False) as d:
        X=d['X'].astype(np.float32); y=d['y'].astype(np.int64); groups=d['groups'].astype(str)
    Xf=features(X); rows=[]
    for fold,(tr,te) in enumerate(grouped_folds(groups),1):
        print(f'fold={fold}',flush=True); rows.append(fitfold(Xf,y,groups,tr,te,fold))
    OUT_CSV.parent.mkdir(parents=True,exist_ok=True)
    with OUT_CSV.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary={k:{'mean':float(np.mean([r[k] for r in rows])),'sd':float(np.std([r[k] for r in rows],ddof=1))} for k in ('activity_accuracy','activity_macro_f1','theme_accuracy')}
    out={'study':'MMASD therapy-theme auxiliary supervision','status':'completed one-seed adapted local-release experiment',
        'dataset':{'clips':len(y),'subjects':len(np.unique(groups)),'classes':CLASSES,'themes':THEME_NAMES,
        'mapping':{'robotic':['as','bs','ce','sq'],'music':['dr','mfs','ms','sac'],'yoga':['tr','tw']},
            'note':'Counts and class distribution in the local OpenPose archive differ from the published 1,202-clip configuration.'},
        'protocol':{'outer':'same five filename-subject grouped folds as the local MLP reference','validation':'same 18% grouped inner validation rule','seed':SEED},
        'method':{'model':'Shared MLP trunk with activity and 3-theme heads','loss':'weighted activity cross entropy + 0.25 x weighted theme cross entropy','baseline':'plain temporal-summary MLP, 70.17% mean accuracy and 0.6338 mean macro-F1'},
        'summary':summary,'fold_results':rows}
    OUT_JSON.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({'summary':summary,'json':str(OUT_JSON)},indent=2),flush=True)

if __name__=='__main__': main()
