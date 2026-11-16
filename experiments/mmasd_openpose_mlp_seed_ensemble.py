"""Three-seed probability ensemble for the local MMASD 2D temporal MLP."""
from __future__ import annotations
import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/".codex-deps")); sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from experiments.mmasd_openpose_temporal_features import grouped_folds,metric,features,MLP

DATA=ROOT/"data/processed/mmasd_openpose/mmasd_openpose_64frames.npz"
OUT_JSON=ROOT/"results/mmasd_openpose_mlp_seed_ensemble_2026_11_12.json"
OUT_CSV=ROOT/"results/mmasd_openpose_mlp_seed_ensemble_folds_2026_11_12.csv"
SEED=42; N_SEEDS=3; BATCH=64; EPOCHS=100; PATIENCE=12

def fit_seed(Z,y,fit,va,te,seed):
    rng=np.random.default_rng(seed); torch.manual_seed(seed); model=MLP(Z.shape[1])
    counts=np.bincount(y[fit],minlength=10).astype(np.float32); w=np.zeros(10,np.float32); ok=counts>0
    w[ok]=counts[ok].sum()/(ok.sum()*counts[ok]); w=torch.tensor(w)
    opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-4)
    best=-1; state=None; wait=0; best_epoch=0
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
            best=f; state={k:v.detach().clone() for k,v in model.state_dict().items()}; wait=0; best_epoch=epoch
        else:
            wait+=1
            if wait>=PATIENCE: break
    model.load_state_dict(state); model.eval()
    with torch.no_grad(): probs=torch.softmax(model(torch.tensor(Z[te])),dim=1).numpy()
    return probs,best_epoch

def main():
    with np.load(DATA,allow_pickle=False) as d:
        X=d['X'].astype(np.float32); y=d['y'].astype(np.int64); groups=d['groups'].astype(str)
    Xf=features(X); rows=[]
    for fold,(tr,te) in enumerate(grouped_folds(groups),1):
        rng=np.random.default_rng(SEED+fold); ug=np.unique(groups[tr]); rng.shuffle(ug); nv=max(1,round(.18*len(ug))); vg=ug[:nv]
        va=tr[np.isin(groups[tr],vg)]; fit=tr[~np.isin(groups[tr],vg)]
        mu=Xf[fit].mean(0); sd=Xf[fit].std(0); sd[sd<1e-5]=1.; Z=np.clip((Xf-mu)/sd,-8,8).astype(np.float32)
        fold_probs=[]; epochs=[]
        for si in range(N_SEEDS):
            print(f'fold={fold} seed={SEED+100*si+fold}',flush=True)
            p,e=fit_seed(Z,y,fit,va,te,SEED+100*si+fold); fold_probs.append(p); epochs.append(e)
        pred=np.mean(fold_probs,axis=0).argmax(1); acc,f=metric(y[te],pred)
        rows.append({'fold':fold,'train_clips':len(fit),'validation_clips':len(va),'test_clips':len(te),
            'seeds':N_SEEDS,'best_epochs':';'.join(map(str,epochs)),'accuracy':acc,'macro_f1':f})
    OUT_CSV.parent.mkdir(parents=True,exist_ok=True)
    with OUT_CSV.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary={k:{'mean':float(np.mean([r[k] for r in rows])),'sd':float(np.std([r[k] for r in rows],ddof=1))} for k in ('accuracy','macro_f1')}
    result={'study':'MMASD 2D temporal MLP three-seed probability ensemble','status':'completed adapted local-release experiment',
        'dataset':{'clips':len(y),'subjects':len(np.unique(groups)),'classes':10,'archive_note':'Local archive differs from published 1,202-clip configuration.'},
        'protocol':{'outer':'same five filename-subject grouped folds','inner_validation':'18% grouped, train-only','seed_schedule':[SEED,SEED+100,SEED+200]},
        'model':{'architecture':'same temporal summary MLP as single model','ensemble':'mean of three validation-selected seed softmax outputs'},
        'single_seed_reference':{'accuracy_mean':0.7016943452206361,'macro_f1_mean':0.6338083599102715},
        'summary':summary,'fold_results':rows}
    OUT_JSON.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'summary':summary,'json':str(OUT_JSON)},indent=2),flush=True)

if __name__=='__main__': main()
