"""Nested grouped RBF kernel-ridge reference on MMASD OpenPose descriptors."""
from __future__ import annotations
import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/".codex-deps")); sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.mmasd_openpose_temporal_features import grouped_folds,metric,features

DATA=ROOT/"data/processed/mmasd_openpose/mmasd_openpose_64frames.npz"
OUT_JSON=ROOT/"results/mmasd_openpose_rbf_prior_adjust_2026_11_12.json"
OUT_CSV=ROOT/"results/mmasd_openpose_rbf_prior_adjust_folds_2026_11_12.csv"
OUT_OOF=ROOT/"results/mmasd_openpose_rbf_prior_adjust_oof_2026_11_12.csv"
SEED=42; GAMMAS=[.025,.05,.1,.2,.4,.8]; ALPHAS=[.001,.01,.1,1.,10.]
WEIGHT_POWERS=[0.,.5,1.]; PRIOR_OFFSETS=[0.,.25,.5,1.]

def normfit(a,b,idx):
    mu=a[idx].mean(0); sd=a[idx].std(0); sd[sd<1e-5]=1.
    return np.clip((a-mu)/sd,-8,8),mu,sd

def kernel(a,b,gamma):
    aa=np.sum(a*a,axis=1)[:,None]; bb=np.sum(b*b,axis=1)[None,:]
    d2=np.maximum(aa+bb-2*a@b.T,0)/a.shape[1]
    return np.exp(-gamma*d2)

def ridge_fit(train_x,train_y,gamma,alpha,weight_power):
    k=kernel(train_x,train_x,gamma)
    one=np.eye(10,dtype=np.float64)[train_y]
    # Balanced class targets prevent the majority class from dominating the fit.
    counts=np.bincount(train_y,minlength=10).astype(np.float64)
    weights=1/np.maximum(counts[train_y],1)**weight_power
    mat=k+np.diag(alpha/weights)
    coef=np.linalg.solve(mat,one)
    return coef

def main():
    with np.load(DATA,allow_pickle=False) as d:
        X=d['X'].astype(np.float32); y=d['y'].astype(np.int64); groups=d['groups'].astype(str); clip_ids=d['clips'].astype(str)
    Xf=features(X).astype(np.float64); rows=[]; oof=[]
    for fold,(outer_tr,te) in enumerate(grouped_folds(groups),1):
        rng=np.random.default_rng(SEED+fold); ug=np.unique(groups[outer_tr]); rng.shuffle(ug); nv=max(1,round(.18*len(ug))); vg=ug[:nv]
        va=outer_tr[np.isin(groups[outer_tr],vg)]; tr=outer_tr[~np.isin(groups[outer_tr],vg)]
        Z,mu,sd=normfit(Xf,Xf,tr); scores=[]
        class_counts=np.bincount(y[tr],minlength=10).astype(np.float64)
        prior_bias=np.log(class_counts.max()/np.maximum(class_counts,1))
        for gamma in GAMMAS:
            kval=kernel(Z[va],Z[tr],gamma)
            for weight_power in WEIGHT_POWERS:
                for alpha in ALPHAS:
                    coef=ridge_fit(Z[tr],y[tr],gamma,alpha,weight_power)
                    raw=kval@coef
                    for beta in PRIOR_OFFSETS:
                        pred=(raw+beta*prior_bias).argmax(1); acc,f1=metric(y[va],pred)
                        scores.append((f1,acc,gamma,alpha,weight_power,beta))
        best=max(scores,key=lambda x:(x[0],x[1]))
        gamma,alpha,weight_power,beta=best[2],best[3],best[4],best[5]
        # Refit the selected configuration using all outer-training subjects.
        mu=Xf[outer_tr].mean(0); sd=Xf[outer_tr].std(0); sd[sd<1e-5]=1
        Z=np.clip((Xf-mu)/sd,-8,8)
        coef=ridge_fit(Z[outer_tr],y[outer_tr],gamma,alpha,weight_power)
        outer_counts=np.bincount(y[outer_tr],minlength=10).astype(np.float64)
        outer_bias=np.log(outer_counts.max()/np.maximum(outer_counts,1))
        pred=(kernel(Z[te],Z[outer_tr],gamma)@coef+beta*outer_bias).argmax(1)
        acc,f1=metric(y[te],pred)
        rows.append({'fold':fold,'train_clips':len(outer_tr),'validation_clips':len(va),'test_clips':len(te),
            'selected_gamma':gamma,'selected_ridge_alpha':alpha,'selected_class_weight_power':weight_power,
            'selected_prior_offset':beta,'validation_accuracy':best[1],
            'validation_macro_f1':best[0],'accuracy':acc,'macro_f1':f1})
        oof.extend({'fold':fold,'clip_id':str(clip_ids[idx]),'subject_id':str(groups[idx]),
                    'true_label':int(y[idx]),'predicted_label':int(pred[j])} for j,idx in enumerate(te))
        print(f'fold={fold} gamma={gamma} alpha={alpha} acc={acc:.4f} f1={f1:.4f}',flush=True)
    OUT_CSV.parent.mkdir(parents=True,exist_ok=True)
    with OUT_CSV.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with OUT_OOF.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(oof[0])); w.writeheader(); w.writerows(oof)
    summary={k:{'mean':float(np.mean([r[k] for r in rows])),'sd':float(np.std([r[k] for r in rows],ddof=1))} for k in ('accuracy','macro_f1')}
    result={'study':'MMASD RBF kernel-ridge with inner-fold class-prior offsets','status':'completed negative ablation on the local release',
        'dataset':{'clips':len(y),'subjects':len(np.unique(groups)),'classes':10,'archive_note':'Local counts differ from the published 1,202-clip setup.'},
        'protocol':{'outer':'five subject-grouped folds','inner':'18% grouped validation from outer training; gamma/ridge selected on validation only; selected model refit on all outer training clips','seed':SEED},
        'search':{'gamma':GAMMAS,'ridge_alpha':ALPHAS,'class_weight_power':WEIGHT_POWERS,'log_prior_offset':PRIOR_OFFSETS},'features':'Same joint/motion temporal summary vector as the MLP reference.',
        'reference':{'temporal_mlp_accuracy':0.7016943452206361,'temporal_mlp_macro_f1':0.6338083599102715},
        'summary':summary,'fold_results':rows,
        'pooled_oof':{'accuracy':float(np.mean([r['true_label']==r['predicted_label'] for r in oof])),
            'macro_f1':float(metric(np.asarray([r['true_label'] for r in oof]),np.asarray([r['predicted_label'] for r in oof]))[1])},
        'oof_file':str(OUT_OOF.relative_to(ROOT))}
    OUT_JSON.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'summary':summary,'json':str(OUT_JSON)},indent=2),flush=True)

if __name__=='__main__': main()
