"""Nested subject-grouped Random Forest / ExtraTrees on MMASD 2D descriptors."""
from __future__ import annotations
import csv,json,sys,warnings
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/".codex-deps")); sys.path.insert(0,str(ROOT))
import numpy as np
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.metrics import accuracy_score,f1_score,precision_recall_fscore_support,confusion_matrix
from experiments.mmasd_openpose_temporal_features import grouped_folds,features

DATA=ROOT/"data/processed/mmasd_openpose/mmasd_openpose_64frames.npz"
OUT_JSON=ROOT/"results/mmasd_openpose_nested_trees_2026_11_12.json"
OUT_CSV=ROOT/"results/mmasd_openpose_nested_trees_folds_2026_11_12.csv"
OUT_OOF=ROOT/"results/mmasd_openpose_nested_trees_oof_2026_11_12.csv"
CLASSES=['as','bs','ce','dr','mfs','ms','sac','sq','tr','tw']
SEED=42; N_TREES=300; K_OPTIONS=('all',800); LEAF_OPTIONS=(1,2)

def model_of(name,leaf,seed):
    cls=RandomForestClassifier if name=='RandomForest' else ExtraTreesClassifier
    return cls(n_estimators=N_TREES,max_features='sqrt',min_samples_leaf=leaf,
        class_weight='balanced',n_jobs=-1,random_state=seed)

def fit_variant(name,k,leaf,fit_x,fit_y,val_x,val_y,seed):
    selector=None
    if k!='all':
        kk=min(int(k),fit_x.shape[1])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            selector=SelectKBest(f_classif,k=kk).fit(fit_x,fit_y)
        a=selector.transform(fit_x); b=selector.transform(val_x)
    else: a,b=fit_x,val_x
    model=model_of(name,leaf,seed).fit(a,fit_y)
    p=model.predict(b)
    return float(f1_score(val_y,p,average='macro',zero_division=0)),selector

def main():
    with np.load(DATA,allow_pickle=False) as d:
        X=d['X'].astype(np.float32); y=d['y'].astype(np.int64); groups=d['groups'].astype(str); clips=d['clips'].astype(str)
    XF=features(X).astype(np.float32); rows=[]; oof=[]
    for fold,(outer_tr,te) in enumerate(grouped_folds(groups),1):
        rng=np.random.default_rng(SEED+fold); ug=np.unique(groups[outer_tr]); rng.shuffle(ug); nv=max(1,round(.18*len(ug))); vg=ug[:nv]
        va=outer_tr[np.isin(groups[outer_tr],vg)]; tr=outer_tr[~np.isin(groups[outer_tr],vg)]
        candidates=[]
        for name in ('RandomForest','ExtraTrees'):
            for k in K_OPTIONS:
                for leaf in LEAF_OPTIONS:
                    f1,_=fit_variant(name,k,leaf,XF[tr],y[tr],XF[va],y[va],SEED+fold)
                    candidates.append((f1,name,k,leaf))
                    print(f'fold={fold} {name} k={k} leaf={leaf} val_macro_f1={f1:.4f}',flush=True)
        best=max(candidates,key=lambda z:z[0]); _,name,k,leaf=best
        selector=None; train_x=XF[outer_tr]; test_x=XF[te]
        if k!='all':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                selector=SelectKBest(f_classif,k=min(int(k),train_x.shape[1])).fit(train_x,y[outer_tr])
            train_x=selector.transform(train_x); test_x=selector.transform(test_x)
        model=model_of(name,leaf,SEED+fold).fit(train_x,y[outer_tr])
        pred=model.predict(test_x); probs=model.predict_proba(test_x)
        acc=float(accuracy_score(y[te],pred)); f1=float(f1_score(y[te],pred,average='macro',zero_division=0))
        prec,rec,cf1,_=precision_recall_fscore_support(y[te],pred,labels=np.arange(10),zero_division=0)
        rows.append({'fold':fold,'outer_train_clips':len(outer_tr),'inner_fit_clips':len(tr),'inner_validation_clips':len(va),
            'test_clips':len(te),'selected_model':name,'selected_k':k,'selected_min_leaf':leaf,'validation_macro_f1':best[0],
            'accuracy':acc,'macro_f1':f1,'macro_precision':float(np.mean(prec)),'macro_recall':float(np.mean(rec))})
        for j,idx in enumerate(te):
            row={'fold':fold,'clip_id':clips[idx],'subject_id':groups[idx],'true_label':CLASSES[y[idx]],'predicted_label':CLASSES[pred[j]]}
            row.update({f'prob_{CLASSES[c]}':float(probs[j,c]) for c in range(len(CLASSES))}); oof.append(row)
        print(f'fold={fold} selected={name}/{k}/leaf{leaf} test_acc={acc:.4f} test_f1={f1:.4f}',flush=True)
    OUT_CSV.parent.mkdir(parents=True,exist_ok=True)
    with OUT_CSV.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with OUT_OOF.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(oof[0]));w.writeheader();w.writerows(oof)
    ytrue=np.asarray([CLASSES.index(r['true_label']) for r in oof]); ypred=np.asarray([CLASSES.index(r['predicted_label']) for r in oof])
    pooled={'accuracy':float(accuracy_score(ytrue,ypred)),'macro_f1':float(f1_score(ytrue,ypred,average='macro',zero_division=0)),
        'macro_precision':float(precision_recall_fscore_support(ytrue,ypred,labels=np.arange(10),zero_division=0)[0].mean()),
        'macro_recall':float(precision_recall_fscore_support(ytrue,ypred,labels=np.arange(10),zero_division=0)[1].mean()),
        'confusion_matrix':confusion_matrix(ytrue,ypred,labels=np.arange(10)).tolist()}
    summary={k:{'mean':float(np.mean([r[k] for r in rows])),'sd':float(np.std([r[k] for r in rows],ddof=1))} for k in ('accuracy','macro_f1','macro_precision','macro_recall')}
    per=precision_recall_fscore_support(ytrue,ypred,labels=np.arange(10),zero_division=0)
    perclass={CLASSES[i]:{'precision':float(per[0][i]),'recall':float(per[1][i]),'f1':float(per[2][i]),'support':int(per[3][i])} for i in range(10)}
    result={'study':'MMASD local 2D nested grouped tree baseline','status':'completed adapted local-release experiment; not matched to published clip selection',
      'dataset':{'clips':len(y),'subjects':len(np.unique(groups)),'classes':CLASSES,'class_counts':dict(zip(CLASSES,np.bincount(y,minlength=10).tolist())),
       'note':'Local archive has 1,374 ten-class clips and 343 drumming examples; published 2D configuration has 1,202 clips and 168 drumming examples. No unverified filtering used.'},
      'protocol':{'outer':'same five filename-subject grouped folds','inner':'18% of outer-training subjects used for model selection; selected model refit on all outer-training subjects','seed':SEED},
      'features':'Joint coordinates/motion global moments and quantiles plus eight-bin temporal summaries.',
      'search':{'models':['RandomForest','ExtraTrees'],'n_estimators':N_TREES,'max_features':'sqrt','min_samples_leaf':LEAF_OPTIONS,'feature_count':K_OPTIONS,'class_weight':'balanced'},
      'metrics':{'fold_mean_sd':summary,'pooled_oof':pooled,'per_class_pooled_oof':perclass},'fold_results':rows,
      'references':{'plain_temporal_mlp_accuracy':0.7016943452206361,'plain_temporal_mlp_macro_f1':0.6338083599102715,
        'nested_rbf_accuracy':0.7168510089119099,'nested_rbf_macro_f1':0.6393536733504924}}
    OUT_JSON.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'summary':summary,'pooled_oof':pooled,'per_class':perclass,'json':str(OUT_JSON),'oof':str(OUT_OOF)},indent=2),flush=True)

if __name__=='__main__': main()
