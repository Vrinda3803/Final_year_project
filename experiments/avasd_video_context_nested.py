"""Nested, video-grouped AV-ASD 3-class audio experiment.

Tests whether within-video acoustic context improves segment classification.
The context uses only audio descriptors and segment order (no labels), and all
segments from a video remain in one outer fold. This is an offline full-video
setting, not a streaming or multimodal evaluation.
"""
from __future__ import annotations
import ast, csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.codex-deps'))
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BUNDLE = Path(r'D:\Downloads\AV_ASD_audio_stage1_audit_bundle (1)')
FEATURES = BUNDLE/'features'/'segment_features.npy'
FEATURE_IDS = BUNDLE/'features'/'segment_features_row_ids.csv'
MANIFEST = BUNDLE/'reports'/'final_segment_manifest.csv'
CLS = ['Aggressive Behavior','Hyper- or Hyporeactivity to Sensory Input','Upper Limb Stereotypies']
PRIORITY = ['Aggressive Behavior','Self-Spinning or Spinning Objects','Upper Limb Stereotypies','Hyper- or Hyporeactivity to Sensory Input']
OUT = ROOT/'results'/'avasd_video_context_nested_2026_11_12.json'
FOLDS = ROOT/'results'/'avasd_video_context_nested_folds_2026_11_12.csv'
OOF = ROOT/'results'/'avasd_video_context_nested_oof_2026_11_12.csv'
SEED=42

def parse_labels(row):
    ans={}
    for name in PRIORITY:
        try:
            v=ast.literal_eval(row.get('annotation__'+name,'[]'))
            ans[name]=int(v[0]['value']) if v else 0
        except (ValueError,TypeError,SyntaxError,KeyError,IndexError): ans[name]=0
    return ans

def load_data():
    raw=np.load(FEATURES).astype(np.float32)
    with FEATURE_IDS.open(newline='',encoding='utf-8-sig') as f: ids=list(csv.DictReader(f))
    index={r['segment_id']:i for i,r in enumerate(ids)}
    rows=[]
    with MANIFEST.open(newline='',encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r.get('feature_row_available')!='True' or int(r.get('annotation_match_count','0') or 0)==0: continue
            sid=r.get('segment_id','')
            if sid not in index: continue
            labels=parse_labels(r)
            label=next((x for x in PRIORITY if labels.get(x,0)),None)
            if label not in CLS: continue
            try: start=float(r.get('start_sec') or 0); duration=float(r.get('duration_sec') or 0)
            except ValueError: start,duration=0.,0.
            rows.append({'x':raw[index[sid]],'y':CLS.index(label),'group':r.get('video_id') or sid,'start':start,'duration':duration,'id':sid})
    X=np.stack([r['x'] for r in rows]); y=np.array([r['y'] for r in rows]); groups=np.array([r['group'] for r in rows])
    # Build context from each video independently. This uses no labels. For each
    # segment: previous/current/next audio descriptors, video mean/std, position,
    # segment duration, and log video segment count.
    by_group=defaultdict(list)
    for i,r in enumerate(rows): by_group[r['group']].append(i)
    context=np.zeros((len(rows), 3*X.shape[1]+2*X.shape[1]+3),dtype=np.float32)
    for g,ix in by_group.items():
        ix=sorted(ix,key=lambda i:(rows[i]['start'],rows[i]['id']))
        xx=X[ix]; mean=xx.mean(0); std=xx.std(0); n=len(ix)
        for pos,i in enumerate(ix):
            prev=xx[pos-1] if pos else mean
            nxt=xx[pos+1] if pos+1<n else mean
            position=pos/max(1,n-1)
            context[i]=np.concatenate([prev,xx[pos],nxt,mean,std,[position,np.log1p(n),rows[i]['duration']]])
    return rows,X,context,y,groups

def candidates():
    # Small, predeclared search. Select only from grouped inner-fold accuracy.
    c=[]
    for mode in ['base','context']:
        for C in [.1,1.,10.]:
            c.append((f'{mode}_rbf_svc_C{C:g}',mode,lambda C=C:make_pipeline(StandardScaler(),SVC(C=C,kernel='rbf',gamma='scale',class_weight='balanced'))))
    for mode in ['base','context']:
        for leaf in [1,3]:
            c.append((f'{mode}_extratrees_leaf{leaf}',mode,lambda leaf=leaf:ExtraTreesClassifier(n_estimators=500,max_features='sqrt',min_samples_leaf=leaf,class_weight='balanced',n_jobs=-1,random_state=SEED)))
    return c

def main():
    rows,X,XC,y,groups=load_data()
    Xs={'base':np.column_stack([X,np.array([r['duration'] for r in rows])]),'context':XC}
    outer=GroupKFold(n_splits=5); cand=candidates(); oof=np.full(len(y),-1,dtype=int); fold_rows=[]
    for fold,(tr,te) in enumerate(outer.split(X,y,groups),1):
        inner=GroupKFold(n_splits=3); scores=[]
        for name,mode,factory in cand:
            vals=[]; fvals=[]
            for it,iv in inner.split(Xs[mode][tr],y[tr],groups[tr]):
                model=factory(); model.fit(Xs[mode][tr[it]],y[tr[it]])
                pred=model.predict(Xs[mode][tr[iv]])
                vals.append(accuracy_score(y[tr[iv]],pred))
                fvals.append(f1_score(y[tr[iv]],pred,average='macro',zero_division=0))
            scores.append((float(np.mean(vals)),float(np.mean(fvals)),name,mode,factory,float(np.std(vals,ddof=1))))
        # Accuracy is the stated target; macro-F1 breaks accuracy ties.
        scores.sort(key=lambda z:(z[0],z[1]),reverse=True)
        inner_acc,inner_macro,name,mode,factory,inner_sd=scores[0]
        model=factory(); model.fit(Xs[mode][tr],y[tr]); pred=model.predict(Xs[mode][te]);oof[te]=pred
        fold_rows.append({'fold':fold,'train_segments':len(tr),'test_segments':len(te),'train_videos':len(np.unique(groups[tr])),'test_videos':len(np.unique(groups[te])),'selected_model':name,'selected_feature_set':mode,'inner_accuracy_mean':inner_acc,'inner_accuracy_sd':inner_sd,'inner_macro_f1':inner_macro,'test_accuracy':accuracy_score(y[te],pred),'test_macro_f1':f1_score(y[te],pred,average='macro',zero_division=0),'test_balanced_accuracy':balanced_accuracy_score(y[te],pred)})
        print(f"fold={fold} selected={name} test_acc={fold_rows[-1]['test_accuracy']:.4f} test_macroF1={fold_rows[-1]['test_macro_f1']:.4f}",flush=True)
    p,r,f,s=precision_recall_fscore_support(y,oof,labels=[0,1,2],zero_division=0)
    group_audit=[]
    for group in np.unique(groups):
        ix=np.where(groups==group)[0]
        group_audit.append({'video_id':str(group),'segments':int(len(ix)),
            'accuracy':float(accuracy_score(y[ix],oof[ix])),
            'macro_f1':float(f1_score(y[ix],oof[ix],average='macro',zero_division=0)),
            'true_class_counts':dict(Counter(CLS[int(v)] for v in y[ix]))})
    group_audit.sort(key=lambda z:z['segments'],reverse=True)
    summary={'study':'AV-ASD 3-class segment-level audio classification with within-video acoustic context','status':'completed nested grouped experiment','research_basis':{'task_paper':'Hear Me, See Me, Understand Me: Audio-Visual Autism Behavior Recognition, IEEE TMM 2025','method_idea':'Use audio/video behavior context; local adaptation tests temporal and recording context from available audio only.'},'data':{'segments':len(y),'videos':len(np.unique(groups)),'classes':dict(zip(CLS,[int((y==i).sum()) for i in range(3)])),'feature_dim_base':int(Xs['base'].shape[1]),'feature_dim_context':int(XC.shape[1]),'grouping':'video_id'},'method':{'feature_context':'previous/current/next audio descriptors, within-video mean and standard deviation, normalized segment position, log segment count, duration','model_candidates':[x[0] for x in cand],'selection':'3-fold GroupKFold within each outer-training set; mean validation accuracy selected, macro-F1 used only as tie-break','evaluation':'5-fold outer GroupKFold by video_id; each video occurs in one outer fold','note':'Offline full-video setting: unlabeled audio summaries from each held-out video are available when constructing its segment features.'},'headline':{'oof_accuracy':float(accuracy_score(y,oof)),'oof_macro_f1':float(f1_score(y,oof,average='macro',zero_division=0)),'oof_balanced_accuracy':float(balanced_accuracy_score(y,oof)),'majority_class_accuracy':float(Counter(y).most_common(1)[0][1]/len(y)),'target_80_accuracy_reached':bool(accuracy_score(y,oof)>=.80)},'per_class':{CLS[i]:{'precision':float(p[i]),'recall':float(r[i]),'f1':float(f[i]),'support':int(s[i])} for i in range(3)},'folds':fold_rows,'limitations':['Three-class labels are generated by a fixed priority rule from overlapping annotations.','The dominant class has 700 of 961 examples; accuracy must be read with macro-F1 and balanced accuracy.','Grouping is by video, not verified participant.','Features are acoustic summaries; the method is not multimodal and does not verify the child as the speaker.','The inner model selection targets accuracy; this is suitable for testing the stated 80% accuracy goal but macro-F1 is reported to expose class imbalance.'],'files':{'script':'experiments/avasd_video_context_nested.py','folds':str(FOLDS.relative_to(ROOT)),'oof':str(OOF.relative_to(ROOT))}}
    largest=group_audit[0]
    summary['data']['largest_video_segment_count']=largest['segments']
    summary['data']['largest_video_fraction']=largest['segments']/len(y)
    summary['data']['outer_fold_test_segment_counts']=[z['test_segments'] for z in fold_rows]
    summary['headline']['video_macro_mean_segment_accuracy']=float(np.mean([z['accuracy'] for z in group_audit]))
    summary['headline']['video_macro_mean_macro_f1']=float(np.mean([z['macro_f1'] for z in group_audit]))
    summary['headline']['accuracy_without_largest_video']=float(accuracy_score(y[groups!=largest['video_id']],oof[groups!=largest['video_id']]))
    summary['files']['per_video']='results/avasd_video_context_nested_per_video_2026_11_12.csv'
    OUT.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with FOLDS.open('w',newline='',encoding='utf-8') as f0:
        w=csv.DictWriter(f0,fieldnames=list(fold_rows[0]));w.writeheader();w.writerows(fold_rows)
    with OOF.open('w',newline='',encoding='utf-8') as f0:
        w=csv.writer(f0);w.writerow(['segment_id','video_id','true_class','predicted_class','fold'])
        fold_of={int(i):fold for fold,(_,te) in enumerate(outer.split(X,y,groups),1) for i in te}
        for i,row in enumerate(rows):w.writerow([row['id'],row['group'],CLS[y[i]],CLS[oof[i]],fold_of[i]])
    with (ROOT/'results'/'avasd_video_context_nested_per_video_2026_11_12.csv').open('w',newline='',encoding='utf-8') as f0:
        w=csv.DictWriter(f0,fieldnames=['video_id','segments','accuracy','macro_f1','true_class_counts']);w.writeheader();w.writerows(group_audit)
    print(json.dumps({'headline':summary['headline'],'per_class':summary['per_class'],'folds':fold_rows,'json':str(OUT)},indent=2),flush=True)
if __name__=='__main__': main()
