"""AV-ASD audio-only video-level classification with nested stratified CV.

One example is created per annotated source video by pooling aligned acoustic
segment features; its label is the majority of the same priority-rule 3-class
segment labels used in the prior experiment. This prevents a very long video
from contributing hundreds of independent votes. It is a distinct video-level
task, and participant independence is unknown.
"""
from __future__ import annotations
import ast,csv,json,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.codex-deps'))
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

BUNDLE=Path(r'D:\Downloads\AV_ASD_audio_stage1_audit_bundle (1)')
CLS=['Aggressive Behavior','Hyper- or Hyporeactivity to Sensory Input','Upper Limb Stereotypies']
PRIORITY=['Aggressive Behavior','Self-Spinning or Spinning Objects','Upper Limb Stereotypies','Hyper- or Hyporeactivity to Sensory Input']
OUT=ROOT/'results'/'avasd_video_level_nested_2026_11_12.json'
FOLDS=ROOT/'results'/'avasd_video_level_nested_folds_2026_11_12.csv'
OOF=ROOT/'results'/'avasd_video_level_nested_oof_2026_11_12.csv'
SEED=42

def labels_of(r):
 d={}
 for c in PRIORITY:
  try:
   z=ast.literal_eval(r.get('annotation__'+c,'[]'));d[c]=int(z[0]['value']) if z else 0
  except (ValueError,TypeError,SyntaxError,IndexError,KeyError):d[c]=0
 return d

def load():
 feat=np.load(BUNDLE/'features'/'segment_features.npy').astype(np.float32)
 with (BUNDLE/'features'/'segment_features_row_ids.csv').open(newline='',encoding='utf-8-sig') as f:ids=list(csv.DictReader(f))
 idx={r['segment_id']:i for i,r in enumerate(ids)}
 by_video=defaultdict(list)
 with (BUNDLE/'reports'/'final_segment_manifest.csv').open(newline='',encoding='utf-8-sig') as f:
  for r in csv.DictReader(f):
   if r.get('feature_row_available')!='True' or int(r.get('annotation_match_count','0') or 0)==0:continue
   sid=r.get('segment_id','')
   if sid not in idx:continue
   d=labels_of(r);label=next((c for c in PRIORITY if d.get(c,0)),None)
   if label not in CLS:continue
   try:start=float(r.get('start_sec') or 0);dur=float(r.get('duration_sec') or 0)
   except ValueError:start,dur=0.,0.
   by_video[r.get('video_id') or sid].append({'x':feat[idx[sid]],'y':CLS.index(label),'start':start,'duration':dur,'id':sid})
 X=[];y=[];vids=[];purity=[];counts=[]
 for vid,ss in sorted(by_video.items()):
  ss=sorted(ss,key=lambda z:(z['start'],z['id']));xx=np.stack([z['x'] for z in ss]);yy=np.array([z['y'] for z in ss]);bc=np.bincount(yy,minlength=3);target=int(np.argmax(bc))
  chunks=[]
  for chunk in np.array_split(np.arange(len(xx)),4):chunks.append(xx[chunk].mean(0) if len(chunk) else np.zeros(xx.shape[1],dtype=np.float32))
  featvec=np.concatenate([xx.mean(0),xx.std(0),xx.min(0),xx.max(0),*chunks,[np.log1p(len(xx)),np.mean([z['duration'] for z in ss]),np.std([z['duration'] for z in ss])]]).astype(np.float32)
  X.append(featvec);y.append(target);vids.append(vid);purity.append(float(bc[target]/len(yy)));counts.append(bc)
 return np.stack(X),np.array(y),np.array(vids),np.array(purity),np.array(counts)

def candidates():
 out=[]
 for c in [.03,.1,1.,10.]:out.append((f'logreg_C{c:g}',lambda c=c:make_pipeline(StandardScaler(),LogisticRegression(C=c,class_weight='balanced',max_iter=3000,random_state=SEED))))
 for c in [.1,1.,10.]:out.append((f'rbf_svc_C{c:g}',lambda c=c:make_pipeline(StandardScaler(),SVC(C=c,kernel='rbf',gamma='scale',class_weight='balanced'))))
 for leaf in [1,2,4]:out.append((f'extratrees_leaf{leaf}',lambda leaf=leaf:ExtraTreesClassifier(n_estimators=400,max_features='sqrt',min_samples_leaf=leaf,class_weight='balanced',n_jobs=-1,random_state=SEED)))
 for leaf in [1,2,4]:out.append((f'randomforest_leaf{leaf}',lambda leaf=leaf:RandomForestClassifier(n_estimators=400,max_features='sqrt',min_samples_leaf=leaf,class_weight='balanced',n_jobs=-1,random_state=SEED)))
 for k in [3,5,7]:out.append((f'knn_k{k}',lambda k=k:make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=k,weights='distance'))))
 return out

def main():
 X,y,vids,purities,counts=load();cand=candidates();outer=StratifiedKFold(5,shuffle=True,random_state=SEED);pred=np.full(len(y),-1);folds=[]
 for fold,(tr,te) in enumerate(outer.split(X,y),1):
  inner=StratifiedKFold(3,shuffle=True,random_state=SEED+fold);scores=[]
  for name,factory in cand:
   acc=[];mac=[]
   for it,iv in inner.split(X[tr],y[tr]):
    m=factory();m.fit(X[tr[it]],y[tr[it]]);z=m.predict(X[tr[iv]])
    acc.append(accuracy_score(y[tr[iv]],z));mac.append(f1_score(y[tr[iv]],z,average='macro',zero_division=0))
   scores.append((float(np.mean(acc)),float(np.mean(mac)),name,factory,float(np.std(acc,ddof=1))))
  scores.sort(key=lambda z:(z[0],z[1]),reverse=True);ia,im,name,factory,isd=scores[0]
  model=factory();model.fit(X[tr],y[tr]);pred[te]=model.predict(X[te])
  folds.append({'fold':fold,'train_videos':len(tr),'test_videos':len(te),'selected_model':name,'inner_accuracy_mean':ia,'inner_accuracy_sd':isd,'inner_macro_f1':im,'test_accuracy':accuracy_score(y[te],pred[te]),'test_macro_f1':f1_score(y[te],pred[te],average='macro',zero_division=0),'test_balanced_accuracy':balanced_accuracy_score(y[te],pred[te])})
  print(f"fold={fold} model={name} test_acc={folds[-1]['test_accuracy']:.3f} macroF1={folds[-1]['test_macro_f1']:.3f}",flush=True)
 p,r,f,s=precision_recall_fscore_support(y,pred,labels=[0,1,2],zero_division=0)
 majority=Counter(y).most_common(1)[0][1]/len(y)
 result={'study':'AV-ASD audio-only 3-class video-level classification','status':'nested stratified evaluation completed','task_note':'Each source video is one sample; the target is its most frequent 3-class segment label under the existing annotation priority rule. This differs from segment-level recognition.','research_basis':'Motivated by the original AV-ASD multimodal behavior-recognition task; local implementation is audio-only due unavailable aligned video/pose features.','data':{'videos':len(y),'segments_pooled':int(counts.sum()),'classes':{CLS[i]:int(np.sum(y==i)) for i in range(3)},'mean_majority_label_purity':float(purities.mean()),'min_majority_label_purity':float(purities.min()),'feature_dimension':int(X.shape[1]),'grouping':'one sample per video_id; participant identity remains unverified'},'method':{'features':'mean/std/min/max and four ordered temporal-bin means over aligned 32-D audio descriptors, plus segment-count and duration summaries','model_selection':'15 predeclared logistic regression, RBF-SVC, ExtraTrees, RandomForest, and KNN configurations; 3-fold stratified CV within each outer training set, selected by accuracy with macro-F1 tie-break','outer_evaluation':'5-fold StratifiedKFold at video level; fixed random seed 42'},'metrics':{'oof_accuracy':float(accuracy_score(y,pred)),'oof_macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),'oof_balanced_accuracy':float(balanced_accuracy_score(y,pred)),'majority_video_class_baseline':float(majority),'accuracy_80_reached':bool(accuracy_score(y,pred)>=.8)},'per_class':{CLS[i]:{'precision':float(p[i]),'recall':float(r[i]),'f1':float(f[i]),'video_support':int(s[i])} for i in range(3)},'folds':folds,'limitations':['87 videos provide a small sample for model selection.','Video IDs are not confirmed participant IDs; participants may appear in multiple videos.','Audio speaker identity is not verified.','The target is a video-level majority label derived from segment annotations, not a clinical diagnosis.','The method is audio-only; no multimodal superiority claim is possible.'],'files':{'fold_csv':str(FOLDS.relative_to(ROOT)),'oof_csv':str(OOF.relative_to(ROOT))}}
 OUT.write_text(json.dumps(result,indent=2),encoding='utf-8')
 with FOLDS.open('w',newline='',encoding='utf-8') as f0:w=csv.DictWriter(f0,fieldnames=list(folds[0]));w.writeheader();w.writerows(folds)
 with OOF.open('w',newline='',encoding='utf-8') as f0:
  w=csv.writer(f0);w.writerow(['video_id','true_dominant_label','predicted_label','segment_count','label_purity','fold'])
  fold_of={int(i):fold for fold,(_,te) in enumerate(outer.split(X,y),1) for i in te}
  for i in range(len(y)):w.writerow([vids[i],CLS[y[i]],CLS[pred[i]],int(counts[i].sum()),purities[i],fold_of[i]])
 print(json.dumps({'metrics':result['metrics'],'per_class':result['per_class'],'data':result['data'],'folds':folds},indent=2),flush=True)
if __name__=='__main__':main()
