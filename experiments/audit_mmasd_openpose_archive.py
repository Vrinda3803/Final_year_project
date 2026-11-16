"""Audit local MMASD OpenPose clip-folder counts against published tables."""
from __future__ import annotations
import collections,csv,json,re,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/"data/raw/mmasd/Dataset_FINAL/2D skeleton/2D_openpose.zip"
OUT_JSON=ROOT/"results/mmasd_openpose_archive_audit_2026_11_12.json"
OUT_CSV=ROOT/"results/mmasd_openpose_archive_audit_2026_11_12.csv"
PAPER={"as":105,"bs":119,"ce":114,"dr":168,"mfs":103,"ms":130,"sac":113,"sq":101,"tr":129,"tw":120}
NAMES={"as":"Arm swing","bs":"Body swing","ce":"Chest expansion","dr":"Drumming",
 "mfs":"Maracas forward shaking","ms":"Maracas shaking","sac":"Sing and clap","sq":"Squat","tr":"Tree pose","tw":"Twist pose","fg":"Frog pose"}

def main():
    groups=collections.defaultdict(set)
    with zipfile.ZipFile(ARCHIVE) as z:
        for p in z.namelist():
            if not p.endswith('_keypoints.json'): continue
            name=Path(p).name; m=re.match(r'(.+?)_\d+_keypoints\.json$',name)
            if not m: continue
            code=m.group(1).split('_')[0]
            groups[code].add(p.rsplit('/',1)[0]+'/'+m.group(1))
    codes=sorted(set(groups)|set(PAPER)); rows=[]
    for c in codes:
        local=len(groups.get(c,set())); published=PAPER.get(c)
        rows.append({'class_code':c,'class_name':NAMES.get(c,c),'local_clip_folders':local,
          'published_10class_count':published if published is not None else '',
          'difference_vs_published':local-published if published is not None else ''})
    OUT_CSV.parent.mkdir(parents=True,exist_ok=True)
    with OUT_CSV.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    n10=sum(len(groups.get(c,set())) for c in PAPER); nall=sum(len(v) for v in groups.values())
    out={'study':'Local MMASD OpenPose archive count audit','archive':str(ARCHIVE.relative_to(ROOT)),
      'method':'Count distinct parent clip folders for frame JSON files using filename prefix before frame number; no activity filtering.',
      'local_counts':{c:len(groups[c]) for c in sorted(groups)},'local_observed_label_count':len(groups),
      'local_total_counted_clip_folders':nall,'local_ten_class_total':n10,
      'published_dataset_total_all_classes':1315,'published_2d_ten_class_total':sum(PAPER.values()),
      'ten_class_total_difference':n10-sum(PAPER.values()),'main_discrepancy':{'drumming_local':len(groups.get('dr',set())),
       'drumming_published':PAPER['dr'],'difference':len(groups.get('dr',set()))-PAPER['dr']},
      'interpretation':'The local OpenPose archive does not match the published clip inventory. Counts alone do not establish which files are duplicates, excluded samples, or a different release. No records were removed to force agreement.',
      'references':{'dataset_release':'https://github.com/Li-Jicheng/MMASD-A-Multimodal-Dataset-for-Autism-Intervention-Analysis','published_2d_table':'https://www.mdpi.com/1424-8220/26/14/4638'},
      'per_class':rows}
    OUT_JSON.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({'local_total_counted_clip_folders':nall,'local_observed_label_count':len(groups),'local_ten_class_total':n10,
      'published_ten_class_total':sum(PAPER.values()),'classes':out['local_counts'],'json':str(OUT_JSON),'csv':str(OUT_CSV)},indent=2))
if __name__=='__main__': main()
