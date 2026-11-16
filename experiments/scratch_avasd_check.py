"""Quick AV-ASD data analysis"""
import csv, json, ast, numpy as np

rows = list(csv.DictReader(open(r'D:\Downloads\AV_ASD_audio_stage1_audit_bundle (1)\reports\final_segment_manifest.csv')))
print('Total rows:', len(rows))

feat_avail = [r for r in rows if r['feature_row_available']=='True']
print('Has features:', len(feat_avail))
matched = [r for r in feat_avail if int(r['annotation_match_count'])>0]
print('Has features + annotation match:', len(matched))

feat_ids = list(csv.DictReader(open(r'D:\Downloads\AV_ASD_audio_stage1_audit_bundle (1)\features\segment_features_row_ids.csv')))
feat_id_to_idx = {r['segment_id']: i for i, r in enumerate(feat_ids)}
print('Feature IDs in npy:', len(feat_id_to_idx))

in_both = [r for r in matched if r['segment_id'] in feat_id_to_idx]
print('Segments with features AND annotation:', len(in_both))

# Count per-class label distribution
agg_count=0; hyp_count=0; upl_count=0; spi_count=0
for r in in_both:
    try:
        agg = int(ast.literal_eval(r['annotation__Aggressive Behavior'])[0]['value'])
        hyp = int(ast.literal_eval(r['annotation__Hyper- or Hyporeactivity to Sensory Input'])[0]['value'])
        upl = int(ast.literal_eval(r['annotation__Upper Limb Stereotypies'])[0]['value'])
        spi = int(ast.literal_eval(r['annotation__Self-Spinning or Spinning Objects'])[0]['value'])
        agg_count+=agg; hyp_count+=hyp; upl_count+=upl; spi_count+=spi
    except: pass

print(f'Positive label counts: Agg={agg_count} Hyp={hyp_count} Upl={upl_count} Spi={spi_count}')
print(f'Total in_both: {len(in_both)}')

# Check unique videos
vids = set(r['video_id'] for r in in_both)
print(f'Unique videos: {len(vids)}')
