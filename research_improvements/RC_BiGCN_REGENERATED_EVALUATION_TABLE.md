# Regenerated Evaluation Table

## Ten-class MMASD activity recognition

The table below was rebuilt from the saved fold summaries and out-of-fold predictions in the project. All rows use the local 1,374-recording, ten-class MMASD configuration and the same five stored cohort folds. The cohort identifiers are not verified participant identities.

| Model / configuration | Main distinction | Mean fold accuracy (%) | Pooled out-of-fold accuracy (%) | Pooled out-of-fold macro-F1 | ECE |
|---|---|---:|---:|---:|---:|
| A0 — fixed-topology reference | Fixed augmented relational topology; Stage 3 model | 57.12 ± 5.88 | 57.13 | 0.5428 | 0.1370 |
| A3 — RCDB-STGT with reliability gate | Dynamic bilateral edges with an additional multiplicative reliability gate | 60.40 ± 11.65 | 60.48 | 0.5947 | 0.1109 |
| A8 — dynamic edges with reliability inputs | Dynamic bilateral-edge learner receives joint confidence; no additional multiplicative gate | 65.92 ± 6.61 | 65.94 | 0.6402 | 0.0733 |
| RBF kernel-ridge baseline | Nested grouped evaluation on temporal movement descriptors | 71.69 ± 9.10 | 71.69 | 0.6920 | Not reported |
| RC-BiGCN | Root-centred, torso-scaled input; reliability-conditioned bilateral graph; multi-scale temporal convolutions | 75.32 ± 6.38 | 75.33 | 0.7361 | Not reported; not calibrated in the saved run |

Accuracy variation is the population standard deviation across the five test folds. Pooled metrics are calculated over the combined held-out predictions. ECE values are included only where recorded by the corresponding experiment. Stage 4 ECE was calculated after validation-based temperature calibration; the RC-BiGCN run saved no calibrated probabilities or ECE, so its ECE is not directly comparable and is left unreported.

### Interpretation

RC-BiGCN has the highest saved pooled accuracy and macro-F1 among the rows shown. Compared with A8, its pooled accuracy is higher by 9.39 percentage points and its pooled macro-F1 by 0.0959. The RBF reference also provides a non-neural comparison; RC-BiGCN is higher by 3.64 accuracy points and 0.0440 pooled macro-F1 points in these saved results.

These are observed comparisons under the shared stored folds, not isolated estimates of the effect of any one RC-BiGCN component. RC-BiGCN changes the input normalization, input channels, spatial graph implementation, and temporal architecture relative to the Stage 4 models. A matched ablation within the RC-BiGCN backbone is needed to attribute the difference specifically to dynamic bilateral learning, reliability conditioning, or the temporal convolutions.

### Audit notes affecting the comparison

- The saved A1 and A8 configurations use the same model settings, and their saved out-of-fold predictions are identical. A1 is therefore omitted as a duplicate, not treated as an independent ablation.
- RC-BiGCN's training script uses class-weighted cross-entropy. It does not implement the confusable-class margin loss described in the model file's introductory docstring.
- The saved RC-BiGCN result JSON does not report ECE. The value `0.0612` in the benchmark CSV is not supported by the run summary and is not used here.
- The parameter-count claim of 72,240 versus approximately 185,000 is not included: the supplied comparison materials do not establish counts from the exact instantiated models with a consistent counting method.
- The previous A0, A3, and A8 table labels should be described carefully. A0 is a fixed augmented relational topology, not simply an anatomical-only graph. A8's reliability values enter the bilateral edge learner; it is not a reliability-free dynamic-edge model.

### Source artifacts

- `research_improvements/rc_bigcn_results.json` and `research_improvements/rc_bigcn_oof.npz`
- `research_stage4/metrics/a0_fixed_topo_repro_summary.json`
- `research_stage4/metrics/a3_rcdb_full_main_summary.json`
- `research_stage4/metrics/a8_rcdb_input_rel_only_summary.json`
- `results/mmasd_openpose_rbf_kernel_2026_11_12.json` (including its pooled OOF metrics)

