# Autism Project Evaluation Report

**Project:** Explainable graph based temporal modeling of social behaviours for autism assessment  
**Results reviewed:** 12 November 2026

## Executive summary

The strongest reproducible result currently present is MMASD 11-class session recognition at **82.87% accuracy and 0.7912 macro-F1**, using a 5-fold subject-grouped evaluation. AV-ASD results vary by task, aggregation level, and grouping unit: the available four-class clip result is 50.55% accuracy / 0.4191 macro-F1, while a three-class segment experiment reports 76.59% accuracy / 0.5370 macro-F1 grouped by video. Neither reaches 90% on both accuracy and macro-F1. There is no local ASBD/SSBD data or standalone evaluation artifact, so its prior 55.90% / 0.5576 claim cannot be verified and should not be reported as an ASBD result.

Scores are evaluation outcomes, not clinical diagnostic performance. Results on these small and imbalanced datasets require independent replication and uncertainty reporting before clinical interpretation.

## Dataset status and best recorded results

| Dataset / task | Evaluation unit and grouping | Best recorded accuracy | Best recorded macro-F1 | 90% target | Evidence status |
|---|---|---:|---:|---|---|
| MMASD, 11-class activity recognition | Session-level; 5-fold GroupKFold by subject | 82.87% | 0.7912 | Not reached | Dated experiment JSON with fold and class metrics |
| AV-ASD, 4-class clip classification | Clip-level; earlier baseline reports subject GroupKFold | 50.55% | 0.4191 | Not reached | Benchmark JSON; advanced alternatives have different evaluation units and lower macro-F1 |
| AV-ASD, 3-class segment classification | Segment-level; 5-fold GroupKFold by video | 76.59% | 0.5370 | Not reached | Dated hyper-tuning JSON; segments are not independent clips |
| ASBD/SSBD, 3-class motor behaviour | No local data or standalone result | — | — | Not assessable | Current benchmark summary repeats AV-ASD metrics without ASBD evidence |

The AV-ASD figures are separate tasks and must not be merged into one dataset score. The project’s benchmark JSON says all tasks use subject grouping, but the advanced AV-ASD artifacts specify grouping by `video_id`; confirm whether videos from one subject can span folds before describing those results as participant-independent.

## MMASD work completed

The workspace contains processed features for **1,693 sessions from 32 subjects**, with 11 classes and ROMP 3D skeletal inputs. Two feature packages are present: sequence features and 506-dimensional rich features. Implemented work includes temporal summary baselines, nested subject-grouped random forest selection, temporal-bin comparisons, multi-bank temporal feature extraction, prototype-distance features, SelectKBest feature selection, and RF / ExtraTrees / XGBoost ensembles. Graph-based MEAT-GNN pipelines and explanatory dashboard figures are also present.

| Experiment | Method | Accuracy | Macro-F1 | Notes |
|---|---|---:|---:|---|
| Nested RF | Pooled temporal kinematic summaries; nested 5 outer × 3 inner GroupKFold by subject | 81.10% | 0.7716 | Most conservative nested baseline saved |
| Nested RF temporal | Pooled vs four-bin temporal features with inner selection | 80.98% | 0.7688 | Temporal bins did not improve the saved OOF result |
| RF / ET / XGB ensemble | 1,274 feature dimensions; subject GroupKFold | 82.04% | 0.7812 | Earlier ensemble baseline |
| Advanced ensemble | 1,754 dimensions: multi-bank + rich + prototype; SelectKBest 800; RF/ET/XGB | **82.87%** | **0.7912** | Best saved overall; fold accuracy range 79.64–88.60% |

Weakest advanced-model classes are squat (F1 0.560), maracas forward shaking (0.598), and maracas shaking (0.737). Strong classes include twisting (0.947), drumming (0.901), and frog pose (0.905). Per-class values are in `results/class_metrics_2026_11_12.csv`.

### Quality-aware graph pilot

A first 11-class MMASD graph pilot compared supervised training with and without a synthetic observed-body-group mask, and masked-feature self-supervised pretraining plus that mask. At 100% labels on clean inputs, the supervised quality-aware model reached 62.33% mean accuracy / 0.5079 macro-F1 across five stored-group folds, compared with 61.23% / 0.4806 for the same supervised model without the quality signal. Under 40% synthetic body-group masking, self-supervised plus quality-aware training reached 53.95% / 0.4407, versus 40.70% / 0.3228 for the no-quality supervised model. However, it did not beat the clean-input baseline or the project's 82.87% / 0.7912 ensemble. The cache has no actual joint-confidence measurements, so the robustness result is limited to synthetic missing-feature tests. Full label-budget, corruption, calibration, and fold results are saved in the quality-aware result JSON and summary/fold CSV files.

A separate pilot used actual OpenPose confidence values from the local 2D archive. Under five filename-subject-grouped folds, the confidence-blind graph+BiGRU control reached 39.84% ± 8.94 accuracy and 0.3219 ± 0.1068 macro-F1; confidence-aware gating reached 34.88% ± 9.37 and 0.2584 ± 0.1161. This one-seed result does not support a clean-data gain from confidence. The archive contains 1,374 clips and a different class distribution from the published 1,202-clip 2D configuration, so this is neither an exact reproduction nor a comparable score. See `mmasd_openpose_quality_graph_2026_11_12.json` for details and limitations.

A temporal-summary MLP on those same clips and outer folds, with the inner validation groups aligned to the graph run, reached 70.17% ± 10.77 accuracy / 0.6338 ± 0.1903 macro-F1. It is a better local reference than the compact graph-GRU, but its high fold variation and one-seed setup remain limitations. The results support further work on temporal feature design before claiming a useful confidence-aware graph improvement.

Three follow-ups did not yield a clear improvement in macro-F1: confidence-weighted descriptors scored 70.24% ± 8.44 accuracy / 0.6255 ± 0.1929 macro-F1; averaging three MLP seeds scored 70.83% ± 8.98 / 0.6226 ± 0.1903; and an auxiliary therapy-theme loss scored 68.57% ± 11.63 / 0.6080 ± 0.2073. The theme mapping was taken from the dataset README. These are exploratory one-seed comparisons on the unmatched local archive, not evidence that the project exceeds published MMASD work.

Adding explicit bone and bone-motion temporal descriptors reached 70.74% ± 11.19 accuracy / 0.6265 ± 0.2082 macro-F1. A nested RBF kernel-ridge model on the joint/motion temporal summaries reached 71.69% ± 9.10 / 0.6394 ± 0.1918, the strongest local 2D pilot so far. The latter is a conventional classifier and only a modest gain over the plain MLP; it is not a novel contribution. All local 2D values remain below the published 89.36% / 0.8812 subject-independent RC-ProtoGCN result and are not directly comparable because of the unresolved archive-count mismatch.

A nested Random Forest / ExtraTrees comparison with fold-only feature selection reached 68.42% ± 8.34 accuracy / 0.6242 ± 0.1845 macro-F1 by fold mean; pooled OOF scores were 68.41% accuracy / 0.6517 macro-F1. In the RBF run, pooled OOF scores were 71.69% accuracy / 0.6920 macro-F1. The 1,374-clip local archive differs from the 1,202-clip ten-class reference inventory, chiefly because local drumming has 343 clips while the paper lists 168. The audit records the discrepancy without filtering the data; this prevents an apples-to-apples claim.

An additional RBF variant tuned inverse-frequency weighting strength and class-prior offsets inside the validation groups. It decreased to 69.29% ± 11.85 accuracy / 0.6207 ± 0.2265 macro-F1, so the unadjusted RBF configuration remains the stronger result. The added settings are recorded as a negative ablation.

## AV-ASD work completed

AV-ASD experiments use an external audio feature and annotation bundle under Downloads, not the project’s `data` folder. The saved work covers audio-only models, gated audio/pose fusion, annotated segment expansion, engineered audio features, prototype distances, feature selection, and RF / ExtraTrees / XGBoost ensembles. Sample counts differ by task: the four-class clip task has 182 clips / 116 subjects, while the tuned three-class segment task has 961 annotated segments / 87 videos.

| Experiment / unit | Accuracy | Macro-F1 | Key observation |
|---|---:|---:|---|
| Four-class audio XGBoost clip benchmark | 50.55% | 0.4191 | Accuracy slightly exceeds the 47.25% saved audio baseline; macro-F1 is lower than its 0.4290 baseline |
| Audio-enhanced four-class model | 40.66% best individual model | 0.3503 | Does not improve the earlier benchmark |
| Gated multimodal four-class result | 46.70% | 0.4144 | Same recorded score as audio-only in that experiment |
| Advanced four-class segment model | 54.38% | 0.3522 | Segment-level accuracy is not comparable to clip-level score; low macro-F1 |
| Advanced four-class video aggregation | 31.90% | 0.3267 | Weak video-level performance |
| Three-class hyper-tuned segment ensemble | **76.59%** | **0.5370** | 961 segments; video-grouped CV. Class F1: aggressive 0.585, hyper/hyporeactivity 0.168, upper-limb stereotypies 0.858 |
| Three-class video aggregation | 54.02% | 0.5016 | Performance falls after segment predictions are aggregated to video |

An additional multi-label experiment now follows the original AV-ASD task formulation, using the 32-dimensional audio descriptors available in the local audit bundle. It pools temporally aligned segments into 259-dimensional clip features and evaluates 10 annotation indicators with five-fold GroupKFold by source video. Only 227 of 928 annotated clips have usable aligned audio features (138 source-video groups). At a fixed 0.5 threshold, the nine behavior labels achieve 0.2811 macro-F1 and 0.3145 micro-F1; macro average precision is 0.2202. Selecting class-wise thresholds on grouped validation data reduces macro-F1 to 0.2688, so the fixed-threshold result is the primary one. This audio-only result is not comparable to the earlier three-/four-class experiments or to the published multimodal result. Target-speaker identity and participant independence are unverified. See `results/avasd_multilabel_audio_grouped_2026_11_12.json` and its fold/OoF CSVs for details.

The 76.59% segment accuracy has a large class imbalance (700/961 segments are upper-limb stereotypies); the low hyper/hyporeactivity recall (0.145) makes accuracy alone misleading. More importantly, one video contributes 506 of 961 segments and all 506 carry the upper-limb-stereotypy label. That video forms an outer test fold by itself and scores 92.49%; the other four folds together score about 58.9% accuracy. The pooled score is therefore dominated by one unusually large, single-class video and should not be treated as a reliable generalization estimate. Participant identities are not verified.

A nested grouped experiment added within-video audio context (previous/current/next segment features and per-video unlabeled acoustic summaries). On the same three-class segment formulation it achieved 67.01% pooled accuracy, 0.4662 macro-F1, and 0.4838 balanced accuracy. Equal-weighting the 87 videos gives 46.28% mean accuracy. It did not reach 80% and fell below the 72.84% majority-class segment baseline on pooled accuracy. Its test-fold accuracies ranged from 4.39% to 92.49%; the context representation did not resolve the extreme group-size and class-distribution problem. Details are in `results/avasd_video_context_nested_2026_11_12.json` and the accompanying fold, OOF, and per-video CSVs.

A separate video-level experiment pools each recording to one sample and predicts its majority segment label, so the 506-segment recording counts once. On 87 videos, nested stratified evaluation achieved 60.92% accuracy, 0.5492 macro-F1, and 0.5527 balanced accuracy, compared with a 52.87% majority-video baseline. Fold accuracy ranged from 50.00% to 76.47%, so the result remains below 80% and varies across folds. This is a distinct video-level target; video IDs are not verified participant IDs. See `results/avasd_video_level_nested_2026_11_12.json` and its fold/OOF tables.

## ASBD / SSBD work and evidence gap

The benchmark summary lists arm flapping, head banging/self-hitting, and spinning, and compares 55.90% accuracy / 0.5576 macro-F1 against a notebook baseline. However, the workspace has no ASBD/SSBD source data, notebook, model artifact, or result JSON supporting that evaluation. The same 55.90% / 0.5576 values also appear for the AV-ASD three-class task. The claimed improvement is therefore an unverified copied value, not a validated cross-dataset transfer result. Mark this dataset **pending data and reproducible evaluation** until its source, subject IDs, split, code, and outputs are added.

## Evaluation integrity and 90% improvement plan

No 90% overall target is currently achieved. The saved experiments show real improvements over some earlier baselines, but they do not justify claiming 90% across the datasets. Improving to 90% should mean a score under a prespecified participant-level split with overall accuracy and macro-F1 reported, not tuning against held-out folds or reporting a peak fold.

1. **MMASD:** Keep nested subject-grouped CV as the selection process; prioritize separating squat and maracas classes, audit label/subject balance, compare temporal models using nested CV, and report mean plus fold variation. The saved advanced ensemble is 82.87% / 0.7912.
2. **AV-ASD:** Replace video-only grouping with subject grouping if a subject can contribute multiple videos; keep segment aggregation inside each fold; tune for macro-F1 and minority-class recall; compare against video-level results. Obtain more examples for hyper/hyporeactivity and audit annotations.
3. **ASBD/SSBD:** Obtain and document the actual data and subject identifiers, reproduce the baseline, then run a leakage-safe participant-grouped evaluation before attempting model improvements.
4. **For all tasks:** Freeze the split protocol and preprocessing inside training folds, save out-of-fold predictions and per-class metrics, report confidence intervals, and retain a final untouched test set if enough participants are available.

## Experiment inventory

The companion `results/experiment_inventory_2026_11_12.csv` records the principal saved result artifacts and their scope. The main experiment scripts also include MMASD MEAT-GNN, high-performance, meta-stacking, and AV-ASD multimodal pipelines; where no matching result artifact is present in the workspace, they are recorded as implementation only, not as measured scores.

An additional attempt to run `experiments/mmasd_super_meta.py` on 12 November 2026 stopped at startup because its Python environment lacked the required packages. It produced no score. Subsequent preliminary PPO adaptive-sampling runs were completed; they did not beat the equal-budget baselines. The saved 82.87% / 0.7912 advanced ensemble remains the strongest saved MMASD supervised result.

## Preliminary adaptive-sampling experiment

The initial PPO policy was evaluated against uniform, random, and descriptor-novelty sampling using the same simple classifier within each dataset. These are five-fold mean scores (mean ± sample SD), not pooled out-of-fold metrics. MMASD folds use stored filename-derived group IDs; AV-ASD is audio-only and video-grouped because aligned visual features and verified participant IDs are not present in the local bundle.

| Dataset / task | Uniform | Random | Heuristic novelty | PPO |
|---|---:|---:|---:|---:|
| MMASD, 11-class recording classification, accuracy | 67.29 ± 6.32% | **67.80 ± 4.98%** | 67.65 ± 5.47% | 66.62 ± 6.03% |
| MMASD, macro-F1 | 0.5762 ± 0.0823 | **0.5939 ± 0.0868** | 0.5843 ± 0.0760 | 0.5701 ± 0.0883 |
| AV-ASD, 3-class video majority label, accuracy | **62.22 ± 7.63%** | 59.87 ± 6.06% | 60.98 ± 9.00% | 60.98 ± 11.53% |
| AV-ASD, macro-F1 | **0.5067 ± 0.0819** | 0.4992 ± 0.0604 | 0.5033 ± 0.0577 | 0.4941 ± 0.0584 |

The policy did not improve either result in this prototype. The MMASD run is not directly comparable to the existing 82.87% ensemble or to published graph models because it uses only sampled window descriptors and a small classifier. The AV-ASD video-level result differs from the 961-segment result and is not multimodal. Fold files and script details are listed in `results/experiment_inventory_2026_11_12.csv` and the corresponding adaptive-sampling JSON/CSV artifacts.

## Output tables

- `results/dataset_scorecard_2026_11_12.csv` — dataset/task scorecard and evidence notes.
- `results/experiment_inventory_2026_11_12.csv` — saved result files and implemented pipelines.
- `results/class_metrics_2026_11_12.csv` — available per-class metrics.
- `results/mmasd_adaptive_sampling_rl_folds_2026_11_12.csv` and `results/avasd_adaptive_sampling_rl_folds_2026_11_12.csv` — fold-level preliminary adaptive-sampling results.

