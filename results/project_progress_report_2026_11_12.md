# Autism Behaviour Analysis Project: Progress and Evaluation Report

**Status date:** 12 November 2026  
**Scope:** Saved MMASD and AV-ASD experiments, a subject-grouped MMASD reproduction attempt based on Akinwumi, Qian, and Ojo (2026), and preliminary adaptive-sampling and quality-aware graph experiments.

## Executive summary

The project has completed several supervised MMASD and AV-ASD experiments, plus two subject-grouped unsupervised evaluations inspired by the base paper, *Modelling learning dynamics in autism therapy through explainable multimodal representation learning*. The strongest saved MMASD activity-recognition result is 82.87% accuracy and 0.7912 macro-F1 on 11 activity classes. A closer temporal-autoencoder adaptation improved held-out cluster compactness and activity-label association over the project's PCA baseline, but its assignment stability (mean ARI 0.620 across three model seeds) remains below the paper's reported ARI above 0.8. These are different protocols and are not a direct reproduction or head-to-head comparison.

The local records do not support an RL policy for choosing therapy actions because therapist actions and outcome-based rewards are not available. A first PPO sampler did not improve over simple equal-budget baselines. A new MMASD quality-aware graph pilot has since tested label efficiency and controlled missing-feature robustness. The quality signal showed a modest benefit in several conditions, particularly under synthetic body-group masking, but self-supervised pretraining did not improve clean-input macro-F1 and the model remains well below the project's strongest supervised score. Treat it as an early method pilot, not a completed performance improvement.

## Base paper and comparability

The base paper studies unsupervised learning dynamics, not 11-class activity classification. It reports 255 pose-and-optical-flow windows from 15 therapy sessions, an eight-frame window with stride four, a denoising temporal Transformer autoencoder, six k-means primitives, transition-graph analyses, and an association between early/late primitive-distribution KL divergence and joint-attention duration. It reports clustering stability above ARI 0.8 and a KL/joint-attention correlation of approximately -0.963.

The paper's reported outputs are not accuracy and macro-F1, so they cannot be compared directly with the project's 82.87% supervised accuracy. Its correlation and explainability results also require local joint-attention data and the exact paper session subset. Those inputs were not found in the project workspace. The paper describes multiple sample counts for its analysis stages, so the original session/window selection should be clarified before claiming exact reproduction.

## MMASD: completed supervised experiments

The saved feature set contains 1,693 recordings, 32 stored subject identifiers, and 11 activity labels. Results below are saved out-of-fold results where indicated. The stored group identifiers have not been independently verified as unique participant identities.

| Experiment | Evaluation | Accuracy | Macro-F1 | Status |
|---|---|---:|---:|---|
| Historical temporal GNN | Earlier fixed split | 76.44% | 0.7530 | Historical; protocol differs from later CV |
| Nested Random Forest | 5 outer x 3 inner GroupKFold; pooled features | 81.10% | 0.7716 | Completed, selection-aware |
| Nested temporal-feature RF | 5 outer x 3 inner GroupKFold; pooled/four-bin choice | 80.98% | 0.7688 | Completed, selection-aware |
| Random Forest | 5-fold GroupKFold | 81.75% | 0.7800 | Completed |
| ExtraTrees | 5-fold GroupKFold | 78.32% | 0.7343 | Completed |
| XGBoost | 5-fold GroupKFold | 80.74% | 0.7694 | Completed |
| Weighted RF + ET + XGBoost | 5-fold GroupKFold | 82.04% | 0.7812 | Exploratory; full-label class-weight and OOF blend selection require correction |
| Advanced multi-bank ensemble | 5-fold GroupKFold by stored subject ID | **82.87%** | **0.7912** | Strongest saved result; fold-local scaling, feature selection, prototypes, and class weights |

The advanced result combines multi-bank temporal and rich kinematic features with feature selection, prototype distances, and a weighted RF/ExtraTrees/XGBoost ensemble. Because several components change together, this result is not a one-factor ablation and does not show which component caused the gain. No saved MMASD result reaches 90% accuracy.

## MMASD: unsupervised learning-dynamics reproduction attempt

The local processed sequences were used to construct 7,049 overlapping eight-step windows from 1,661 recordings. A 128-dimensional PCA embedding and six-cluster k-means model were evaluated using five subject-grouped folds. Each fold's scaler, PCA projection, and cluster centroids were fitted on training subjects; held-out windows were assigned to the learned centroids. Five independent k-means runs were evaluated per fold, with each run selecting the lowest-inertia result from three training-only initializations. This is the non-neural reference for the temporal-autoencoder adaptation below.

The mean held-out cluster stability of 0.742 is numerically below the base paper's reported ARI above 0.8. This is not a like-for-like comparison: the paper reports bootstrap stability on Transformer embeddings, while this project reports agreement across independent k-means runs on held-out subjects using PCA-compressed kinematic windows.

| Held-out metric | Mean across 5 folds | SD across folds |
|---|---:|---:|
| Sampled silhouette | 0.0343 | 0.0416 |
| Davies-Bouldin index | 3.7003 | 0.3743 |
| Calinski-Harabasz index | 80.8577 | 45.4644 |
| ARI against the 11 activity labels | 0.0928 | 0.0899 |
| NMI against the 11 activity labels | 0.2198 | 0.0910 |
| Cluster assignment stability across k-means runs (ARI) | 0.7422 | 0.1083 |

Across 5,388 held-out within-recording transitions, the observed self-transition rate was 0.8168, compared with a mean of 0.7720 under 199 within-recording label-shuffle permutations (one-sided permutation p = 0.005). Early-to-late primitive-distribution KL divergence was computed for 940 recordings with at least four windows (mean 0.1110; median 0.0141). The local project has no session-aligned joint-attention duration values, so the paper's KL/joint-attention correlation was not re-estimated.

### Temporal-autoencoder adaptation

A closer architecture adaptation was then trained on the same windows. It uses a two-layer, four-head temporal Transformer encoder with a 128-dimensional representation, 256-dimensional feed-forward layers, 0.1 dropout, Adam at 1e-3, SmoothL1 reconstruction loss, ten epochs, and three independent training seeds within each of five subject-held-out folds. Gaussian noise and 5% temporal-frame dropout were used for denoising. Because the local input contains 64 kinematic channels per frame rather than the paper's pose-distance plus optical-flow channels, the model is not an exact reproduction.

| Held-out metric | Mean across 5 folds | SD across folds |
|---|---:|---:|
| Reconstruction loss | 0.0141 | 0.0190 |
| Sampled silhouette | 0.1474 | 0.0635 |
| Davies-Bouldin index | 1.9346 | 0.2750 |
| Calinski-Harabasz index | 248.5648 | 92.3816 |
| ARI against the 11 activity labels | 0.2886 | 0.0971 |
| NMI against the 11 activity labels | 0.4406 | 0.0653 |
| Assignment stability across three model seeds (ARI) | 0.6203 | 0.0984 |

Compared with the PCA reference, the autoencoder's held-out clusters were more compact (higher silhouette, lower Davies-Bouldin) and more associated with the known 11 activity labels (higher ARI/NMI). However, assignment stability across seeds decreased from 0.742 to 0.620, so the improved compactness does not establish more reproducible primitives. Fold 2 also had higher reconstruction loss than the other folds, contributing to the large fold variation. The observed self-transition rate was 0.9213 versus a shuffled-label mean of 0.8936 (199 permutations, p = 0.005). Since the windows overlap by 50%, persistence can partly arise from shared frames between adjacent windows; this result alone does not establish meaningful behavioural transitions. Early-to-late KL across 940 recordings had mean 0.0539 and median 0.0000. Joint-attention correlation remains unavailable.

The autoencoder fold-level results are saved in `mmasd_temporal_autoencoder_folds_2026_11_12.csv`, with the full configuration and summary in `mmasd_temporal_autoencoder_2026_11_12.json`. The training code is `experiments/mmasd_temporal_autoencoder_replication.py`.

The first PCA experiment is a transparent baseline. The later temporal-autoencoder run is a closer architectural adaptation, but both remain **partial reproductions** because they use local 64-channel kinematic sequences rather than the paper's 528-dimensional 2D pose distances plus three optical-flow descriptors. Neither run reproduces the paper's exact sample subset, its saliency analysis, or its joint-attention association. The baseline fold metrics are saved in `mmasd_dynamics_replication_folds_2026_11_12.csv`; its protocol and summary are in `mmasd_dynamics_replication_2026_11_12.json`.

### Quality-aware, label-efficient graph pilot

A new spatiotemporal graph model was implemented for the cached 11-class MMASD kinematic sequences. It performs anatomical message passing over the eight body groups, models time with a bidirectional GRU, and can receive an observed-group mask. A masked-feature reconstruction objective was used for self-supervised pretraining. The evaluation compared three variants at 25%, 50%, and 100% label budgets under five size-balanced outer folds grouped by the stored identifiers. Each model was also evaluated with 0%, 20%, and 40% synthetic body-group masking.

| Label budget | Variant | Clean accuracy | Clean macro-F1 | 40% synthetic masking accuracy | 40% synthetic masking macro-F1 |
|---:|---|---:|---:|---:|---:|
| 25% | Supervised, no quality signal | 40.56% ± 10.24 | 0.3304 ± 0.1053 | 33.57% ± 12.46 | 0.2482 ± 0.0846 |
| 25% | Supervised, quality mask | 44.24% ± 9.80 | 0.3372 ± 0.1145 | 38.69% ± 10.53 | 0.3174 ± 0.1336 |
| 25% | Self-supervised + quality mask | 40.80% ± 8.22 | 0.3318 ± 0.1145 | 37.72% ± 10.96 | 0.3164 ± 0.1127 |
| 50% | Supervised, no quality signal | 53.11% ± 7.08 | 0.4197 ± 0.0740 | 38.09% ± 9.28 | 0.2845 ± 0.1069 |
| 50% | Supervised, quality mask | 57.06% ± 7.18 | 0.4395 ± 0.1108 | 45.96% ± 9.93 | 0.3528 ± 0.0820 |
| 50% | Self-supervised + quality mask | 49.90% ± 4.78 | 0.4058 ± 0.0887 | 45.09% ± 6.45 | 0.3667 ± 0.0997 |
| 100% | Supervised, no quality signal | 61.23% ± 8.73 | 0.4806 ± 0.0942 | 40.70% ± 14.45 | 0.3228 ± 0.1229 |
| 100% | Supervised, quality mask | 62.33% ± 8.64 | 0.5079 ± 0.1093 | 51.30% ± 6.96 | 0.4174 ± 0.1158 |
| 100% | Self-supervised + quality mask | 60.09% ± 3.43 | 0.4750 ± 0.0733 | **53.95% ± 2.52** | **0.4407 ± 0.0822** |

Scores are mean ± sample standard deviation across the five outer folds. The explicit quality mask modestly improved clean macro-F1 over the same supervised model without that signal at all three label budgets. At 40% synthetic masking, the self-supervised variant had the strongest mean at the 100% label budget, while the quality-aware supervised variant was strongest at 25% and 50%. The masked reconstruction loss decreased in every fold, but pretraining did not improve clean-input macro-F1 over supervised quality-aware training. Fold variability is substantial, and the evaluation uses one training seed.

The cache contains kinematic summaries but no measured joint-confidence values or real missing-joint masks. The corruption experiment therefore tests robustness to controlled synthetic body-group dropout; it does not establish robustness to naturally occurring pose-estimation failures. This graph pilot also uses a smaller model than the feature-ensemble system and achieved at most 62.33% clean accuracy / 0.5079 macro-F1. It does not beat the saved 82.87% / 0.7912 ensemble or published graph models, and its input/splits do not yet reproduce their benchmark protocol.

Code and outputs: `models/quality_aware_stgcn.py`, `experiments/mmasd_quality_aware_label_efficiency.py`, `results/mmasd_quality_aware_label_efficiency_folds_2026_11_12.csv`, `results/mmasd_quality_aware_label_efficiency_summary_2026_11_12.csv`, and `results/mmasd_quality_aware_label_efficiency_2026_11_12.json`.

### OpenPose confidence-aware graph experiment (local 2D archive)

To test the quality signal with real pose-estimator confidence values, the local OpenPose archive was processed into clip-level BODY_25 sequences. Each clip was resampled to 64 frames, centered on MidHip, scaled by shoulder width, and represented with normalized coordinates and first-order motion. A compact anatomical graph encoder and bidirectional GRU were evaluated in five filename-subject-grouped folds. The proposed variant uses joint confidence to gate graph features and weight temporal pooling; the control uses the same network without confidence. The validation groups were drawn only from outer training subjects.

| Variant | Accuracy, mean ± SD | Macro-F1, mean ± SD |
|---|---:|---:|
| Confidence-blind graph + BiGRU | 39.84% ± 8.94 | 0.3219 ± 0.1068 |
| Confidence-aware graph + BiGRU | 34.88% ± 9.37 | 0.2584 ± 0.1161 |

The confidence-aware variant did not improve this run, so it is not presented as a successful contribution. The local archive contains 1,374 ten-class clip folders across 32 filename-derived IDs, including 343 drumming clips. This differs from the 1,202-clip class distribution reported in Ay et al. (2026); no unverified filtering was applied to force a count match. The experiment is therefore a local-release pilot, not a reproduction or a matched comparison with RC-ProtoGCN. The filename IDs are not independently verified participant identities. Scores are from one seed, and the model is substantially simpler than ProtoGCN. The output cache, script, fold table, and full JSON are `data/processed/mmasd_openpose/mmasd_openpose_64frames.npz`, `experiments/mmasd_openpose_quality_graph.py`, `results/mmasd_openpose_quality_graph_folds_2026_11_12.csv`, and `results/mmasd_openpose_quality_graph_2026_11_12.json`.

As a stronger reference on the same cached clips and folds, a class-weighted MLP was trained on global coordinate/motion summaries, 10th and 90th percentiles, and ordered eight-bin temporal means and standard deviations. After aligning its inner validation groups with the graph experiment, it reached 70.17% ± 10.77 accuracy and 0.6338 ± 0.1903 macro-F1. The large fold variation, especially in macro-F1, cautions against treating this single-seed result as stable. It provides evidence that temporal summary features are more effective than the current compact graph-GRU implementation on this local archive; it does not establish that a novel graph contribution has improved performance. Outputs are saved in `experiments/mmasd_openpose_temporal_features.py`, `results/mmasd_openpose_temporal_features_folds_2026_11_12.csv`, and `results/mmasd_openpose_temporal_features_2026_11_12.json`.

Follow-up ablations used the same five outer folds and aligned 18% grouped inner-validation rule:

| Local 2D variant | Accuracy, mean ± SD | Macro-F1, mean ± SD | Interpretation |
|---|---:|---:|---|
| Temporal-summary MLP, one seed | 70.17% ± 10.77 | **0.6338 ± 0.1903** | Strongest single model in this pilot |
| Confidence-weighted temporal descriptors | 70.24% ± 8.44 | 0.6255 ± 0.1929 | Accuracy is effectively tied; macro-F1 is lower |
| Three-seed probability ensemble | **70.83% ± 8.98** | 0.6226 ± 0.1903 | Small accuracy increase; macro-F1 decreases |
| Theme auxiliary head with documented theme labels | 68.57% ± 11.63 | 0.6080 ± 0.2073 | Did not improve activity recognition |
| Joint/bone, motion, and multi-scale temporal MLP | 70.74% ± 11.19 | 0.6265 ± 0.2082 | Added bone streams; no macro-F1 improvement |
| Nested RBF kernel ridge on temporal summaries | **71.69% ± 9.10** | **0.6394 ± 0.1918** | Best local 2D pilot; inner-fold search over kernel width and ridge penalty |
| RBF with tuned class-prior offsets | 69.29% ± 11.85 | 0.6207 ± 0.2265 | Inner-validation prior adjustment reduced both metrics |
| Nested Random Forest / ExtraTrees | 68.42% ± 8.34 | 0.6242 ± 0.1845 | Inner-fold choice among model, feature count, and leaf size; pooled OOF accuracy 68.41%, macro-F1 0.6517 |

The nested RBF model's pooled out-of-fold score is 71.69% accuracy and 0.6920 macro-F1; the table reports fold means and standard deviations. The theme labels follow the dataset README: robotic-assisted therapy includes arm swing, body swing, chest expansion, and squat; rhythm includes drumming, both maracas activities, and sing-and-clap; yoga includes tree and twist pose in this ten-class subset. The hierarchy run uses this corrected mapping. The nested RBF model improves slightly over the plain MLP on the same local data, but remains far below the published subject-independent result. It is a conventional classifier, not a novel method. None of the tested follow-ups establishes a new contribution or evidence of outperforming prior work. A reproducible archive audit is saved in `experiments/audit_mmasd_openpose_archive.py` and `results/mmasd_openpose_archive_audit_2026_11_12.csv/.json`; it confirms that the local archive's 1,374 ten-class clip inventory does not match the published 1,202-clip table. No clips were removed to force a match. Detailed scripts and outputs are recorded in the experiment inventory.

## AV-ASD: completed results

These AV-ASD experiments have different labels and evaluation units and should remain separate in reporting.

### Recent-paper-informed multi-label audio evaluation

The new AV-ASD experiment follows the original AV-ASD paper's multi-label behavior-recognition formulation, reported in [*Hear Me, See Me, Understand Me: Audio-Visual Autism Behavior Recognition*](https://ieeexplore.ieee.org/abstract/document/10812841) (IEEE Transactions on Multimedia, 2025). Recent work such as [*Multimodal behavioral analysis for autism spectrum disorder assessment*](https://www.sciencedirect.com/science/article/pii/S0031320326001202) (Pattern Recognition, 2026) further motivates combining behavioral cues across modalities, but its clinical data and modalities are not available in this project bundle. The local files contain segment-level acoustic summaries only, so the experiment below is an audio-only adaptation; it does not reproduce either paper's multimodal system.

The evaluation maps VAD segments to annotated clips using the saved temporal-alignment records, pools the 32-dimensional descriptors into 259-dimensional clip vectors (global statistics, four ordered temporal bins, and segment/duration summaries), and retains the ten annotation indicators as multi-label targets. Five outer folds keep clips from a source video together. In each outer training partition, one grouped validation partition selects logistic-regression regularization and per-label thresholds; preprocessing is fitted on training clips only. The headline results use the fixed 0.5 threshold because validation-selected thresholds did not improve the pooled out-of-fold macro-F1.

| Evaluation | Clips / source-video groups | Metric | Out-of-fold result |
|---|---:|---|---:|
| 10-label multi-label, nine behavior labels (primary) | 227 / 138 | Macro-F1, fixed threshold 0.5 | 0.2811 |
| 10-label multi-label, nine behavior labels (primary) | 227 / 138 | Micro-F1, fixed threshold 0.5 | 0.3145 |
| Nine behavior labels, ignoring the rare Background label | 227 / 138 | Macro average precision | 0.2202 |
| Ten indicators, exact set match | 227 / 138 | Exact-match rate, fixed threshold 0.5 | 0.0176 |
| Nine behavior labels (validation-selected thresholds) | 227 / 138 | Macro-F1 | 0.2688 |

The 227 clips are only a subset of the 928 annotated clips in the full AV-ASD collection; just seven have a positive Background annotation. The rare classes, especially Object Lining-Up (11 positive clips), are difficult to estimate reliably. The saved audio audit also marks target-speaker identity as unverified, and `source_video_id` is not verified as a participant identifier. Consequently, this is a source-video-grouped, audio-only benchmark on an incomplete feature subset, not participant-independent evaluation or an autism assessment result. It cannot be ranked directly against the earlier four-class and three-class scores, which use different labels, sample units, and scoring rules. The new run does not reach a 90% target and does not show a reliable performance improvement over the earlier task-specific scores; its contribution here is a more faithful multi-label evaluation of the available audio data.

The experiment code is `experiments/avasd_multilabel_audio_grouped.py`. Its fold-level results, out-of-fold predictions, and full configuration are saved as `results/avasd_multilabel_audio_grouped_folds_2026_11_12.csv`, `results/avasd_multilabel_audio_grouped_oof_2026_11_12.csv`, and `results/avasd_multilabel_audio_grouped_2026_11_12.json`.

| Task and unit | Experiment | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Four-class, 182 clips | Audio XGBoost benchmark | 50.55% | 0.4191 |
| Four-class, 1,131 annotated segments | Engineered audio-feature ensemble | 54.38% | 0.3522 |
| Four-class, video level | Aggregated segment features | 31.90% | 0.3267 |
| Three-class, 961 annotated segments | Engineered-feature ensemble | 76.59% | 0.5370 |
| Three-class, video level | Aggregated segment predictions | 54.02% | 0.5016 |

The three-class segment result is not a reliable headline estimate as currently pooled. Although its 5-fold GroupKFold implementation keeps each video together, one video contributes 506 of 961 segments (52.7%) and consists entirely of upper-limb-stereotypy examples. An audit found 506 distinct segment IDs spanning about 29 minutes, so this is a very long recording rather than duplicated rows. That single fold had 92.49% accuracy; the other four folds together yield about 58.9% accuracy. Thus the pooled 76.59% is heavily driven by one very large, single-class video. The class imbalance is also substantial (700 upper-limb, 206 aggressive, 55 hyper-/hyporeactivity segments), and the minority class has F1 0.168. This score should be reported with the fold sizes and this limitation, not presented as evidence that the method generalizes at 76.6%. Video grouping also does not establish participant independence.

#### New within-video audio-context experiment

Following the context-aware multimodal direction in recent AV-ASD research, a new audio-only experiment adds previous/current/next segment descriptors and unlabeled per-video mean and standard deviation summaries. A small candidate set of SVC and ExtraTrees models is selected with three-fold GroupKFold inside each outer training fold; results are measured on five outer video-grouped folds. The held-out video's unlabeled audio summaries are available to this offline method, so the setting assumes the full video is available at prediction time.

| Measure | Result |
|---|---:|
| Pooled segment accuracy | 67.01% |
| Macro-F1 | 0.4662 |
| Balanced accuracy | 0.4838 |
| Mean accuracy when each of 87 videos receives equal weight | 46.28% |
| Accuracy after excluding the largest video, as a diagnostic only | 38.68% |
| Majority-class segment baseline | 72.84% |

This context experiment does not reach 80% and does not beat the majority-class baseline on pooled accuracy. Its five fold accuracies vary sharply, from 4.39% to 92.49%. The largest video alone contains 506 segments, all from the same target class; removing it is only a diagnostic and must not be treated as a new benchmark. The result indicates that this feature-only approach is not robust enough for a performance claim. The more immediate research need is to verify the evaluation unit, obtain participant identifiers, inspect the unusually large video, and add aligned visual or pose features before attempting a multimodal model.

Code and outputs: `experiments/avasd_video_context_nested.py`, `results/avasd_video_context_nested_2026_11_12.json`, `results/avasd_video_context_nested_folds_2026_11_12.csv`, `results/avasd_video_context_nested_oof_2026_11_12.csv`, and `results/avasd_video_context_nested_per_video_2026_11_12.csv`.

#### Video-level audio classification

Because the segment counts are highly uneven, a separate experiment created one sample per video. Each video is represented by 259 pooled acoustic and temporal-summary features, and its target is the most frequent segment label under the same fixed priority rule. This produced 87 video examples with class counts of 21 aggressive, 20 hyper-/hyporeactivity, and 46 upper-limb-stereotypy videos. Nested 5-fold stratified evaluation, with three-fold inner model selection, achieved 60.92% accuracy, 0.5492 macro-F1, and 0.5527 balanced accuracy. The majority-video-class baseline is 52.87%; fold accuracy ranged from 50.00% to 76.47%. It improves on that baseline but remains below the 80% target. Because videos are not confirmed independent participants, this result is video-level only and cannot establish participant-independent performance.

The implementation and outputs are `experiments/avasd_video_level_nested.py`, `results/avasd_video_level_nested_2026_11_12.json`, `results/avasd_video_level_nested_folds_2026_11_12.csv`, and `results/avasd_video_level_nested_oof_2026_11_12.csv`.

#### AV-ASD modality availability and next experiment gate

The local AV-ASD audit bundles contain 32-dimensional acoustic segment descriptors, annotations, and temporal alignment reports. The project has no AV-ASD raw video, extracted frame, pose/keypoint, or speech-embedding files in the inspected data locations. The target participant in the audio is also unverified. The prepared MEAT-GNN AV-ASD script expects MediaPipe keypoints under `data/processed/keypoints/av_asd/`; that input directory is not present, so its multimodal graph experiment has not run. The official AV-ASD repository documents a separate video-download and clip-extraction process and notes that some source videos may no longer be available; it should be treated as a method/data-preparation reference, not evidence that those raw files are in this project. Until suitably authorized and aligned video/pose inputs and participant identifiers are available, the experiments above are the extent of what can be evaluated locally, and multimodal fusion or participant-independent claims should remain pending.

## ASBD and SSBD status

The current Chapter 4 text records an ASBD audit of 132 downloaded video files, with heuristic labels assigned to 122. These counts should be retained only if supported by the corresponding audit log or spreadsheet. No reproducible ASBD classifier result is saved in the project artifacts.

SSBD is discussed as a published dataset, but local source videos and a standalone re-evaluation result were not found. A previous 55.90% / 0.5576 ASBD/SSBD claim duplicates the AV-ASD three-class result and is not supported as a separate ASBD/SSBD evaluation in the current project records.

## Preliminary PPO adaptive-sampling experiment

The initial implementation uses a PPO actor-critic to choose strides between candidate windows. It trains inside each outer training fold and compares eight-window MMASD samples or up to eight AV-ASD audio segments with uniform, random, and deterministic novelty-based sampling. The reward is a descriptor-diversity proxy; it is not the decoder reconstruction reward described in the earlier proposal, and it does not represent a clinical or therapy outcome. Every sampler uses the same small PyTorch classifier in this first comparison.

| Dataset and unit | Sampler | Mean accuracy across folds | SD | Mean macro-F1 across folds | SD |
|---|---|---:|---:|---:|---:|
| MMASD, 1,693 recordings, 11 classes; stored-group folds | Uniform | 67.29% | 6.32 pp | 0.5762 | 0.0823 |
| MMASD, same task and folds | Random | **67.80%** | 4.98 pp | **0.5939** | 0.0868 |
| MMASD, same task and folds | Heuristic novelty | 67.65% | 5.47 pp | 0.5843 | 0.0760 |
| MMASD, same task and folds | PPO | 66.62% | 6.03 pp | 0.5701 | 0.0883 |
| AV-ASD, 87 videos, 3-class video majority label | Uniform | **62.22%** | 7.63 pp | **0.5067** | 0.0819 |
| AV-ASD, same task and folds | Random | 59.87% | 6.06 pp | 0.4992 | 0.0604 |
| AV-ASD, same task and folds | Heuristic novelty | 60.98% | 9.00 pp | 0.5033 | 0.0577 |
| AV-ASD, same task and folds | PPO | 60.98% | 11.53 pp | 0.4941 | 0.0584 |

These are means and sample standard deviations of five fold scores, not pooled out-of-fold metrics. The MMASD groups are filename-derived and have not been independently confirmed as participant IDs. AV-ASD is audio-only because the local bundle lacks aligned visual/pose features, and its participant-independent status is unverified. The AV-ASD table evaluates videos with a majority label, so it is not comparable to the earlier 961-segment result. Neither PPO result exceeds its strongest same-run simple sampler; neither result is a matched comparison with a published graph or multimodal model. These experiments do not establish a 90% result or that the project outperforms prior work.

The results and fold-level scores are saved in `mmasd_adaptive_sampling_rl_2026_11_12.json`, `mmasd_adaptive_sampling_rl_folds_2026_11_12.csv`, `avasd_adaptive_sampling_rl_2026_11_12.json`, and `avasd_adaptive_sampling_rl_folds_2026_11_12.csv`. Scripts: `experiments/mmasd_adaptive_sampling_rl.py` and `experiments/avasd_adaptive_sampling_rl.py`.

RL-based frame selection and RL in autism-related tasks both have prior work. A defensible novelty claim would need to be limited to the specific integration and evaluation, and should only be made if a revised method improves a prespecified metric on held-out groups. The next experiment should strengthen the reward using only training-fold evidence, compare with the established MMASD/AV-ASD backbones under matched splits, and retain a separate held-out evaluation. The 82.87% MMASD supervised result is not evidence of outperforming the learning-dynamics base paper, whose outcomes are unsupervised stability and behavioral-structure measures.
