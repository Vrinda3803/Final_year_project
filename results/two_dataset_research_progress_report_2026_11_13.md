# Research Progress Report: MMASD and AV-ASD

**Prepared:** 13 November 2026  
**Project aim:** Develop and evaluate behavior-recognition methods on MMASD and AV-ASD, compare them fairly with the relevant published work, and identify a defensible research contribution.  
**Scope:** This report covers the saved local experiments for MMASD and AV-ASD only. It distinguishes measured results, exploratory pilots, and published results. It does not treat activity recognition as autism diagnosis.

## 1. Executive summary

The project has evaluated two different behavior-recognition problems. MMASD is used for therapy-activity recognition from pose or kinematic data; AV-ASD is used for autism-related behavior recognition from annotated, often multi-label video clips. Their labels, input modalities, sample units, and evaluation protocols differ, so one dataset's score cannot be ranked directly against the other.

The strongest saved MMASD result is **82.87% accuracy and 0.7912 macro-F1** for an 11-class, stored-grouped activity-recognition experiment using a multi-bank feature representation and a Random Forest/ExtraTrees/XGBoost ensemble. This result is below the recent published 10-class 2D-skeleton RC-ProtoGCN subject-independent result (89.29% pooled accuracy, 0.8837 F1), but the task and data are not matched. A local 10-class 2D OpenPose archive experiment reached 71.69% mean fold accuracy and 0.6394 mean fold macro-F1; its clip inventory differs from the paper's 1,202 clips, so it is not a reproduction.

The strongest local AV-ASD number for the earlier three-class segment task is 76.59% accuracy and 0.5370 macro-F1. That accuracy is dominated by one unusually long video containing **506 of 961 segments**, all assigned to upper-limb stereotypies. It should not be treated as a stable estimate of generalization. A nested video-level experiment reached 60.92% accuracy and 0.5492 macro-F1 on 87 videos, above a 52.87% majority-video baseline but below the project's 80% accuracy aim. The closest local experiment to the published AV-ASD multi-label task uses only 227 clips with usable audio, and achieves 0.2811 behavior macro-F1 at a fixed threshold.

No local AV-ASD experiment currently matches the paper's ten-label, multimodal, 928-clip test task. The local AV-ASD bundle provides audio descriptors, annotations, and alignment files, but no aligned visual or pose features. The project has not yet demonstrated a result that outperforms the relevant published work under a matched protocol. The current strongest contribution is a set of auditable experiments and evaluation findings, not a confirmed novel model improvement.

## 2. How to read the results

- **Accuracy** is the fraction of correctly classified examples. It can look high when one class dominates.
- **Macro-F1** averages F1 equally across classes and is more informative when classes are imbalanced.
- **Pooled out-of-fold (OOF)** scores combine the held-out predictions from all folds. Mean fold scores are also reported for some experiments; those two summaries can differ.
- **GroupKFold by a stored ID** prevents records with the same stored ID from crossing folds. It does not prove that the ID is a unique participant unless that identity has been verified.
- **Video-level and segment-level tasks are different.** In the local AV-ASD results, a segment can be a short audio segment, while a video-level example aggregates many segments.
- **Published results below use the paper's own metric and split.** Similar-looking percentages are not automatically comparable.

## 3. Datasets and local evidence

### MMASD

The project has a processed set of 1,693 activity recordings with 11 activity labels and 32 stored group identifiers. The stored identifiers are derived from filenames and have not been independently checked as participant identities. The main feature-based experiment uses temporal and kinematic feature banks; a separate processed OpenPose archive provides 2D pose sequences for a ten-class pilot.

### AV-ASD

The published dataset contains 928 annotated clips from 569 source videos, with ten multi-label behavior indicators. The local audio audit bundle contains 5,410 segment-level 32-dimensional acoustic descriptors plus annotation and temporal alignment reports. It does not provide full aligned video, pose/keypoints, or speech embeddings in the inspected project/download locations. Target-speaker identity in audio is unverified. In one local three-class task there are 961 annotated segments from 87 video IDs; the label counts are 206 aggressive, 55 hyper-/hyporeactivity, and 700 upper-limb stereotypies.

## 4. MMASD work completed

### 4.1 Supervised 11-class activity recognition

The main supervised path compared compact temporal features, a Random Forest, ExtraTrees, XGBoost, and ensembles. Nested experiments used five outer folds and three inner group folds for model selection. The advanced pipeline combined 1,248 multi-bank temporal features with 506 rich features (1,754 before selection), applied fold-local RobustScaler and ANOVA SelectKBest to retain 800 features, added 11 training-fold class-prototype distances, then fitted RF, ExtraTrees, and XGBoost models. The saved ensemble weights are 0.40, 0.35, and 0.25. Evaluation uses five folds grouped by stored subject ID, with seed 42.

| Saved experiment | Evaluation / representation | Accuracy | Macro-F1 | Interpretation |
|---|---|---:|---:|---|
| Historical temporal GNN | Earlier fixed split | 76.44% | 0.7530 | Historical score; split differs from later grouped CV |
| Nested Random Forest | 5 outer × 3 inner group folds; pooled features | 81.10% | 0.7716 | Selection-aware baseline |
| Nested pooled vs. four-bin RF | 5 outer × 3 inner group folds | 80.98% | 0.7688 | Four-bin option did not improve the nested result |
| Random Forest | 5-fold stored-group CV | 81.75% | 0.7800 | Single-model benchmark |
| ExtraTrees | 5-fold stored-group CV | 78.32% | 0.7343 | Single-model benchmark |
| XGBoost | 5-fold stored-group CV | 80.74% | 0.7694 | Single-model benchmark |
| Weighted RF + ExtraTrees + XGBoost | 5-fold stored-group CV | 82.04% | 0.7812 | Exploratory: saved version uses full-label class weights and searches blend weights on pooled OOF predictions |
| **Advanced multi-bank ensemble** | **5-fold stored-group CV; 800 selected base features + 11 prototype distances** | **82.87%** | **0.7912** | Strongest saved local MMASD result; pipeline-level gain, not a one-factor ablation |

The 82.87% score is a supervised **activity-recognition** result. It is not an autism-vs-nonautism assessment. The gain over the earlier 82.04% ensemble is 0.83 percentage points in accuracy and about 0.010 in macro-F1, but several pipeline components changed together. The local stored-group IDs still need participant-identity verification.

### 4.2 Unsupervised learning-dynamics adaptation

The project attempted an adaptation of Akinwumi, Qian, and Ojo's 2026 temporal-autoencoder paper. The published method uses 255 pose/optical-flow windows from 15 therapy sessions, eight-frame windows with stride four, a denoising Transformer autoencoder, six clusters, transition graphs, and saliency analysis. It reports cluster agreement above ARI 0.8 and a KL-divergence/joint-attention correlation near −0.963 (p ≈ 0.002). These outputs are not classification accuracy.

The local adaptation used 7,049 eight-step windows from 1,661 recordings, 64 kinematic channels per frame, six clusters, and five stored-subject folds. It lacks the paper's exact pose-distance-plus-optical-flow inputs, sample subset, and joint-attention annotations.

| Local unsupervised variant | Held-out result | What it means |
|---|---|---|
| PCA + k-means baseline | Silhouette 0.0343; activity ARI 0.0928; NMI 0.2198; cluster-seed stability ARI 0.7422 | Reference clustering on local kinematics |
| Denoising temporal Transformer autoencoder + k-means | Silhouette 0.1474; activity ARI 0.2886; NMI 0.4406; seed-stability ARI 0.6203 | Better cluster compactness and label association than PCA; less stable between seeds |

The autoencoder's observed self-transition rate is 0.9213 versus a shuffled mean of 0.8936 (199 permutations, p = 0.005). Overlapping windows share half of their frames, so this persistence can partly be caused by construction. The KL values were computed locally, but the paper's correlation with joint attention was not re-estimated because local aligned attention-duration data are unavailable. This remains a **partial adaptation**, not an exact reproduction.

### 4.3 Quality-aware spatiotemporal graph pilot

A graph model with an anatomical graph over eight body groups and a bidirectional GRU was evaluated with label budgets of 25%, 50%, and 100%. A quality-mask input represented observed groups; a masked-feature reconstruction objective supplied self-supervised pretraining. Outer folds were grouped by stored IDs, and synthetic 0%, 20%, and 40% body-group masking tested controlled robustness.

| Labels used | Model | Clean accuracy / macro-F1 | At 40% synthetic masking: accuracy / macro-F1 |
|---:|---|---:|---:|
| 25% | Supervised, no quality signal | 40.56% / 0.3304 | 33.57% / 0.2482 |
| 25% | Supervised, quality mask | 44.24% / 0.3372 | 38.69% / 0.3174 |
| 25% | Self-supervised + quality mask | 40.80% / 0.3318 | 37.72% / 0.3164 |
| 50% | Supervised, no quality signal | 53.11% / 0.4197 | 38.09% / 0.2845 |
| 50% | Supervised, quality mask | 57.06% / 0.4395 | 45.96% / 0.3528 |
| 50% | Self-supervised + quality mask | 49.90% / 0.4058 | 45.09% / 0.3667 |
| 100% | Supervised, no quality signal | 61.23% / 0.4806 | 40.70% / 0.3228 |
| 100% | Supervised, quality mask | 62.33% / 0.5079 | 51.30% / 0.4174 |
| 100% | Self-supervised + quality mask | 60.09% / 0.4750 | **53.95% / 0.4407** |

The quality mask helped some clean and masked conditions, but self-supervised pretraining did not improve clean-input macro-F1 over the supervised quality-aware model. The cache has no real pose-confidence or missing-joint measurements; the missingness experiment is synthetic. This is a promising robustness question, but this run does not beat the 82.87% supervised ensemble.

### 4.4 Local 2D OpenPose archive experiments

The local OpenPose archive contains 1,374 clips, ten classes, and 32 filename-derived IDs. The recent comparison paper uses 1,202 clips in its ten-class 2D setup; the local distribution also contains 343 drumming clips versus 168 in the paper. We did not remove examples to force matching. All results below are local pilots, not protocol-matched replications.

| Local 2D approach | Accuracy (mean ± SD) | Macro-F1 (mean ± SD) | Main finding |
|---|---:|---:|---|
| Confidence-blind graph + BiGRU | 39.84% ± 8.94 | 0.3219 ± 0.1068 | Compact graph pilot |
| Confidence-aware graph + BiGRU | 34.88% ± 9.37 | 0.2584 ± 0.1161 | Confidence gating did not help |
| Temporal-summary MLP | 70.17% ± 10.77 | 0.6338 ± 0.1903 | Stronger conventional baseline |
| Confidence-weighted temporal MLP | 70.24% ± 8.44 | 0.6255 ± 0.1929 | Accuracy tied; macro-F1 lower |
| Three-seed MLP probability ensemble | 70.83% ± 8.98 | 0.6226 ± 0.1903 | Slight accuracy increase, lower macro-F1 |
| Theme auxiliary head | 68.57% ± 11.63 | 0.6080 ± 0.2073 | Did not improve activity classification |
| Joint/bone/motion multi-stream MLP | 70.74% ± 11.19 | 0.6265 ± 0.2082 | Extra streams did not improve macro-F1 |
| Nested RBF kernel ridge | **71.69% ± 9.10** | **0.6394 ± 0.1918** | Best local 2D pilot by fold means; pooled OOF 71.69% accuracy / 0.6920 macro-F1 |
| RBF with class-prior offsets | 69.29% ± 11.85 | 0.6207 ± 0.2265 | Prior correction reduced both metrics |
| Nested RF / ExtraTrees | 68.42% ± 8.34 | 0.6242 ± 0.1845 | Pooled OOF 68.41% / 0.6517 |

The local graph variant did not outperform the feature-based models. The project has not yet shown a graph-model gain on MMASD under a matched, subject-independent protocol.

### 4.5 PPO adaptive temporal sampling

PPO was tested as a sampler, not as a therapy policy. Its reward was training-set descriptor novelty with a small selection cost; no therapist actions or clinical outcomes exist in the data. On MMASD, the same small classifier was evaluated under five stored-group folds with an eight-window budget.

| Sampler | Accuracy, mean ± SD | Macro-F1, mean ± SD |
|---|---:|---:|
| Uniform | 67.29% ± 6.32 | 0.5762 ± 0.0823 |
| Random | 67.80% ± 4.98 | 0.5939 ± 0.0868 |
| Novelty heuristic | 67.65% ± 5.47 | 0.5843 ± 0.0760 |
| PPO | 66.62% ± 6.03 | 0.5701 ± 0.0883 |

PPO did not beat uniform, random, or heuristic sampling in this experiment. The result should be reported as a negative preliminary ablation, not as evidence that reinforcement learning improves autism assessment.

## 5. AV-ASD work completed

### 5.1 Published AV-ASD method and benchmark

Deng et al.'s AV-ASD work frames the problem as ten-label behavior recognition over 928 clips from 569 source videos. The paper describes 553 training, 193 validation, and 182 test clips, and reports macro-F1. Its feature systems use CLIP for visual features, ImageBind for visual/video/audio features, Whisper for speech, temporal models, and LLaVA instruction tuning. The paper describes a random split of clips; it does not establish source-video- or participant-disjoint evaluation in the split description.

| Published method | Modality / setup | Test macro-F1 |
|---|---|---:|
| Dummy baseline | Positive predictions | 26.83% |
| CLIP + MLP | Visual | 45.72% |
| ImageBind temporal | Video | 50.87% |
| ImageBind temporal | Audio | 41.74% |
| Whisper temporal | Speech | 39.69% |
| ImageBind + Whisper temporal fusion | Audio + speech | 53.17% |
| **LLaVA-ASD** | **Fine-tuned visual-language model; visual + speech is the best reported modality combination** | **59.77%** |

The fine-tuned LLaVA modality table reports 53.45% for visual only, 52.54% for visual+audio, 59.77% for visual+speech, and 57.84% for visual+audio+speech. Thus adding every modality did not give the top result. These are F1 percentages, not accuracy percentages. The paper's 59.77% is not a three-class accuracy result.

Other reported AV-ASD baselines include CLIP linear probe 45.03% and MLP 45.72%; ImageBind image linear/MLP 12.12%/39.19%; ImageBind video linear/MLP 10.14%/44.06%; ImageBind audio linear/MLP 27.54%/28.47%; Whisper speech linear/MLP 34.04%/36.48%; and ImageBind+Whisper linear/MLP 37.36%/39.97%. The paper's zero-shot GPT-4V scores range from 16.49% (vision) to 33.88% (vision+speech), while zero-shot LLaVA ranges from 4.60% to 15.61% across the reported prompts. Their temporal variants are listed in the preceding table. This confirms that fine-tuning and temporal fusion outperform many straightforward feature probes, but zero-shot language-vision models are not strong label classifiers in these experiments.

### 5.2 Earlier local AV-ASD experiments

The local experiments use different tasks and should not be merged into one leaderboard.

| Local experiment | Unit / labels | Accuracy | Macro-F1 | Caveat |
|---|---|---:|---:|---|
| Early saved audio XGBoost | 182 clips, four-class | 50.55% | 0.4191 | Not a verified replication of the paper's multimodal test model |
| Earlier four-class clip baseline | 182 clips | 47.25% | 0.4290 | Macro-F1 is higher than the later XGBoost result despite lower accuracy |
| Audio-enhanced four-class model | Clip experiment | 40.66% best individual model | 0.3503 | Did not improve the earlier benchmark |
| “Gated multimodal” saved result | Four-class clip task | 46.70% | 0.4144 | Saved script uses audio OOF probabilities and class-prior modulation, not a visual feature matrix; it is not true multimodal fusion |
| Engineered audio-feature ensemble | 1,131 annotated segments, four-class | 54.38% | 0.3522 | Segment-level score; cannot compare directly to clip-level paper result |
| Aggregated four-class model | Video level | 31.90% | 0.3267 | Video-level aggregation was weak |
| Historical clip experiment | 161 clips, three-class | 55.90% | 0.5576 | Saved in benchmark JSON; different task and protocol |
| Hyper-tuned engineered ensemble | 961 segments, three-class | 76.59% | 0.5370 | Pooled accuracy dominated by one 506-segment, single-class video |
| Aggregated three-class predictions | Video level | 54.02% | 0.5016 | Different target unit |

For the 961-segment result, class supports are 206 aggressive, 55 hyper-/hyporeactivity, and 700 upper-limb stereotypies. Class F1 values are 0.585, 0.168, and 0.858, respectively. The fold accuracy range is 41.23%–92.49%; fold macro-F1 ranges from 0.223 to 0.651. One 29-minute video supplies 506 of 961 segments, all with the upper-limb label. It is held out as its own fold, so the issue is extreme sample weighting by one source recording rather than duplicated segment IDs. The score is valid only as pooled segment accuracy for that split; it is not robust evidence of performance on a typical new video.

### 5.3 New audio-only experiments

#### Multi-label clip experiment

This experiment preserves the published ten-indicator task as far as the local files permit. The pipeline maps aligned VAD segments to annotated clips and creates a 259-dimensional clip vector from the 32-dimensional acoustic features: global mean, standard deviation, minimum, maximum, four ordered temporal-bin means, and segment/duration summaries. It uses one class-balanced logistic regression per behavior. Five outer folds keep clips from a source video together; a grouped validation partition within each outer training set selects regularization and per-label thresholds. The primary table uses a fixed 0.5 threshold because validation-selected thresholds scored lower.

| Metric | Local result |
|---|---:|
| Clips with usable aligned audio / source-video groups | 227 / 138 |
| Behavior macro-F1, fixed 0.5 threshold (nine behaviors) | 0.2811 |
| Behavior micro-F1, fixed 0.5 threshold | 0.3145 |
| Behavior macro average precision | 0.2202 |
| Ten-label exact-match rate, fixed threshold | 0.0176 |
| Behavior macro-F1 with fold-selected thresholds | 0.2688 |

Only 227 of 928 annotated clips had usable aligned audio features. Target-speaker identity is unverified; Background has only seven positive clips in this subset; and source-video IDs are not participant IDs. This is the closest local task definition to the paper, but it uses a partial audio-only subset and a different evaluation protocol. Its result cannot be compared numerically to the paper's 59.77% multimodal macro-F1 as if they were the same benchmark.

#### Video-context segment experiment

This pilot adds previous/current/next acoustic descriptors, recording-level mean and standard deviation, normalized temporal position, segment count, and duration. Model type and settings are selected using three-fold grouped validation inside each outer training fold; evaluation uses five outer GroupKFold folds by video.

| Metric | Result |
|---|---:|
| Pooled segment accuracy | 67.01% |
| Macro-F1 | 0.4662 |
| Balanced accuracy | 0.4838 |
| Equal-weight mean accuracy across 87 videos | 46.28% |
| Majority-class segment baseline | 72.84% |

This method does not beat the majority-class pooled accuracy baseline. Test-fold accuracy ranges from 4.39% to 92.49%, so performance is unstable. The result does not meet the 80% aim.

#### Video-level classifier

To prevent the 506-segment recording from contributing hundreds of votes, a separate dataset was formed with one pooled feature vector per video and the majority segment label as its target. A small set of logistic-regression, SVC, tree, and KNN models was selected with three-fold inner stratified CV and evaluated with five-fold stratified CV across the 87 videos.

| Metric | Result |
|---|---:|
| Video-level accuracy | 60.92% |
| Macro-F1 | 0.5492 |
| Balanced accuracy | 0.5527 |
| Majority-video class baseline | 52.87% |
| Fold accuracy range | 50.00%–76.47% |

This result exceeds its baseline but does not reach 80%. It predicts a video-level majority behavior label, not ten simultaneous labels. Participant identities are unverified, and the target is derived from segment labels.

### 5.4 PPO adaptive segment sampling

The AV-ASD PPO sampler uses descriptor novelty as a reward proxy and an eight-segment budget; it does not learn a clinical or treatment policy. In the saved five-fold video-level evaluation, uniform sampling reached 62.22% accuracy / 0.5067 macro-F1, random 59.87% / 0.4992, novelty heuristic 60.98% / 0.5033, and PPO 60.98% / 0.4941. PPO did not improve over uniform sampling, and the method remains audio-only.

### 5.5 AV-ASD data and modality gap

No AV-ASD raw video, extracted frames, pose/keypoint features, or speech embeddings were found in the inspected local locations. The prepared MEAT-GNN script expects MediaPipe keypoints under `data/processed/keypoints/av_asd/`, but that directory is absent. Accordingly, no local AV-ASD graph or audio-visual fusion result has been measured. Any text calling the saved audio-prior model “multimodal” should be corrected.

## 6. Comparison with the published work

### 6.1 MMASD comparison

| Study / experiment | Task and protocol | Accuracy | Macro-F1 | Comparison status |
|---|---|---:|---:|---|
| Recent RC-ProtoGCN, pooled result | 10-class 2D pose; 5-fold subject-independent | 89.29% | 0.8837 | Strongest strict published reference; 1,202 clips / 32 subjects |
| ProtoGCN baseline | Same paper, subject-independent | 87.73% | 0.8703 | Baseline for RC regularizer |
| CTR-GCN baseline | Same paper, subject-independent | 88.46% | 0.8796 | Re-trained comparison |
| RC-ProtoGCN | 80/20 holdout | 96.30% | 0.9594 | Easier protocol; do not compare with subject-held-out CV as if matched |
| BiLSTM + attention, as reported in the recent comparison paper | 10-class 2D; 80/20 holdout | 94.84% | 0.9400 | Holdout comparison; reported session-split reimplementation drops to 83.13% accuracy |
| Local advanced ensemble | 11-class processed features; 5-fold stored-group CV | 82.87% | 0.7912 | Different labels and inputs; stored IDs unverified |
| Local 2D OpenPose RBF pilot | 10-class local archive; 1,374 clips; stored filename groups | 71.69% fold mean | 0.6394 fold mean | Archive count/distribution mismatches paper; not matched |

The recent RC-ProtoGCN paper reports 5-fold subject-independent pooled accuracy 89.29% and macro-F1 0.8837, compared with base ProtoGCN's 87.73% and 0.8703. Its method targets the pairs most often confused during training by adding a training-only Refine–Confusable margin regularizer; the test-time backbone remains ProtoGCN. The local 82.87% result is not a direct competitor because it uses 11 labels and a different representation. The local 2D pilot is closer in modality and number of classes, but its clip count and class distribution do not match the published setup.

For completeness, the same article's protocol table reports:

| Protocol | ProtoGCN accuracy / macro-F1 | RC-ProtoGCN accuracy / macro-F1 |
|---|---:|---:|
| Random 5-fold | 94.99% / 0.9459 | 95.30% / 0.9486 |
| Session-independent 5-fold | 88.82% / 0.8809 | 90.03% / 0.8933 |
| Subject-independent 5-fold | 87.73% / 0.8703 | 89.29% / 0.8837 |
| 80/20 holdout | — | 96.30% / 0.9594 |

The article also reports 94.84% accuracy / 0.940 macro-F1 for the BiLSTM-attention paper under an 80/20 holdout, and 83.13% accuracy for a session-split reimplementation of that model. The difference illustrates how the split affects the apparent result; the 94.84% holdout should not be compared directly to a participant-held-out result.

### 6.2 MMASD learning-dynamics paper comparison

The 2026 learning-dynamics paper reports unsupervised discovery of six behavioral primitives from 255 pose/optical-flow windows in 15 therapy sessions, clustering stability above ARI 0.8, transition-graph analyses, saliency, and a KL/joint-attention correlation of approximately −0.963. The local PCA and Transformer experiments operate on 7,049 windows from 1,661 recordings with 64 kinematic channels and do not have the selected 15 sessions or joint-attention values. Local temporal-autoencoder ARI stability is 0.6203; PCA cluster-seed stability is 0.7422. These differences make this an adaptation and exploratory comparison, not a claim to have reproduced or surpassed the paper.

### 6.3 AV-ASD comparison

The AV-ASD paper's main reported metric is macro-F1 on its ten-label task. Its 59.77% LLaVA-ASD result is the strongest result in the cited baseline/modality tables; ImageBind+Whisper temporal fusion reports 53.17%. The local 3-class segment accuracy of 76.59% is a different metric/task and is dominated by one long video. The local 10-label audio-only result is macro-F1 0.2811 on 227 available clips. Neither supports a statement that the project outperforms the paper. A proper comparison would require the same 928 clips, their multi-label targets, aligned audio/video/speech features, and a documented split at least as strict as the paper's—and preferably source-video- and participant-disjoint evaluation.

## 7. What is and is not novel so far

The evaluated ideas include prototype-distance features for the MMASD ensemble; an unsupervised temporal-autoencoder adaptation; a quality-aware graph with synthetic missing-body-group robustness; a confidence-aware graph pilot; temporal-context pooling for AV-ASD; video-level AV-ASD pooling; and PPO-based sampling. Testing an idea is not by itself a novel contribution.

So far, the evidence does **not** show that the quality-aware graph, confidence-aware graph, temporal-context features, or PPO sampler beats a strong matched baseline. The advanced MMASD ensemble is the best saved local supervised result, but its component-level novelty has not been isolated. The strongest defensible contribution at this stage is an audited comparison that exposes task, group-size, modality, and split limitations, plus the implementation of candidate approaches. A thesis claim of improved state of the art would be premature.

## 8. Progress against the 80% goal

- **MMASD:** 82.87% accuracy is reached on the local 11-class stored-group feature benchmark. This does not equal the stricter recent 10-class 2D subject-independent graph benchmark, where published RC-ProtoGCN reports 89.29% pooled accuracy. The local 2D pilot is 71.69% by fold mean.
- **AV-ASD:** No robust 80% result is reached. The best-seeming 76.59% segment accuracy is dominated by one source video; the nested video-level result is 60.92% accuracy, and the closest multi-label audio result is 0.2811 macro-F1.
- Do not try to reach 80% by choosing a favorable fold, changing label priority after seeing test results, excluding difficult videos, or comparing accuracy from one task with F1 from another.

## 9. Recommended next steps

1. **MMASD:** Reconcile the local labels, feature set, and split with the recent 1,202-clip 10-class paper benchmark. Verify participant IDs. Re-run a protocol-matched subject-independent comparison between the local model, a published baseline, and the proposed contribution. Add repeated seeds and paired fold-level analysis.
2. **AV-ASD:** Recover or obtain authorized aligned video/pose and speech features, plus source-video and participant IDs. Rebuild the ten-label task on the same clip set and evaluate with source-video/participant grouping. Keep audio-only, visual-only, speech-only, and fusion ablations.
3. **Novel method claim:** Pre-register one specific addition, such as a reliability-aware fusion/gating method or a calibrated abstention strategy for missing modalities. Compare it against the same fold-local baselines and ablate each component. Treat any gain as preliminary until repeated and externally validated.
4. **Reinforcement learning:** Do not frame PPO as therapy-policy learning without actions and clinical outcomes. It can remain a data-selection or acquisition-policy experiment with equal annotation budgets and a clear utility target.

## 10. Main reproducibility artifacts

| Area | Main artifacts |
|---|---|
| MMASD advanced supervised benchmark | `experiments/mmasd_advanced_90_results_2026_11_12.json`; `results/experiment_inventory_2026_11_12.csv` |
| MMASD unsupervised PCA and Transformer | `experiments/mmasd_dynamics_replication.py`; `experiments/mmasd_temporal_autoencoder_replication.py`; `results/mmasd_dynamics_replication_2026_11_12.json`; `results/mmasd_temporal_autoencoder_2026_11_12.json` |
| MMASD quality-aware graph | `models/quality_aware_stgcn.py`; `experiments/mmasd_quality_aware_label_efficiency.py`; `results/mmasd_quality_aware_label_efficiency_2026_11_12.json` |
| MMASD local 2D archive | `experiments/mmasd_openpose_quality_graph.py`; `experiments/mmasd_openpose_rbf_kernel.py`; `results/mmasd_openpose_archive_audit_2026_11_12.json` |
| MMASD / AV-ASD PPO samplers | `experiments/mmasd_adaptive_sampling_rl.py`; `experiments/avasd_adaptive_sampling_rl.py`; corresponding `results/*adaptive_sampling_rl_2026_11_12.json` |
| AV-ASD multi-label audio | `experiments/avasd_multilabel_audio_grouped.py`; `results/avasd_multilabel_audio_grouped_2026_11_12.json` |
| AV-ASD segment/video context | `experiments/avasd_video_context_nested.py`; `experiments/avasd_video_level_nested.py`; corresponding JSON, fold, OOF, and per-video CSV files in `results/` |
| Project-level score records | `results/benchmarks_2026_11_12.json`; `results/benchmarks_summary_2026_11_12.md`; `results/dataset_scorecard_2026_11_12.csv`; `results/experiment_inventory_2026_11_12.csv` |

## References

1. Deng et al., “Hear Me, See Me, Understand Me: Audio-Visual Autism Behavior Recognition,” *IEEE Transactions on Multimedia*, 2025. [IEEE record](https://ieeexplore.ieee.org/abstract/document/10812841); [author manuscript with result tables](https://arxiv.org/html/2406.02554).
2. Ay et al., “Skeleton-Based Activity Recognition for Children with Autism Using Graph Convolutional Networks,” *Sensors*, 2026. [Full paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC13431331/).
3. Kassir et al., “A BiLSTM with Attention Approach Using 2D Skeletons for Autism Activity Recognition,” ICCA 2025. [DOI](https://doi.org/10.1109/ICCA66035.2025.11431081). Results and protocol comparison are also summarized in Ay et al. (2026).
4. Akinwumi, Qian, and Ojo, “Modelling learning dynamics in autism therapy through explainable multimodal representation learning,” *Frontiers in Neuroscience*, 2026. [Full paper](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2026.1737098/full); [PubMed record](https://pubmed.ncbi.nlm.nih.gov/42523732/).

