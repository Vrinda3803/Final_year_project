# MMASD report insertion guide

This guide maps the copy-ready LaTeX blocks in `mmasd_chapter4_insertions.tex` to the S7 report. It does not modify the report or any non-MMASD section. The existing report already contains MMASD text, so the main Chapter 4 result paragraph and tables should be replaced with the updated MMASD material rather than duplicated.

## Where to paste each block

| S7 report location | What to do |
|---|---|
| Chapter 4, Section 4.2 `Datasets`, subsection `Multimodal Dataset for Autism Intervention Analysis (MMASD)` | Replace the existing MMASD dataset description. Replace the old Table 4.3 fixed train/validation/test split with the five-fold table in the insertion file if this section describes the current results. Leave SSBD, ASBD, and AV-ASD sections unchanged. |
| Chapter 4, Section 4.4.1 `Experimental Scenarios` | Add the supplied Scenario 5 after the existing MMASD graph-temporal scenario. |
| Chapter 4, Section 4.4.2 `Parameter Tuning` | Add the paragraph on nested CV and the advanced ensemble after the old MMASD GNN settings. |
| Chapter 4, Section 4.5 `Results`, subsection `MMASD Results` | Replace the old MMASD model comparison table and its immediate performance paragraphs. Also replace the old Table 4.7 class-wise GNN scores and the paragraph interpreting them. Keep the existing SSBD and AV-ASD result sections. |
| Chapter 4, Section 4.5, after the MMASD model comparison | Add the outer-fold table. |
| Chapter 4, Section 4.5, after the fold table | Add the new advanced-model class-wise table in place of the old Table 4.7 and its accompanying class-wise interpretation. |
| Chapter 4, Section 4.5, after the class-wise table | Add the ablation-status table and explanation. It explicitly distinguishes measured comparisons from a formal ablation. |
| Chapter 4, Section 4.5, existing `Explainability Results` subsection | Replace only the MMASD explainability paragraph with the supplied MMASD explainability subsection. Keep the other dataset results intact. |
| Chapter 4, Section 4.6 `Analysis of Results` | Add the MMASD analysis subsection while retaining analysis of the other datasets. |
| Chapter 4, Section 4.7 `Observations` | Add the five MMASD-specific `\\item` entries inside the existing `enumerate` list. Do not paste them outside the list. |
| Chapter 4, Section 4.8 `Comparative Analysis` | Replace only the three MMASD rows in the existing overall comparison table with the supplied nested RF, nested temporal-feature RF, and advanced ensemble rows. Keep SSBD and AV-ASD rows unchanged, then add the supplied paragraph below the table. |
| Abstract and Chapter 5, Section 5.1 `Conclusion` (optional) | The insertion file now provides an updated MMASD conclusion passage. Replace only the older MMASD passage that reports 76.44% as the main result; retain the rest of the conclusion. |
| Chapter 5, subsection `Comparative Evaluation and Model Validation` | Replace only the sentence claiming the temporal GNN outperformed the Random Forest baselines with the updated paragraph. It distinguishes the newer best result from the older fixed-split GNN result. |
| Chapter 5, subsection `Enhancement of Explainability` | Clarify that the existing attribution results apply to the earlier temporal GNN and have not been evaluated for the advanced multi-bank ensemble. |

The insertion file labels each block with its location. It uses plain LaTeX tables so it does not require adding a package to the report preamble. Copy one block at a time and compile after insertion; the report source uses `\\input{chapters/chapter4}`, but that chapter source was not included with the pasted template.

The parameter tables belong in Chapter 4, Section 4.4.2. In Chapter 5, summarize the best result and its limits; do not repeat the full parameter tables there.

The old 1,153 / 332 / 208 table is not necessarily wrong; it describes the earlier fixed split used for the graph-temporal experiment. If you want to preserve that history, move or relabel it under Scenario 4 as the historical graph-temporal split. Do not present it as the partition behind the newer five-fold results.

## What the experiments show

- The best saved completed MMASD result is **82.87% out-of-fold accuracy and 0.7912 macro-F1**, from the advanced multi-bank feature pipeline and weighted RF / ExtraTrees / XGBoost ensemble.
- The advanced pipeline uses 1,248 multi-bank features plus 506 rich features, selects 800 within each outer training fold, and appends 11 training-fold prototype distances. The model input after selection and prototype augmentation is 811 dimensions; 1,754 is the pre-selection concatenated feature dimension.
- The nested pooled Random Forest result is **81.10% / 0.7716**. The nested pooled-versus-four-bin temporal comparison is **80.98% / 0.7688**, so the tested four-bin representation did not improve the baseline.
- The saved 82.04% ensemble is included as an exploratory comparator. Its script computed class weights from the full label vector before splitting and tuned blend weights against the full out-of-fold predictions; repeat both choices inside nested training folds before treating its score as an unbiased estimate.
- A formal ablation is not present. The advanced model changes several components at once. The insertion text reports it as a combined-pipeline comparison and proposes one-component-at-a-time reruns for a future controlled ablation.
- The 76.44% / 0.7530 historical GNN result is kept distinct because its earlier fixed split is not matched to the later subject-grouped five-fold experiments.
- Class-wise F1 is strongest for twist pose (0.947), frog pose (0.905), and drumming (0.901), and weakest for squat (0.560) and maracas forward shaking (0.598).

## Parameter table and visual material

The insertion file now includes two parameter tables under Section 4.4.2: the exact saved configuration of the strongest 82.87% pipeline, and the fold-specific choices from the nested temporal-feature Random Forest experiment. The first table records settings, not a claim that an exhaustive hyperparameter search found a global optimum.

The project already contains joint and temporal attribution figures (`dashboard/static/mmasd_figures/Fig1_top15_joint_importance.png` through `Fig4_combined_explainability.png`). The insertion file includes an optional LaTeX block for the combined figure. It belongs under Explainability Results and describes the earlier temporal GNN only; do not present it as an explanation of the advanced ensemble. These are the only saved MMASD figures located. The project does not currently contain a performance chart for the newer five-fold results. A fold-wise accuracy/macro-F1 plot or a per-class F1 bar chart could be made directly from the saved results, but should be labeled as a new visualization of those saved numbers.

## Keep these qualifications

1. Use `subject-grouped GroupKFold`, not “stratified GroupKFold”; GroupKFold is not stratified.
2. Describe MMASD as eleven-class structured activity recognition. Its labels are intervention activities, not diagnoses or direct measures of social communication.
3. Do not call the 88.60% peak-fold accuracy the overall score. The aggregate out-of-fold accuracy is 82.87%.
4. The saved data uses 32 stored subject groups. Verify that those IDs represent distinct participants before claiming population-level participant independence.
5. The saved explainability plots rank joint indices and temporal positions. Verify the 71-joint convention and track identity before assigning anatomy or clinical meaning. Temporal bins are not seconds unless sampling is confirmed.
6. Do not claim three random seeds, confidence intervals, statistical significance, or a formal ablation; the current MMASD artifacts do not establish them.

## Keep the report sections separate

The S7 PDF and pasted Amrita report template use Chapter 4 sections 4.2 through 4.8. The project file `paper/paper_draft.tex` is a separate IEEE-style manuscript, not the LaTeX source for the supplied S7 template. If updating that article later, the MMASD insertion point is its `\\subsection{MMASD 11-Class Activity Recognition}` under `\\section{Results}`; do not paste the Chapter 4 block structure into the IEEE manuscript.

