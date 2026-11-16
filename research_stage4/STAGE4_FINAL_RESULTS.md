# Stage 4 Final Results: RCDB-STGT

**Status:** A0–A9 training and automated post-processing completed. **A8 is not an independent experiment**: inspection of the implementation shows A1 and A8 both instantiate the model with `use_reliability_gate=False`, and reliability is supplied to the edge learner in both configurations. Their matching scores should not be interpreted as an independent replication.

## Ablation results

| ID | Configuration | Mean fold accuracy | Fold SD | Pooled OOF macro-F1 | Mean ECE |
|---|---|---:|---:|---:|---:|
| A0 | Fixed bilateral topology (E2 reproduction) | 57.12% | ±5.88% | 0.5428 | 0.1370 |
| A1 | Dynamic bilateral; no multiplicative gate | 65.92% | ±6.61% | 0.6402 | 0.0733 |
| A2 | Fixed topology with reliability only | 57.12% | ±5.88% | 0.5428 | 0.1370 |
| A3 | Full RCDB-STGT (dynamic + reliability gate + smoothing) | 60.40% | ±11.65% | 0.5947 | 0.1109 |
| A4 | RCDB-STGT without temporal smoothing | 59.53% | ±4.50% | 0.5736 | 0.1074 |
| A5 | RCDB-STGT with phase stream | 60.55% | ±6.40% | 0.5942 | 0.0817 |
| A6 | RCDB-STGT with wrists only | 61.28% | ±4.26% | 0.5872 | 0.0825 |
| A7 | RCDB-STGT with five bilateral pairs | 58.95% | ±11.32% | 0.5728 | 0.0919 |
| A8 | Configured identically to A1; not an independent ablation | 65.92% | ±6.61% | 0.6402 | 0.0733 |
| A9 | RCDB-STGT with confusable-class margin loss | 61.22% | ±2.25% | 0.5878 | 0.0838 |

A0–A9 use the same five cohort-grouped folds and seed 42. Accuracy and standard deviation are computed across folds; pooled macro-F1 uses the combined out-of-fold predictions. ECE is the mean fold expected calibration error.

## Selected statistical comparisons

These are the comparisons present in the saved statistical output; they are not an exhaustive pairwise test of every ablation. Fold-level tests use only five observations and should be interpreted cautiously. Clip-level tests use the pooled held-out predictions.

| Comparison | Mean accuracy difference | Fold t-test p | McNemar p | Bootstrap 95% CI for accuracy difference |
|---|---:|---:|---:|---:|
| A0 → A3 | +3.28 pp | 0.5298 | 0.0153 | [+0.73, +5.97] pp |
| A0 → A1 | +8.81 pp | 0.0080 | 1.01e-11 | [+6.33, +11.28] pp |
| A1 → A3 | -5.52 pp | 0.2343 | 1e-07 | [-7.42, -3.49] pp |
| A2 → A3 | +3.28 pp | 0.5298 | 0.0153 | [+0.73, +5.97] pp |
| A4 → A3 | +0.87 pp | 0.8795 | 0.472 | [-1.53, +3.35] pp |
| A6 → A7 | -2.33 pp | 0.6832 | 0.112 | [-4.88, +0.44] pp |
| A8 → A3 | -5.52 pp | 0.2343 | 1e-07 | [-7.42, -3.49] pp |

## Sensor corruption benchmark

The robustness benchmark covers A0, A1, and A3. These are zero-shot perturbations applied at evaluation time.

| Model | Condition | Accuracy | Macro-F1 |
|---|---|---:|---:|
| A0 | Clean | 57.12% | 0.4905 |
| A0 | Random 20% joint dropout | 55.38% | 0.4621 |
| A0 | Random 40% joint dropout | 49.63% | 0.3946 |
| A0 | Both wrists occluded | 38.91% | 0.2732 |
| A0 | Both ankles occluded | 48.19% | 0.3798 |
| A0 | 15-frame blackout | 52.35% | 0.4303 |
| A1 | Clean | 65.92% | 0.5876 |
| A1 | Random 20% joint dropout | 60.49% | 0.5329 |
| A1 | Random 40% joint dropout | 51.11% | 0.4135 |
| A1 | Both wrists occluded | 44.79% | 0.3936 |
| A1 | Both ankles occluded | 59.09% | 0.4884 |
| A1 | 15-frame blackout | 59.26% | 0.5082 |
| A3 | Clean | 60.40% | 0.5405 |
| A3 | Random 20% joint dropout | 56.77% | 0.5037 |
| A3 | Random 40% joint dropout | 49.86% | 0.4293 |
| A3 | Both wrists occluded | 41.27% | 0.3616 |
| A3 | Both ankles occluded | 55.93% | 0.4724 |
| A3 | 15-frame blackout | 56.24% | 0.4888 |

## Figures

Publication figures are in `research_stage4/visualizations/`: `fig1_ablation_matrix_benchmark.png`, `fig2_robustness_stress_comparison.png`, `fig3_dynamic_bilateral_heatmap_drumming.png`, `fig3_dynamic_bilateral_heatmap_tree_pose.png`, and `fig4_confusion_matrices_comparison.png`.

## Interpretation notes

- The highest mean accuracy among these runs is A1/A8 at 65.92%; A8 duplicates A1 in the current implementation, so this is one configuration, not two independent confirmations.
- A3 is the full reliability-gated RCDB-STGT model and has 60.40% mean fold accuracy and 0.5947 pooled OOF macro-F1.
- A5–A9 results are specific to this dataset and grouped-fold protocol. They do not establish clinical effectiveness or participant-independent generalization.
- Check implementation and split assumptions before making claims from these results. In particular, cohort IDs are not verified participant IDs.
