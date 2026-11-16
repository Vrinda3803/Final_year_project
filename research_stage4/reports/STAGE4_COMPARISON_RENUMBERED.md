# Stage 4 Comparison Table (Renumbered)

Display IDs follow the requested numbering. Source run IDs are shown only to help locate the saved experiment files. The RBF baseline is omitted. One model configuration was run twice with identical settings; its result is listed once.

| Display ID | Source run | Configuration | Mean accuracy | Fold SD | Pooled macro-F1 | Mean ECE |
|---|---|---|---:|---:|---:|---:|
| A0 | A0 | Fixed bilateral topology | 57.12% | ±5.88% | 0.5428 | 0.1370 |
| A1 | A2 | Fixed topology, reliability only | 57.12% | ±5.88% | 0.5428 | 0.1370 |
| A2 | A3 | Full RCDB-STGT | 60.40% | ±11.65% | 0.5947 | 0.1109 |
| A3 | A4 | RCDB-STGT without temporal smoothing | 59.53% | ±4.50% | 0.5736 | 0.1074 |
| A4 | A5 | RCDB-STGT with phase stream | 60.55% | ±6.40% | 0.5942 | 0.0817 |
| A5 | A6 | RCDB-STGT with wrists only | 61.28% | ±4.26% | 0.5872 | 0.0825 |
| A6 | A7 | RCDB-STGT with five bilateral pairs | 58.95% | ±11.32% | 0.5728 | 0.0919 |
| A7 | A8 | Dynamic bilateral connections with reliability input | 65.92% | ±6.61% | 0.6402 | 0.0733 |
| A8 | A9 | RCDB-STGT with confusable-class margin loss | 61.22% | ±2.25% | 0.5878 | 0.0838 |

Accuracy and standard deviation are calculated across the five cohort-grouped folds. Pooled macro-F1 is computed from all out-of-fold predictions. ECE is the mean expected calibration error across folds.

## Brief interpretation

- Display A7, which learns dynamic bilateral connections and uses joint reliability as an input, has the highest mean accuracy (65.92%) and pooled macro-F1 (0.6402) in this table. An identical configuration was run under a second experiment ID; that duplicate is not counted as an additional result.
- Display A2 (source A3), the full RCDB-STGT model, scores 60.40% mean accuracy and 0.5947 pooled macro-F1.
- The results describe this dataset and cohort-grouped evaluation protocol. Cohort identifiers are not verified participant identifiers.
