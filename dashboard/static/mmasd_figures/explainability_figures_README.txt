MMASD Temporal GNN explainability figure outputs
================================================

Fig1_top15_joint_importance:
Top 15 joint indices ranked by the supplied attribution magnitude.

Fig2_temporal_importance:
Global attribution across the eight model-input temporal steps.

Fig3_schematic_71_joint_index_map:
Index inventory only. Grid locations are arbitrary and MUST NOT be
interpreted as anatomical joint positions or skeletal connectivity.

Fig4_combined_explainability:
Two-panel summary of joint and temporal importance.

Interpretation limitations:
- Gradient x input is an attribution method, not causal evidence.
- Global summaries do not demonstrate clinical relevance.
- The attribution aggregation is over valid detections; detections
  have not been verified as consistent person tracks.
- Verify the 71-joint convention before labeling body parts.
- Temporal indices 0-7 are not seconds unless sampling is verified.
- The values represent the loaded CSV files when present, otherwise
  the explicitly embedded fallback values.