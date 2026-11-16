# Physical Therapy Activity Recognition from Skeleton Sequences

This repository contains the research code, experiment records, visualizations, and report materials for a study of physical therapy activity recognition from two-dimensional OpenPose skeleton sequences in the local MMASD dataset.

The final ten-class experiments compare graph-based model configurations, including the Reliability-Conditioned Dynamic Bilateral Spatio-Temporal Graph Transformer (RCDB-STGT). Evaluation uses five folds grouped by the available recording-cohort identifiers. Those identifiers have not been verified as unique participant identities, so results are described as cohort-grouped evaluation.

The activity labels represent recorded therapy activities. The project does not diagnose autism, assess social behaviour, or establish clinical effectiveness.

## Repository contents

- `research_stage4/`: final ten-class experiment code, configurations, metrics, predictions summaries, reports, and figures.
- `research_stage2/` and `research_stage3/`: earlier research experiments and supporting materials.
- `models/`, `experiments/`, `evaluation/`, and `dashboard/`: model, experiment, evaluation, and demonstration code.
- `paper/`, `REFERENCES.bib`, and the report files in the repository root: research and report materials.

## Dataset and local setup

The raw MMASD archives, processed skeleton arrays, and local dependency bundle are not included in this repository. The source archives are large, and the recordings and derived pose data should only be redistributed when their data-use terms and privacy requirements explicitly allow it. Place the authorized dataset files in the paths expected by the relevant configuration or update those paths locally before running an experiment.

The final experiments use processed two-dimensional OpenPose sequences with 64 frames, 25 BODY_25 joints, four pose/motion channels, and separate joint-confidence values. Data and model checkpoints are excluded from version control by `.gitignore`.

## Reproducibility

Experiment-specific commands, configuration names, data paths, and evaluation details are documented alongside the scripts in `research_stage4/`. Review the relevant configuration before running a command, because dataset paths and available dependencies are machine-specific. Checkpoints and scratch outputs are not tracked.

## Interpretation

Reported performance applies to the dataset configuration and cohort-grouped evaluation used in this project. It should not be interpreted as participant-independent clinical performance or as an autism assessment result.
