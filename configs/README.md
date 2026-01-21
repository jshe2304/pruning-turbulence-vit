Overview: Centralized configuration for training, pruning, and inference.

Usage: Point entry scripts to a TOML file in `train/`, `prune/`, or `inference/`; adjust `hyperparams.yaml` or TOMLs to match your experiment.

Contents:
- hyperparams.yaml: Global/default hyperparameters referenced by scripts or TOMLs.
- inference/: TOML files for inference, analysis, and visualization runs.
- prune/: TOML files that configure pruning strategies and parameters.
- train/: TOML files that configure model/training settings and schedules.
