Overview: Source tree for training, pruning, distillation, inference, and analysis of vision transformers for turbulence forecasting.

Usage: Run the `script_*.py` entry points with Python using configurations from `config/`; submit jobs via `jobs/`; import modules from `models/`, `trainers/`, and `inference/` as needed.

Contents:
- trainers/: Training, pruning, and distillation loops plus trainer utilities.
- jobs/: HPC batch scripts for submitting experiments to a scheduler.
- notebooks/: Jupyter notebooks for exploration and visualization.
- models/: Model definitions, including the vision transformer and submodules.
- inference/: Inference, rollout, metrics, and visualization utilities.
- config/: TOML/YAML configuration files for training, pruning, and inference.
- data/: Dataset wrappers and loading utilities.
- __init__.py: Marks this directory as a Python package.
- script_prune_unstructured.py: Entry point to run unstructured pruning.
- script_prune_attention_heads.py: Entry point to prune transformer attention heads.
- script_finetune_l1.py: Entry point to fine-tune with L1 regularization for sparsity.
- script_short_analysis.py: Entry point to run short-horizon analysis/evaluation.
- script_train.py: Entry point to train a model from scratch or resume.
- script_distill.py: Entry point to run knowledge distillation.
- script_make_inference.py: Entry point to run inference/rollouts over datasets.
- script_long_analysis.py: Entry point to run long-horizon analysis/evaluation.
- script_make_gif.py: Entry point to create GIF visualizations of rollouts.
