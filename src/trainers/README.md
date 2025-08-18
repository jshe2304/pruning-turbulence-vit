Overview: Training, pruning, and distillation loops plus supporting utilities for vision transformer experiments.

Usage: Called by the top-level `script_*.py` entry points; import trainer modules directly to build custom pipelines; see `utils/` for common helpers.

Contents:
- __init__.py: Package marker for trainer modules.
- distill.py: Training loop for knowledge distillation from a teacher model.
- finetune.py: Standard fine-tuning routines for pretrained checkpoints.
- finetune_l1.py: Fine-tuning with L1 regularization to promote sparsity.
- prune_attention_heads.py: Routines to prune transformer attention heads.
- prune_unstructured.py: Routines for unstructured/magnitude-based pruning.
- train.py: Standard supervised training loop and orchestration.
- utils/: Shared training utilities (losses, importance scores, structured pruning, helpers). 