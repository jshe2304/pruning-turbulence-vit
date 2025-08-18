Overview: Helper utilities shared across training, pruning, and distillation workflows.

Usage: Imported by trainer scripts and loops; extend these utilities to add new pruning criteria or training logic.

Contents:
- compute_loss.py: Loss computation helpers used during training and evaluation.
- importance_scores.py: Importance score calculations for pruning.
- structured_pruning.py: Implementations for structured pruning operations.
- train.py: Utility functions used within training loops (e.g., logging, steps).
