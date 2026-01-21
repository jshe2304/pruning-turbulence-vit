Overview: Utilities for inference, rollouts, metrics, and visualization for short and long analyses.

Usage: Called by top-level scripts in `scripts/`; can also be imported from trainers or notebooks.

Contents:
- __init__.py: Package marker for inference utilities.
- long_analysis.py: Runs long-horizon rollout evaluation and aggregates results.
- make_gif.py: Creates GIF visualizations from rollouts and fields.
- make_inference.py: Runs inference/rollouts and writes outputs for downstream analysis.
- short_analysis.py: Runs short-horizon rollout evaluation and aggregates results.

utils/:
- io_utils.py: I/O helpers for loading/saving predictions, configs, and artifacts.
- long_metrics.py: Metric computations specific to long-horizon evaluation.
- plot_config.py: Plotting configuration and styles for figures and GIFs.
- rollout.py: Core rollout utilities for stepping the model over time.
- short_metrics.py: Metric computations specific to short-horizon evaluation.
