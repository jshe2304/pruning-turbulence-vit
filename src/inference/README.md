Overview: Utilities for inference, rollouts, metrics, and visualization for short and long analyses.

Usage: Called by top-level `script_make_inference.py`, `script_short_analysis.py`, `script_long_analysis.py`, and `script_make_gif.py`; can also be imported from trainers or notebooks.

Contents:
- __init__.py: Package marker for inference utilities.
- io_utils.py: I/O helpers for loading/saving predictions, configs, and artifacts.
- long_analysis_all.py: Orchestrates multi-run long-horizon analyses across datasets or configs.
- long_analysis.py: Runs long-horizon rollout evaluation and aggregates results.
- long_metrics.py: Metric computations specific to long-horizon evaluation.
- make_gif.py: Creates GIF visualizations from rollouts and fields.
- make_inference.py: Runs inference/rollouts and writes outputs for downstream analysis.
- plot_config.py: Plotting configuration and styles for figures and GIFs.
- rollout.py: Core rollout utilities for stepping the model over time.
- short_analysis.py: Runs short-horizon rollout evaluation and aggregates results.
- short_metrics.py: Metric computations specific to short-horizon evaluation.
