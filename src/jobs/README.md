Overview: Batch job scripts for submitting training, pruning, inference, and analysis runs on an HPC cluster.

Usage: Adjust paths/modules as needed, then submit with your scheduler (e.g., `qsub train.pbs` or `sbatch train.pbs`).

Contents:
- distill.pbs: Submits a knowledge distillation job to train a student from a teacher model.
- finetune_l1.pbs: Submits an L1-regularized fine-tuning job to encourage sparsity.
- long_analysis.pbs: Submits a long-horizon evaluation/analysis job on extended rollouts.
- make_gif.pbs: Submits a job to generate GIF visualizations from saved rollouts.
- make_inference.pbs: Submits a batch inference job to produce predictions/rollouts.
- prune_attention_heads.pbs: Submits a pruning job targeting transformer attention heads.
- prune_unstructured.pbs: Submits an unstructured (magnitude-based) pruning job.
- short_analysis.pbs: Submits a short-horizon evaluation/analysis job on shorter rollouts.
- train.pbs: Submits a standard training job for the vision transformer model.
