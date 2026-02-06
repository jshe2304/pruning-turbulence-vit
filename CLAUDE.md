# CLAUDE.md

## Project Overview

Research project investigating how pruning (and model compression more generally) affects Vision Transformers (ViTs) trained for 2D turbulence fluid dynamics emulation. 

This codebase supports modular implementations of training, pruning, and evaluation procedures. 

## Package organization

### `src/models/`
- `vision_transformer.py`: Encoder-decoder Vision Transformer
- `modules/`: Implementations of attention, mlp, embeddings, positional encodings, conv (SubPixel 3D decoder)

### `src/training/`
- `train.py`: Standard supervised training loop with DDP
- `prune_unstructured.py`: Iterative prune-finetune loop with importance scores (L1, Fisher, Taylor, random)
- `prune_attention_heads.py`: Structured attention head pruning
- `distill.py`: Knowledge distillation via hidden layer MSE matching
- `utils/`: Helper functions as well as importance score implementations

### `src/inference/`
- `make_inference.py`: Generate rollout predictions
- `short_analysis.py`: Analyses for short leadtime trajectories
- `long_analysis.py`: Analyses for long leadtime trajectories
- `make_gif.py`: Generate GIF animation of rollout

### `src/data/`
- `datasets`: PyTorch dataset for 2D turbulence frames with configurable stride/steps

## Auxiliary code
- `configs/`: TOML files for running training, pruning, etc. jobs
- `notebooks/`: analysis and visualization notebooks
- `jobs/`: PBS scripts for running compute jobs on cluster
- `scripts/`: Python scripts setting up and running routines from `src/`

## Code style

This is a research project, so code should be amenable to rapid iteration and experimentation. 
- Prioritize simple, readable implementations. 
- Don't be too defensive, avoid try-except unless truly necessary. 
- Prefer shallow dependency trees. 