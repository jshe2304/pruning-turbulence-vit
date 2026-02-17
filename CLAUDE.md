# CLAUDE.md

## Project Overview

Research project investigating how pruning (and model compression more generally) affects Vision Transformers (ViTs) trained for 2D turbulence fluid dynamics emulation. 

This codebase supports modular implementations of training, pruning, and evaluation procedures. 

## Project data

The datas used for model training are fluid emulations computed using the py2d package, which is installed at @/glade/u/home/jshen/py2d. The data is stored in .mat files in either @/glade/derecho/scratch/jshen/2DTurbData/results or @/glade/derecho/scratch/dpatel/2DTurbData/results. The file tree is organized so that the path specifies the parameters of the run. 

For example, @/glade/derecho/scratch/dpatel/2DTurbData/results/Re5000_fkx0fky4_r0.1_b20/NoSGS/NX256/dt0.0002_IC1 corresponds to a run with:
- Reynold's number = 5000
- Forcing numbers kx = 0, ky=4
- Friction coefficient = 0.1
- Coriolis coefficent = 20
- NoSGS
- Grid size of 256 pixels
- Timestep 0.0002
- Initial condition 1

## Project workflow

The pipeline is as follows:
1. Train model
2. Prune model (optional)
3. Run short analysis
4. Run inference
5. Run long analysis (needs inference)

All outputs should be in run-specific directories in @/glade/derecho/scratch/jshen/pruning-turbulence-vit
- New training/pruning runs accept a config file, which specifies the output directory
- New runs save a copy of their config file into the output directory
- To resume a training/pruning run, pass in the output directory so the program can identify the config file and resume the run properly. 

## Output file structure

Ideally, the outputs of training and pruning runs obey the following structure under root `/glade/derecho/scratch/jshen/pruning-turbulence-vit/`:
- `<run_name>/`: Folder corresponding to a specific train/prune run
- `<run_name>/checkpoints/`: Model checkpoints (`*.tar`) containing weights, optimizer, epoch
- `<run_name>/inference/`: Inference `.npy` files
- `<run_name>/short_analyses/`: Short analysis outputs
- `<run_name>/long_analyses/`: Long analysis outputs


## Package organization

`src/models/`
- `vision_transformer.py`: Encoder-decoder Vision Transformer
- `modules/`: Implementations of attention, mlp, embeddings, positional encodings, conv (SubPixel 3D decoder)

`src/training/`
- `train.py`: Standard supervised training loop with DDP
- `prune_unstructured.py`: Iterative prune-finetune loop with importance scores (L1, Fisher, Taylor, random)
- `prune_attention_heads.py`: Structured attention head pruning
- `distill.py`: Knowledge distillation via hidden layer MSE matching
- `utils/`: Helper functions as well as importance score implementations

`src/inference/`
- `make_inference.py`: Generate rollout predictions
- `short_analysis.py`: Analyses for short leadtime trajectories
- `long_analysis.py`: Analyses for long leadtime trajectories
- `make_gif.py`: Generate GIF animation of rollout

`src/data/`
- `datasets`: PyTorch dataset for 2D turbulence frames with configurable stride/steps

`configs/`: TOML files for running training, pruning, etc. jobs
`notebooks/`: analysis and visualization notebooks
`jobs/`: PBS scripts for running compute jobs on cluster
`scripts/`: Python scripts setting up and running routines from `src/`
`datagen/`: Code for running Py2D fluid simulations and producing data files

## Code style

This is a research project, so code should be amenable to rapid iteration and experimentation. 
- Prioritize simple, readable implementations. 
- Don't be too defensive, avoid try-except unless truly necessary. 
- Prefer shallow dependency trees. 