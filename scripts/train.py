"""
Vision Transformer pre-training script.

To run, pass in:
- A path to a TOML config file to start a new run
- A path to an output directory to resume from a previous run

The TOML config should contain the following sections:
- model: The model to train
- training: The training parameters
- train_dataset: The training dataset
- validation_dataset: The validation dataset
"""

import sys
import toml
import os
import wandb

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.models import create_model
from src.data.py2d_dataset import Py2DDataset
from src.data.multi_py2d_dataset import MultiPy2DDataset
from src.training.train import train

def main(config: dict):

    # Set up distributed training

    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Create output and checkpoint directories (only rank 0 process)

    output_dir = config['output_dir']
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    if local_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize wandb (only rank 0 process)

    logger = wandb.init(
        project="turbulence-vit-train",
        id=config.get('wandb_id', None),
        config=config,
        resume='allow'
    ) if local_rank == 0 else None

    # Save config with wandb_id (only for new runs, rank 0 only)

    if local_rank == 0 and 'wandb_id' not in config: 
        config['wandb_id'] = logger.id
        config_path = os.path.join(output_dir, 'config.toml')
        with open(config_path, 'w') as f:
            toml.dump(config, f)

    if world_size > 1: dist.barrier()

    # Adjust batch size for distributed training

    config['training']['batch_size'] //= world_size

    # Load checkpoint if provided

    optimizer_state = model_state_dict = None
    start_epoch = 0
    if 'checkpoint_file' in config and config['checkpoint_file'] is not None:
        state_dict = torch.load(config['checkpoint_file'], map_location=device, weights_only=False)
        start_epoch = state_dict.pop('epoch', None) + 1
        optimizer_state = state_dict.pop('optimizer_state', None)
        model_state_dict = state_dict.pop('model_state', state_dict)

    # Initialize model

    model = create_model(**config['model']).to(device)
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Initialize datasets

    Dataset = MultiPy2DDataset if 'data_dirs' in config['train_dataset'] else Py2DDataset
    train_dataset = Dataset(**config['train_dataset'])
    validation_dataset = Dataset(**config['validation_dataset'])

    # Train model

    train(
        model, device,
        train_dataset, validation_dataset,
        checkpoint_dir=checkpoint_dir,
        optimizer_state=optimizer_state, 
        start_epoch=start_epoch, 
        **config['training'],
        logger=logger
    )

    # Clean up logger

    if local_rank == 0 and logger is not None: 
        logger.finish()

    dist.destroy_process_group()

if __name__ == '__main__':

    path = sys.argv[1]

    if path.endswith('.toml'):
        config = toml.load(path)
    elif os.path.isdir(path):
        config = toml.load(os.path.join(path, 'config.toml'))
        config['checkpoint_file'] = os.path.join(path, 'checkpoints/last.tar')
    else:
        raise ValueError(f"Invalid path: {path}\n")

    main(config)
