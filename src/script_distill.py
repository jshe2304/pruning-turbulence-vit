"""
Vision Transformer distillation script.

To run, pass in a path to a TOML config file as an argument. 
The TOML should contain the following sections:
- model: The model to train
- training: The training parameters
- train_dataset: The training dataset
- validation_dataset: The validation dataset
"""

import sys
import toml
import os
import wandb
from datetime import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from models.vision_transformer import ViT
from data.datasets import TimeSeriesDataset
from trainers.distill import distill

def main(config: dict):

    # Set up distributed training

    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Make output directory (only rank 0 process)

    if local_rank == 0:
        output_dir = config['training']['output_dir']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_dir, timestamp)
        config['training']['output_dir'] = output_dir

    # Initialize wandb (only rank 0 process)

    logger = wandb.init(
        project="turbulence-vit-train",
        config=config,
    ) if local_rank == 0 else None

    # Adjust batch size for distributed training

    config['training']['batch_size'] //= world_size

    # Initialize model

    model = ViT(**config['model']).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Initialize teacher model

    teacher = ViT(**config['teacher']).to(device)
    state_dict = torch.load(config['teacher_checkpoint_file'], map_location=device, weights_only=False)
    teacher.load_state_dict(state_dict)
    teacher = DDP(teacher, device_ids=[local_rank], output_device=local_rank)

    # Initialize datasets

    train_dataset = TimeSeriesDataset(**config['train_dataset'])
    validation_dataset = TimeSeriesDataset(**config['validation_dataset'])

    # Distill model

    distill(
        model, teacher, device, 
        train_dataset, validation_dataset, 
        **config['training'], 
        logger=logger
    )

    # Clean up

    if local_rank == 0 and logger is not None: 
        logger.finish()

    dist.destroy_process_group()

if __name__ == '__main__':

    config_path = sys.argv[1]
    config = toml.load(config_path)

    main(config)
