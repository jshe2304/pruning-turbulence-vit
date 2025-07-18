"""
Vision Transformer iterative pruning script. 
Here, we implement a prune-finetune-repeat loop on a pretrained model. 
Does not support distributed training. 

To run, pass in a path to a TOML config file as an argument. 
The TOML should contain the following sections:
"""

import sys
import toml
import os
import wandb
from datetime import datetime

import torch

from models.vit import ViT

from data.datasets import TimeSeriesDataset
from trainers.prune_iterative import prune_iterative

if __name__ == '__main__':

    # Load config

    config_path = sys.argv[1]
    config = toml.load(config_path)

    # Make output directory

    output_dir = config['finetuning']['output_dir']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, timestamp)
    config['finetuning']['output_dir'] = output_dir

    # Initialize wandb

    logger = wandb.init(project="turbulence-vit-prune", config=config)

    # Initialize model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ViT(**config['model'])

    # Load checkpoint (may need to rename keys)

    state_dict = torch.load(config['model']['checkpoint_path'], map_location=device)
    try:
        model.load_state_dict(state_dict)
    except:
        state_dict = {
            k.replace("module.", ""): v
            for k, v in state_dict.items()
        }
        model.load_state_dict(state_dict)
    model.to(device)

    # Initialize datasets

    train_dataset = TimeSeriesDataset(**config['train_dataset'])
    validation_dataset = TimeSeriesDataset(**config['validation_dataset'])

    # Train model

    prune_iterative(
        model, device, 
        train_dataset, validation_dataset, 
        **config['pruning'], 
        **config['finetuning'], 
        logger=logger
    )

    # Clean up

    logger.finish()
