"""
Iterative global pruning script with reinitialization. 

To run, pass in a path to a TOML config file as an argument. 
The TOML should contain the following sections:
- model: The model to train
- training: The training parameters
- train_dataset: The training dataset
- validation_dataset: The validation dataset
"""

import sys
import toml

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from models.vit import ViT

from data.datasets import TimeSeriesDataset
from trainers.prune_iterative import prune_iterative

if __name__ == '__main__':

    # Load config

    config_path = sys.argv[1]
    config = toml.load(config_path)

    # Device

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_gpus = torch.cuda.device_count()

    # Initialize model

    model = ViT(**config['model'])

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=list(range(n_gpus)))
    model.to(device)

    # Initialize data

    train_dataset = TimeSeriesDataset(**config['train_dataset'])
    validation_dataset = TimeSeriesDataset(**config['validation_dataset'])

    # Load pretrained model

    model.load_state_dict(torch.load(config['pretrained_model']))

    # Iterative pruning

    prune_iterative(
        model, device, 
        train_dataset, validation_dataset, 
        **config['pruning']
    )
