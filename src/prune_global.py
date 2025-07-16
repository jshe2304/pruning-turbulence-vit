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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from models.vit import ViT

from utils.data import TimeSeriesDataset
from utils.training import sample_loss, train_one_epoch

def initialize(
    model, device, 
    train_dataset, 
    learning_rate, weight_decay,
    warmup_start_factor, warmup_total_batches, 
    epochs, batch_size, 
    ):
    """
    Initialize the model. 

    Args:
        model: The model to initialize
        device: The device to use
        train_dataset: The training dataset
    """

    # Dataloader

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Optimizer

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Schedulers

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=warmup_start_factor, total_iters=warmup_total_batches
    )

    # Training

    for epoch in range(epochs):
        train_one_epoch(model, optimizer, device, train_dataloader, warmup_scheduler)

    return model, optimizer, warmup_scheduler

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

    # Create initialization checkpoint

    
