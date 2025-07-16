"""
Vision Transformer pre-training script.

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

def pretrain(
    model, device, 
    train_dataset, validation_dataset, 
    learning_rate, weight_decay,
    warmup_start_factor, warmup_total_iters, plateau_factor, plateau_patience,
    epochs, batch_size, 
    checkpoint_period, log_dir, checkpoint_dir, 
    ):
    """
    Train the model for a given number of epochs.

    Args:
        model: The model to train
        device: The device to use
        train_dataset: The training dataset
        validation_dataset: The validation dataset
        learning_rate: The learning rate
        weight_decay: The weight decay
        warmup_start_factor: The start factor for the warmup phase
        warmup_total_iters: The total number of iterations for the warmup phase
        plateau_factor: The factor for the plateau phase
        plateau_patience: The patience for the plateau phase
        epochs: The number of epochs to train for
        batch_size: The batch size to use
        checkpoint_period: How often to save the model (epochs)
        log_dir: The directory to save the logs
        checkpoint_dir: The directory to save the checkpoints
    """

    # Dataloader

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Optimizer

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Schedulers

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=warmup_start_factor, total_iters=warmup_total_iters
    )
    plateau_scheduler = ReduceLROnPlateau(
        optimizer, factor=plateau_factor, patience=plateau_patience, mode='min'
    )

    # Training 

    for epoch in range(epochs):

        # Train

        train_one_epoch(model, optimizer, device, train_dataloader)

        # Sample losses

        train_loss = sample_loss(
            model, train_dataset, 
            n_samples=4096, batch_size=batch_size, 
            device=device
        )
        validation_loss = sample_loss(
            model, validation_dataset, 
            n_samples=4096, batch_size=batch_size, 
            device=device
        )

        # Step schedulers

        warmup_scheduler.step()
        plateau_scheduler.step(validation_loss)

        # Log losses

        with open(log_dir + 'train_losses.csv', 'a') as f:
            f.write(f'{train_loss},{validation_loss}\n')

        # Save model

        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(), checkpoint_dir + f'checkpoint_{epoch}.pt')

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

    # Pretrain

    pretrain(
        model, device, 
        train_dataset, validation_dataset, 
        **config['training']
    )
