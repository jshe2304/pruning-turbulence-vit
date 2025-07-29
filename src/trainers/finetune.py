"""
Trainer functions for end-to-end training of models. 

Single-device and distributed training are implemented separately for simplicity. 
"""

import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .utils import train_one_epoch, sample_loss

def finetune(
    model, device, 
    train_dataset, validation_dataset, 
    learning_rate, weight_decay,
    epochs, batch_size, 
    checkpoint_period, output_dir, 
    logger=None,
    **kwargs
    ):
    """
    Finetune the model on a single device. 
    The same routine used for training, without learning rate schedulers. 
    Does not support distributed training. 

    Args:
        model: The model to train
        device: The device to use
        train_dataset: The training dataset
        validation_dataset: The validation dataset
        learning_rate: The learning rate
        weight_decay: The weight decay
        epochs: The number of epochs to train for
        batch_size: The batch size to use
        checkpoint_period: How often to save the model (epochs)
        output_dir: The directory to save the logs and checkpoints
        logger: The wandb logger
        **kwargs: Overflow arguments
    """

    # Logging paths

    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Dataloader

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=4
    )

    # Optimizer

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training

    for epoch in range(epochs):

        # Train

        train_one_epoch(
            model, train_dataloader, device, 
            optimizer
        )

        # Sample losses

        train_loss = sample_loss(
            model, train_dataset, batch_size=batch_size, device=device
        )
        validation_loss = sample_loss(
            model, validation_dataset, batch_size=batch_size, device=device
        )

        # Log to wandb

        if logger is not None:
            logger.log({
                "train_loss": train_loss,
                "validation_loss": validation_loss, 
                "lr": optimizer.param_groups[0]['lr']
            })

        # Save model

        if epoch % checkpoint_period == 0:
            torch.save(
                model.state_dict(), 
                os.path.join(checkpoint_dir, f'epoch_{epoch}.pt')
            )
