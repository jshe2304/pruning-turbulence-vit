"""
Trainer functions for end-to-end training of models. 

Single-device and distributed training are implemented separately for simplicity. 
"""

import os

import torch
from torch.optim import AdamW
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .utils import train_one_epoch, sample_loss, sample_loss_distributed

def train(
    model, device, 
    train_dataset, validation_dataset, 
    learning_rate, weight_decay,
    warmup_start_factor, warmup_epochs, plateau_factor, plateau_patience,
    epochs, batch_size, 
    checkpoint_period, output_dir, 
    logger=None,
    **kwargs
    ):
    """
    Train the model on a single device. 

    Args:
        model: The model to train
        device: The device to use
        train_dataset: The training dataset
        validation_dataset: The validation dataset
        learning_rate: The learning rate
        weight_decay: The weight decay
        warmup_start_factor: The start factor for the warmup phase
        warmup_epochs: The total number of epochs for the warmup phase
        plateau_factor: The factor for the plateau phase
        plateau_patience: The patience for the plateau phase
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

    # Schedulers

    warmup_steps = warmup_epochs * len(train_dataloader)
    warmup = lr_scheduler.LinearLR(
        optimizer, start_factor=warmup_start_factor, total_iters=warmup_steps
    )
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=plateau_factor, patience=plateau_patience
    )

    # Training

    for epoch in range(epochs):

        # Train

        train_one_epoch(
            model, train_dataloader, device, 
            optimizer, scheduler=warmup
        )

        # Sample losses

        train_loss = sample_loss(
            model, train_dataset, batch_size=batch_size, device=device
        )
        validation_loss = sample_loss(
            model, validation_dataset, batch_size=batch_size, device=device
        )

        # Step scheduler

        scheduler.step(validation_loss)

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

def train_distributed(
    model, device, 
    train_dataset, validation_dataset, 
    learning_rate, weight_decay,
    warmup_start_factor, warmup_epochs, plateau_factor, plateau_patience,
    epochs, batch_size, 
    checkpoint_period, output_dir, 
    logger=None,
    **kwargs
    ):
    """
    Train the model using distributed training. 

    Args:
        model: The model to train
        device: The device to use
        train_dataset: The training dataset
        validation_dataset: The validation dataset
        learning_rate: The learning rate
        weight_decay: The weight decay
        warmup_start_factor: The start factor for the warmup phase
        warmup_epochs: The total number of epochs for the warmup phase
        plateau_factor: The factor for the plateau phase
        plateau_patience: The patience for the plateau phase
        epochs: The number of epochs to train for
        batch_size: The batch size to use
        checkpoint_period: How often to save the model (epochs)
        output_dir: The directory to save the logs and checkpoints
        logger: The wandb logger
        **kwargs: Overflow arguments
    """

    local_rank = int(os.environ["LOCAL_RANK"])

    # Logging paths

    if local_rank == 0:
        checkpoint_dir = os.path.join(output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Dataloader

    sampler = DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4
    )

    # Optimizer

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Schedulers

    warmup_steps = warmup_epochs * len(train_dataloader)
    warmup = lr_scheduler.LinearLR(
        optimizer, start_factor=warmup_start_factor, total_iters=warmup_steps
    )
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=plateau_factor, patience=plateau_patience
    )

    # Training

    for epoch in range(epochs):

        # Train

        sampler.set_epoch(epoch)
        train_one_epoch(
            model, train_dataloader, device, 
            optimizer, scheduler=warmup
        )

        # Sample losses

        train_loss = sample_loss_distributed(
            model, train_dataset, batch_size=batch_size, device=device
        )
        validation_loss = sample_loss_distributed(
            model, validation_dataset, batch_size=batch_size, device=device
        )

        # Step scheduler

        scheduler.step(validation_loss)

        # Logging

        if local_rank == 0:

            # Log losses to wandb

            if logger is not None:
                logger.log({
                    "train_loss": train_loss,
                    "validation_loss": validation_loss, 
                    "lr": optimizer.param_groups[0]['lr']
                })

            # Save model

            if epoch % checkpoint_period == 0:
                torch.save(
                    model.module.state_dict(), 
                    os.path.join(checkpoint_dir, f'epoch_{epoch}.pt')
                )
