import os

import torch
from torch.optim import AdamW
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .utils import train_one_epoch, sample_loss

from time import time

def train(
    model, device, 
    train_dataset, validation_dataset, 
    learning_rate, weight_decay,
    warmup_start_factor, warmup_total_iters, plateau_factor, plateau_patience,
    epochs, batch_size, 
    checkpoint_period, output_dir,
    **kwargs
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
        output_dir: The directory to save the logs and checkpoints
        **kwargs: Overflow arguments
    """

    # Logging paths

    if int(os.environ["LOCAL_RANK"]) == 0:
        log_file = os.path.join(output_dir, 'losses.csv')
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

    warmup_scheduler = lr_scheduler.LinearLR(
        optimizer, start_factor=warmup_start_factor, total_iters=warmup_total_iters
    )
    plateau_scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=plateau_factor, patience=plateau_patience
    )

    # Training

    for epoch in range(epochs):

        # Train

        start_time = time()
        sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, device, train_dataloader)
        print(f"Epoch {epoch}: {time() - start_time} seconds")

        # Sample losses

        start_time = time()
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
        print(f"Compute losses: {time() - start_time} seconds")

        # Step schedulers

        warmup_scheduler.step()
        plateau_scheduler.step(validation_loss)

        # Log losses

        if int(os.environ["LOCAL_RANK"]) != 0: continue

        with open(log_file, 'a') as f:
            f.write(f'{train_loss},{validation_loss}\n')

        # Save model

        if (epoch + 1) % checkpoint_period == 0:
            torch.save(
                model.state_dict(), 
                os.path.join(checkpoint_dir, f'epoch_{epoch}.pt')
            )
