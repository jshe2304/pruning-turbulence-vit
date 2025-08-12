"""
Trainer functions for end-to-end training of models. 

Single-device and distributed training are implemented separately for simplicity. 
"""

import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.utils.data.distributed import DistributedSampler

from .utils import train_one_epoch, compute_loss

def cosine_finetune(
    model, device, # Model
    train_dataset, validation_dataset, # Data
    epochs, batch_size, # Data loader
    warmup_start_factor, num_warmup_epochs, # Warmup scheduler
    num_decay_epochs, eta_min_factor,  # Cosine scheduler
    lr, weight_decay, optimizer_state=None, # Optional existing optimizer
    logger=None, # Logging
    coast=False, 
    **kwargs # Overflow arguments
    ):
    """
    Finetune the model with a warmup followed by a cosine decay. 

    Args:
        model: The model to train
        device: The device to use
        train_dataset: The training dataset
        validation_dataset: The validation dataset
        epochs: The number of epochs to train for
        batch_size: The batch size to use
        lr: The initial learning rate
        weight_decay: The weight decay
        warmup_start_factor: The starting learning rate factor for the warmup scheduler
        num_warmup_epochs: The number of epochs for the warmup scheduler
        logger: The wandb logger
        **kwargs: Overflow arguments
    """

    local_rank = int(os.environ["LOCAL_RANK"])

    # Check if coasting

    if coast:
        num_warmup_epochs = epochs
        warmup_start_factor = 1.0
        lr *= 0.0001

    # Dataloader

    sampler = DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4
    )

    # Optimizer and schedulers

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    warmup_steps = num_warmup_epochs * len(train_dataloader)
    warmup = lr_scheduler.LinearLR(
        optimizer, start_factor=warmup_start_factor, total_iters=warmup_steps
    )

    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_decay_epochs - num_warmup_epochs, 
        eta_min=lr * eta_min_factor
    )

    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    # Training

    for epoch in range(epochs):

        # Train

        sampler.set_epoch(epoch)
        train_one_epoch(
            model, train_dataloader, device, 
            optimizer, scheduler=warmup
        )

        # Sample losses

        train_loss = compute_loss(
            model, train_dataset, batch_size=batch_size, device=device
        )
        validation_loss = compute_loss(
            model, validation_dataset, batch_size=batch_size, device=device
        )

        # Step scheduler

        if epoch >= num_warmup_epochs: scheduler.step() # (validation_loss)

        # Logging

        if local_rank == 0 and logger is not None:
            logger.log({
                "train_loss": train_loss,
                "validation_loss": validation_loss, 
                "lr": optimizer.param_groups[0]['lr']
            })

    return epochs

def plateau_finetune(
    model, device, # Model
    train_dataset, validation_dataset, # Data
    epochs, batch_size, # Data loader
    warmup_start_factor, num_warmup_epochs, # Warmup scheduler
    plateau_factor, plateau_patience, early_stop_lr_threshold, # Plateau scheduler
    lr, weight_decay, optimizer_state=None, # Optimizer
    output_dir=None, logger=None, # Logging
    coast=False, 
    **kwargs # Overflow arguments
    ):
    """
    Finetune the model with a warmup followed by a cosine decay. 

    Args:
        model: The model to train
        device: The device to use
        train_dataset: The training dataset
        validation_dataset: The validation dataset
        epochs: The number of epochs to train for
        batch_size: The batch size to use
        lr: The initial learning rate
        weight_decay: The weight decay
        warmup_start_factor: The starting learning rate factor for the warmup scheduler
        num_warmup_epochs: The number of epochs for the warmup scheduler
        output_dir: The output directory
        logger: The wandb logger
        **kwargs: Overflow arguments
    """

    local_rank = int(os.environ["LOCAL_RANK"])

    # Dataloader

    sampler = DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4
    )

    # Optimizer and schedulers

    if coast: lr = early_stop_lr_threshold
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    warmup_steps = num_warmup_epochs * len(train_dataloader)
    warmup = lr_scheduler.LinearLR(
        optimizer, start_factor=warmup_start_factor, total_iters=warmup_steps
    ) if not coast else None

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=plateau_factor, patience=plateau_patience
    )

    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    # Training

    total_parameters = model.module.n_parameters()
    for epoch in range(epochs):

        # Train

        sampler.set_epoch(epoch)
        train_one_epoch(
            model, train_dataloader, device, 
            optimizer, scheduler=warmup
        )

        # Sample losses

        train_loss = compute_loss(
            model, train_dataset, batch_size=batch_size, device=device
        )
        validation_loss = compute_loss(
            model, validation_dataset, batch_size=batch_size, device=device
        )

        # Step scheduler

        scheduler.step(validation_loss) # (validation_loss)

        # Logging

        if local_rank == 0 and logger is not None:

            pruned_parameters = model.module.n_pruned_parameters()
            unpruned_parameters = total_parameters - pruned_parameters
            proportion_remaining = unpruned_parameters / total_parameters
            logger.log({
                "train_loss": train_loss,
                "validation_loss": validation_loss, 
                "lr": optimizer.param_groups[0]['lr'], 
                "unpruned_parameters": unpruned_parameters, 
                "proportion_remaining": proportion_remaining,
            })

            if output_dir is not None:
                torch.save(
                    {
                        'model_state': model.module.state_dict(), 
                        'optimizer_state': optimizer.state_dict()
                    }, 
                    os.path.join(output_dir, f"last.tar")
                )

        # Early stopping

        if optimizer.param_groups[0]['lr'] <= early_stop_lr_threshold:
            return optimizer.state_dict()

    return optimizer.state_dict()

finetuners = {
    'cosine': cosine_finetune, 
    'plateau': plateau_finetune
}