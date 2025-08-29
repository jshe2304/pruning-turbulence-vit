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

def finetune(
    model, device, # Model
    optimizer_state, # Optimizer
    train_dataset, validation_dataset, # Data
    epochs, batch_size, num_rollout_steps, # Data loader
    warmup_start_factor, num_warmup_epochs, # Warmup scheduler
    plateau_factor, plateau_patience, early_stop_lr_threshold, # Plateau scheduler
    lr, weight_decay, # Optimizer
    checkpoint_dir=None, logger=None, # Logging 
    **kwargs # Overflow arguments
    ):
    """
    Finetune the model with a warmup followed by a cosine decay. 

    Args:
        model: The model to train
        device: The device to use
        optimizer_state: The optimizer state to use
        train_dataset: The training dataset
        validation_dataset: The validation dataset
        batch_size: The batch size to use
        num_rollout_steps: The number of steps to roll out for the validation dataset
        lr: The initial learning rate
        weight_decay: The weight decay
        warmup_start_factor: The starting learning rate factor for the warmup scheduler
        num_warmup_epochs: The number of epochs for the warmup scheduler
        plateau_factor: The factor to reduce the learning rate
        plateau_patience: The patience for the plateau scheduler
        early_stop_lr_threshold: The learning rate threshold for early stopping
        continuing: Whether to continue training from a checkpoint
        checkpoint_dir: The output directory
        logger: The wandb logger
        **kwargs: Overflow arguments
    """

    local_rank = int(os.environ["LOCAL_RANK"])
    total_parameters = model.module.n_parameters()
    pruned_parameters = model.module.n_pruned_parameters()
    unpruned_parameters = total_parameters - pruned_parameters
    proportion_remaining = unpruned_parameters / total_parameters

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
    ) if optimizer_state is None else None

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=plateau_factor, patience=plateau_patience
    )

    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    # Training

    for epoch in range(epochs):

        # Train

        sampler.set_epoch(epoch)
        train_one_epoch(
            model, device, 
            train_dataloader, num_rollout_steps, 
            optimizer, scheduler=warmup
        )

        # Sample rollout loss

        rollout_loss = compute_loss(
            model, validation_dataset, 
            num_rollout_steps=num_rollout_steps, batch_size=batch_size, device=device
        )

        # Sample single-step loss if needed

        loss = rollout_loss
        if num_rollout_steps > 1:
            validation_dataset.target_step //= num_rollout_steps
            loss = compute_loss(
                model, validation_dataset, 
                num_rollout_steps=1, batch_size=batch_size, device=device
            )
            validation_dataset.target_step *= num_rollout_steps

        # Step scheduler

        scheduler.step(rollout_loss) # (validation_loss)

        # Logging

        if local_rank == 0 and logger is not None:

            logger.log({
                "validation_loss": loss, 
                "rollout_loss": rollout_loss, 
                "lr": optimizer.param_groups[0]['lr'], 
                "unpruned_parameters": unpruned_parameters, 
                "proportion_remaining": proportion_remaining, 
            })

            if checkpoint_dir is not None:
                torch.save(
                    {
                        'model_state': model.module.state_dict(), 
                        'optimizer_state': optimizer.state_dict()
                    }, 
                    os.path.join(checkpoint_dir, f"last.tar")
                )

        # Early stopping

        if optimizer.param_groups[0]['lr'] <= early_stop_lr_threshold:
            return optimizer.state_dict()

    return optimizer.state_dict()