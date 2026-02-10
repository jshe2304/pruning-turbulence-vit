"""
Trainer function for end-to-end training and finetuning of models.

Single-device and distributed training are implemented separately for simplicity.
"""

import os

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .utils import compute_loss


def train_one_epoch(
    model, device,
    train_dataloader, num_rollout_steps,
    optimizer, scheduler=None,
    **kwargs
    ):
    """
    Train the model for one epoch minimizing the MSE loss.

    Args:
        model: The model to train
        optimizer: The optimizer to use
        device: The device to use
        train_dataloader: The dataloader for the training data
        num_rollout_steps: The number of steps to roll out for the validation dataset
    """

    model.train()
    for *inputs, target in train_dataloader:
        inputs = [x.to(device) for x in inputs]
        target = target.to(device)

        optimizer.zero_grad()

        for _ in range(num_rollout_steps):
            y_pred = model(*inputs)
            prev_ic = inputs[0][:, :, :-1, :, :].contiguous()
            inputs[0] = torch.cat([y_pred, prev_ic], dim=2)

        loss = F.mse_loss(y_pred, target)
        loss.backward()
        optimizer.step()
        if scheduler is not None: scheduler.step()

def train(
    model, device,
    train_dataset, validation_dataset,
    epochs, batch_size,
    lr, weight_decay,
    warmup_start_factor,
    plateau_factor, plateau_patience,
    output_dir,
    # Optional parameters (enable finetune-specific behavior)
    warmup_epochs=None,             # Warmup epochs (alias: num_warmup_epochs)
    optimizer_state=None,           # Resume from saved optimizer state
    num_rollout_steps=1,            # Multi-step rollout for loss
    early_stop_lr_threshold=0,      # Early stopping (0 = disabled)
    checkpoint_period=None,         # Periodic checkpoints (None = disabled)
    save_best=True,                 # Save best.tar based on validation loss
    logger=None,
    **kwargs
    ):
    """
    Train or finetune the model using distributed training.

    Args:
        model: The model to train
        device: The device to use
        train_dataset: The training dataset
        validation_dataset: The validation dataset
        epochs: The number of epochs to train for
        batch_size: The batch size to use
        lr: The learning rate
        weight_decay: The weight decay
        warmup_start_factor: The start factor for the warmup phase
        warmup_epochs: The total number of epochs for the warmup phase
        plateau_factor: The factor for the plateau phase
        plateau_patience: The patience for the plateau phase
        output_dir: The directory to save the logs and checkpoints
        optimizer_state: The optimizer state to resume from (None = fresh start)
        num_rollout_steps: Number of rollout steps for loss computation
        early_stop_lr_threshold: Stop training when LR drops below this (0 = disabled)
        checkpoint_period: Save checkpoint every N epochs (None = disabled)
        save_best: Whether to save best.tar based on validation loss
        logger: The wandb logger
        **kwargs: Overflow arguments
    """

    # Handle alias for warmup_epochs
    if warmup_epochs is None:
        warmup_epochs = kwargs.pop('num_warmup_epochs', 0)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # Logging paths

    if local_rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    # Dataloader

    sampler = DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4
    )

    # Optimizer

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    # Schedulers

    warmup = None
    if optimizer_state is None:
        warmup_steps = warmup_epochs * len(train_dataloader)
        warmup = lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_start_factor, total_iters=warmup_steps
        )

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=plateau_factor, patience=plateau_patience
    )

    # Pruning metrics (if available)

    has_pruning = hasattr(model.module, 'n_pruned_parameters')
    if has_pruning:
        total_parameters = model.module.n_parameters()
        pruned_parameters = model.module.n_pruned_parameters()
        unpruned_parameters = total_parameters - pruned_parameters
        proportion_remaining = unpruned_parameters / total_parameters

    # Training

    best_validation_loss = float('inf')
    for epoch in range(epochs):

        # Train

        sampler.set_epoch(epoch)
        train_one_epoch(
            model, device,
            train_dataloader, num_rollout_steps,
            optimizer, scheduler=warmup
        )

        # Sample losses

        train_loss = compute_loss(
            model, train_dataset, num_rollout_steps=num_rollout_steps,
            batch_size=batch_size, device=device
        )
        validation_loss = compute_loss(
            model, validation_dataset, num_rollout_steps=num_rollout_steps,
            batch_size=batch_size, device=device
        )

        # Sample single-step loss if using multi-step rollout

        single_step_loss = validation_loss
        if num_rollout_steps > 1:
            validation_dataset.target_step //= num_rollout_steps
            single_step_loss = compute_loss(
                model, validation_dataset, num_rollout_steps=1,
                batch_size=batch_size, device=device
            )
            validation_dataset.target_step *= num_rollout_steps

        # Step scheduler

        scheduler.step(validation_loss)

        # Logging

        if local_rank == 0:

            # Print to stdout for PBS log files
            print(f"Epoch {epoch:4d} | train_loss={train_loss:.6e} | val_loss={validation_loss:.6e} | lr={optimizer.param_groups[0]['lr']:.2e}")

            # Log losses to wandb

            if logger is not None:
                log_dict = {
                    "train_loss": train_loss,
                    "validation_loss": single_step_loss,
                    "lr": optimizer.param_groups[0]['lr']
                }
                if num_rollout_steps > 1:
                    log_dict["rollout_loss"] = validation_loss
                if has_pruning:
                    log_dict["unpruned_parameters"] = unpruned_parameters
                    log_dict["proportion_remaining"] = proportion_remaining
                logger.log(log_dict)

            # Save best model

            if save_best and validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                torch.save(
                    {
                        'epoch': epoch,
                        'validation_loss': validation_loss,
                        'model_state': model.module.state_dict(),
                        'optimizer_state': optimizer.state_dict()
                    },
                    os.path.join(output_dir, "best.tar")
                )

            # Save periodic checkpoint

            if checkpoint_period is not None and epoch % checkpoint_period == 0:
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state': model.module.state_dict(),
                        'optimizer_state': optimizer.state_dict()
                    },
                    os.path.join(output_dir, f"epoch_{epoch}.tar")
                )

            # Save last checkpoint (always)

            torch.save(
                {
                    'epoch': epoch,
                    'model_state': model.module.state_dict(),
                    'optimizer_state': optimizer.state_dict()
                },
                os.path.join(output_dir, "last.tar")
            )

        # Early stopping

        if early_stop_lr_threshold > 0 and optimizer.param_groups[0]['lr'] <= early_stop_lr_threshold:
            return optimizer.state_dict()

    return optimizer.state_dict()
