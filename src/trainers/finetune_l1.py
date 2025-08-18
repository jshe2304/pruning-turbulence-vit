"""
Finetuning with L1 regularization.

Patterned after existing finetuners but adds an L1 penalty term to the
supervised loss during training.
"""

import os

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .utils import compute_loss


def _l1_penalty(parameters):
    total_l1 = 0.0
    for p in parameters:
        if p is None:
            continue
        total_l1 = total_l1 + p.abs().sum()
    return total_l1


def finetune_l1(
    model, device,  # Model (wrapped in DDP)
    train_dataset, validation_dataset,  # Data
    epochs, batch_size,  # Data loader
    warmup_start_factor, num_warmup_epochs,  # Warmup scheduler
    plateau_factor, plateau_patience, early_stop_lr_threshold,  # Plateau scheduler
    lr, weight_decay, l1_lambda, optimizer_state=None,  # Optimizer + reg
    output_dir=None, logger=None,  # Logging
    **kwargs,
):
    """
    Finetune the model with MSE + L1 regularization, using warmup then
    ReduceLROnPlateau. Works with DistributedDataParallel models.

    Args mirror `plateau_finetune` with the addition of `l1_lambda`.
    """

    local_rank = int(os.environ["LOCAL_RANK"])

    # Logging paths

    if local_rank == 0 and output_dir is not None:
        checkpoint_dir = os.path.join(output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)


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
    ) if num_warmup_epochs > 0 else None

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=plateau_factor, patience=plateau_patience
    )

    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    # Training
    for epoch in range(epochs):
        sampler.set_epoch(epoch)

        # Train epoch with L1 regularization
        model.train()
        for img, target in train_dataloader:
            optimizer.zero_grad()
            pred = model(img.to(device))
            mse_loss = F.mse_loss(pred, target.to(device))
            l1_loss = _l1_penalty(model.parameters())
            loss = mse_loss + l1_lambda * l1_loss
            loss.backward()
            optimizer.step()
            if warmup is not None:
                warmup.step()

        # Sample losses
        train_loss = compute_loss(
            model, train_dataset, batch_size=batch_size, device=device
        )
        validation_loss = compute_loss(
            model, validation_dataset, batch_size=batch_size, device=device
        )

        # Step scheduler
        scheduler.step(validation_loss)

        # Logging
        if local_rank == 0 and logger is not None:
            logger.log({
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "lr": optimizer.param_groups[0]['lr'],
                "l1_lambda": l1_lambda,
            })

        # Early stop if LR small enough
        if optimizer.param_groups[0]['lr'] <= early_stop_lr_threshold:
            if local_rank == 0 and checkpoint_dir is not None:
                torch.save(
                    {
                        'model_state': model.module.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                    },
                    os.path.join(checkpoint_dir, f"last.tar"),
                )
            return optimizer.state_dict()

        # Save checkpoint each epoch (last only)
        if local_rank == 0 and checkpoint_dir is not None:
            torch.save(
                {
                    'model_state': model.module.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                },
                os.path.join(checkpoint_dir, f"last.tar"),
            )

    return optimizer.state_dict()



