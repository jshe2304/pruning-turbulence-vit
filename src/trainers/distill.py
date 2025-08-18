"""
Trainer functions for end-to-end training of models. 

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

def activation_hook(out_dict, key):
    def hook(module, input, output):
        out_dict[key] = output
    return hook

def distill_one_epoch(
    model, teacher, device, 
    train_dataloader,
    distill_weight, optimizer, scheduler=None, 
    **kwargs
    ):
    """
    Distill the model for one epoch minimizing the MSE loss.

    Args:
        model: The model to train
        teacher: The teacher model
        device: The device to use
        train_dataloader: The dataloader for the training data
        distill_weight: The weight for the distillation loss
        optimizer: The optimizer to use
        scheduler: The scheduler to use
        **kwargs: Overflow arguments
    """

    # Register hooks

    activations = {}
    model.module.decoder_norm.register_forward_hook(activation_hook(activations, 'student'))
    teacher.module.decoder_norm.register_forward_hook(activation_hook(activations, 'teacher'))

    model.train()
    teacher.eval()
    for img, target in train_dataloader:
        optimizer.zero_grad()
    
        # Student forward pass

        pred = model(img.to(device))

        # Teacher forward pass
        with torch.no_grad():
            teacher(img.to(device))

        # Compute loss

        label_loss = F.mse_loss(pred, target.to(device))

        distill_loss = F.mse_loss(activations['student'], activations['teacher'])

        # Backward pass

        loss = label_loss + distill_weight * distill_loss
        loss.backward()

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

def distill(
    model, teacher, device, # Model
    train_dataset, validation_dataset, # Data
    epochs, batch_size, # Data loader
    lr, weight_decay, distill_weight, decay_factor, # Optimizer
    warmup_start_factor, warmup_epochs, # Warmup scheduler
    plateau_factor, plateau_patience, # Plateau scheduler
    output_dir, checkpoint_period=None, logger=None, # Logging
    **kwargs
    ):
    """
    Train the model. 

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

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

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
        distill_one_epoch(
            model, teacher, device, 
            train_dataloader, 
            distill_weight * (decay_factor ** epochs) , optimizer, scheduler=warmup
        )

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
