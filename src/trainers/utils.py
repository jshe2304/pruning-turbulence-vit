import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F

def train_one_epoch(
    model, train_dataloader, device, 
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
    """

    model.train()
    for img, target in train_dataloader:
        optimizer.zero_grad()
        pred = model(img.to(device))
        loss = F.mse_loss(pred, target.to(device))
        loss.backward()
        optimizer.step()

        if scheduler is not None: scheduler.step()

@torch.no_grad()
def sample_loss(model, data_tensor, n_samples=4096, batch_size=128, device='cpu'):
    """
    Estimates the loss of the model on a dataset. 
    No support for distributed inference, only single GPU/CPU. 

    Args:
        model: The model to evaluate. 
        data_tensor: The dataset to evaluate on. 
        n_samples: The number of samples to use for evaluation. 
        batch_size: The batch size to use for evaluation. 
        device: The device to use for evaluation. 
    """

    model.eval()

    dataloader = DataLoader(
        data_tensor, batch_size=batch_size, shuffle=True, num_workers=4
    )
    
    total_loss, samples_processed = 0., 0
    for img, target in dataloader:
        if samples_processed >= n_samples: break

        this_batch_size = img.size(0)

        pred = model(img.to(device))
        total_loss += F.mse_loss(pred, target.to(device), reduction='sum').item()

        samples_processed += this_batch_size 

    return total_loss / samples_processed

@torch.no_grad()
def sample_loss_distributed(model, data_tensor, n_samples=4096, batch_size=128, device='cpu'):
    """
    Estimates the loss of the model on a dataset. 
    Supports distributed inference. 

    Args:
        model: The model to evaluate. 
        data_tensor: The dataset to evaluate on. 
        n_samples: The number of samples to use for evaluation. 
        batch_size: The batch size to use for evaluation. 
        device: The device to use for evaluation. 
    """

    model.eval()

    sampler = DistributedSampler(data_tensor, shuffle=True)
    dataloader = DataLoader(
        data_tensor, batch_size=batch_size, sampler=sampler, num_workers=4
    )
    
    total_loss, samples_processed = 0., 0
    for img, target in dataloader:
        if samples_processed >= n_samples: break
        
        this_batch_size = img.size(0)

        pred = model(img.to(device))
        total_loss += F.mse_loss(
            pred, target.to(device), 
            reduction='sum'
        ).item()

        samples_processed += this_batch_size 

    total_loss = torch.tensor(total_loss, device=device)
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    total_loss = total_loss.item()

    return total_loss / samples_processed
