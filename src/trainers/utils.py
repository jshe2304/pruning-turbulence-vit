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
def sample_loss(model, dataset, n_samples=4096, batch_size=32, device='cpu'):
    """
    Estimates the loss of the model on a dataset. 
    No support for distributed inference, only single GPU/CPU. 

    Args:
        model: The model to evaluate. 
        dataset: The dataset to evaluate on. 
        n_samples: The number of samples to use for evaluation. 
        batch_size: The batch size to use for evaluation. 
        device: The device to use for evaluation. 
    """

    model.eval()

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    
    total_loss, samples_processed = 0., 0
    for img, target in dataloader:
        if samples_processed >= n_samples: break
        this_batch_size = img.size(0)

        pred = model(img.to(device))
        sample_loss = F.mse_loss(pred, target.to(device)).item()
        batch_loss = sample_loss * this_batch_size
        total_loss += batch_loss
        samples_processed += this_batch_size

    return total_loss / samples_processed

@torch.no_grad()
def sample_loss_distributed(model, dataset, n_samples=4096, batch_size=32, device='cpu'):
    """
    Estimates the loss of the model on a dataset. 
    Supports distributed inference. 

    Args:
        model: The model to evaluate. 
        dataset: The dataset to evaluate on. 
        n_samples: The number of samples to use for evaluation. 
        batch_size: The batch size to use for evaluation. 
        device: The device to use for evaluation. 
    """

    model.eval()

    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=4
    )
    
    total_loss, samples_processed = 0., 0
    for img, target in dataloader:
        if samples_processed >= n_samples: break
        
        this_batch_size = img.size(0)

        pred = model(img.to(device))
        sample_loss = F.mse_loss(pred, target.to(device)).item()
        batch_loss = sample_loss * this_batch_size
        total_loss += batch_loss

        samples_processed += this_batch_size 

    mean_loss = total_loss / samples_processed
    mean_loss = torch.tensor(mean_loss, device=device)
    dist.all_reduce(mean_loss, op=dist.ReduceOp.AVG)
    mean_loss = mean_loss.item()

    return mean_loss
