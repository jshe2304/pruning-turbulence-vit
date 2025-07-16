import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train_one_epoch(model, optimizer, device, train_dataloader, scheduler=None):
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
        loss = torch.nn.functional.mse_loss(pred, target.to(device))
        loss.backward()
        optimizer.step()

        if scheduler is not None: scheduler.step()

def sample_loss(model, data_tensor, n_samples=4096, batch_size=128, device='cpu'):
    """
    Evaluates the full validation loss. 

    Args:
        model: The model to evaluate. 
        data_tensor: The dataset to evaluate on. 
        n_samples: The number of samples to use for evaluation. 
        batch_size: The batch size to use for evaluation. 
        device: The device to use for evaluation. 
    """

    dataloader = DataLoader(data_tensor, batch_size=batch_size, shuffle=True)
    n_batches = n_samples // batch_size
    
    model.eval()
    with torch.no_grad():
        
        losses = []
        for i, (img, target) in enumerate(dataloader):
            if i >= n_batches: break

            pred = model(img.to(device))
            loss = nn.functional.mse_loss(pred, target.to(device))
            losses.append(loss.item())

        return np.mean(losses)
