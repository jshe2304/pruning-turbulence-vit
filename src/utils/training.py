import torch
import torch.nn as nn

def train_one_epoch(model, dataloader, optimizer):
    """
    Trains the model for one epoch. 
    """

    # Training loop

    model.train()
    for x, y in dataloader:
        optimizer.zero_grad()
        pred = model(x)
        loss = nn.functional.mse_loss(pred, y)
        loss.backward()
        optimizer.step()

def sample_loss(model, data_tensor, n=2048, device='cpu'):
    """
    Samples a loss from the model. 
    """

    # Generate batches of size n

    indices = torch.randperm(len(data_tensor) - 1)[:n]

    x, y = data_tensor[indices], data_tensor[indices + 1]
    x, y = x.to(device), y.to(device)

    # Evaluate loss

    model.eval()
    with torch.no_grad():
        pred = model(x)
        loss = nn.functional.mse_loss(pred, y)
        return loss.item()
