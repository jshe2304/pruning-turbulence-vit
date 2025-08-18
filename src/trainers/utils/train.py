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
