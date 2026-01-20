import torch
import torch.nn.functional as F

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
    for ic, target in train_dataloader:
        ic, target = ic.to(device), target.to(device)

        optimizer.zero_grad()

        for _ in range(num_rollout_steps):
            y_pred = model(ic)
            prev_ic = ic[:, :, :-1, :, :].contiguous()
            ic = torch.cat([y_pred, prev_ic], dim=2)
        
        loss = F.mse_loss(y_pred, target)
        loss.backward()
        optimizer.step()
        if scheduler is not None: scheduler.step()