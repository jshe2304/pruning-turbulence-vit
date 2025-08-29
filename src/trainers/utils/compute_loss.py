import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F

@torch.no_grad()
def compute_loss(model, dataset, num_rollout_steps, n_samples=4096, batch_size=32, device='cpu'):
    """
    Compute the MSE loss of the model on a dataset. 
    Supports distributed inference. 

    Args:
        model: The model to evaluate. 
        dataset: The dataset to evaluate on. 
        n_samples: The number of samples to use for evaluation. 
        batch_size: The batch size to use for evaluation. 
        device: The device to use for evaluation. 
    """

    model.eval()

    sampler = DistributedSampler(dataset, shuffle=True) if dist.is_initialized() else None
    shuffle = not dist.is_initialized()
    dataloader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle,
        num_workers=4, pin_memory=True
    )
    
    total_loss, samples_processed = 0., 0
    for ic, target in dataloader:
        if samples_processed >= n_samples: break
        ic, target = ic.to(device), target.to(device)
        
        this_batch_size = ic.size(0)

        for _ in range(num_rollout_steps):
            y_pred = model(ic)
            prev_ic = ic[:, :, :-1, :, :].contiguous()
            ic = torch.cat([y_pred, prev_ic], dim=2)

        sample_loss = F.mse_loss(y_pred, target).item()
        batch_loss = sample_loss * this_batch_size
        total_loss += batch_loss

        samples_processed += this_batch_size

    mean_loss = total_loss / samples_processed
    mean_loss = torch.tensor(mean_loss, device=device)
    if sampler is not None: dist.all_reduce(mean_loss, op=dist.ReduceOp.AVG)
    mean_loss = mean_loss.item()

    return mean_loss
