import torch
import torch.nn.functional as F
import torch.distributed as dist

from torch.utils.data import DataLoader, DistributedSampler

def compute_importance_scores(model, dataset, importance_metric, device, **kwargs):
    """
    Compute the importance scores for the model.
    """

    if importance_metric == 'l1':
        return None
    elif importance_metric == 'fisher':
        return _compute_fisher_importance(model, dataset, device, **kwargs)
    elif importance_metric == 'taylor':
        return _compute_taylor_importance(model, dataset, device, **kwargs)
    else:
        raise ValueError(f"Invalid importance metric: {importance_metric}")

def _compute_fisher_importance(model, dataset, device, batch_size=16, num_batches=8):
    """
    Compute the importance scores for the model using the Fisher information.
    """

    model.eval()

    # Get parameters to prune

    parameters_to_prune = model.module.get_parameters_to_prune(bias=False)

    # Make dataloader

    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, 
        sampler=sampler, shuffle=False, 
        num_workers=4, pin_memory=True, drop_last=True
    )
    sampler.set_epoch(0)

    # Initialize scores

    g2 = {
        (submodule, param_name): torch.zeros_like(getattr(submodule, param_name), requires_grad=False) 
        for submodule, param_name in parameters_to_prune
    }

    # Accumulate square gradients

    seen = 0
    for input, target in dataloader:
        if seen >= num_batches: break

        # Forward pass and backward pass

        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        model.zero_grad(set_to_none=True)
        preds = model(input)
        per_sample_mse = (preds - target).pow(2)
        if per_sample_mse.ndim > 2:
            per_sample_mse = per_sample_mse.flatten(1)
        loss = per_sample_mse.mean(1).mean() # mean over features, then over batch
        loss.backward()

        # Update sum of squared gradients

        global_batch_size  = input.size(0) * dist.get_world_size()
        with torch.no_grad():
            for (submodule, param_name) in parameters_to_prune:
                param = getattr(submodule, param_name)
                if param.grad is not None:
                    g2[(submodule, param_name)].add_(param.grad.pow(2) * global_batch_size)

        seen += 1

    # Compute scores

    scores = {}
    with torch.no_grad():
        for (submodule, param_name) in g2.keys():
            F_diag = g2[(submodule, param_name)] / seen
            w = getattr(submodule, param_name)
            scores[(submodule, param_name)] = F_diag * w.pow(2)

    return scores

def _compute_taylor_importance(model, dataset, device, batch_size=16, num_batches=8):
    """
    Compute the importance scores for the model using the Taylor expansion.
    """

    model.eval()

    # Get parameters to prune

    parameters_to_prune = model.module.get_parameters_to_prune(bias=False)

    # Make dataloader

    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, 
        sampler=sampler, shuffle=False, 
        num_workers=4, pin_memory=True, drop_last=True
    )
    sampler.set_epoch(0)

    # Initialize scores

    scores = {
        (submodule, param_name): torch.zeros_like(getattr(submodule, param_name), requires_grad=False) 
        for submodule, param_name in parameters_to_prune
    }

    # Accumulate square gradients

    seen = 0
    for input, target in dataloader:
        if seen >= num_batches: break

        # Forward pass and backward pass

        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        model.zero_grad(set_to_none=True)
        preds = model(input)
        per_sample_mse = (preds - target).pow(2)
        if per_sample_mse.ndim > 2:
            per_sample_mse = per_sample_mse.flatten(1)
        loss = per_sample_mse.mean(1).mean() # mean over features, then over batch
        loss.backward()

        # Update sum of squared gradients

        global_batch_size  = input.size(0) * dist.get_world_size()
        with torch.no_grad():
            for (submodule, param_name) in parameters_to_prune:
                param = getattr(submodule, param_name)
                if param.grad is not None:
                    scores[(submodule, param_name)].add_((param * param.grad).pow(2) * global_batch_size)

        seen += 1

    # Compute scores

    with torch.no_grad():
        for (submodule, param_name) in scores.keys():
            scores[(submodule, param_name)].div_(seen)

    return scores
