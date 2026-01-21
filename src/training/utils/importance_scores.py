import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import AdamW

from torch.utils.data import DataLoader

def compute_importance_scores(importance_metric, model, device, **kwargs):
    """
    Router function to compute pruning importance scores using different metrics. 
    """

    if importance_metric == 'l1':
        return None
    elif importance_metric == 'taylor':
        return _compute_taylor_importance(model, device, **kwargs)
    elif importance_metric == 'fisher':
        return _compute_fisher_importance(model, device, **kwargs)
    elif importance_metric == 'squisher':
        return _compute_squisher_importance(model, device, **kwargs)
    else:
        return None

def _compute_taylor_importance(model, device, dataset, batch_size=1, num_batches=1024, **kwargs):
    """
    Compute first-order Taylor scores for unstructured pruning and return them in the
    format expected by prune.global_unstructured(..., importance_scores=...).

    Intended for execution in an initialized DDP environment. 
    
    Returns:
        (parameters_to_prune, importance_scores_dict)
        where importance_scores_dict maps (module, "weight") -> score tensor.
    """

    # Get parameters to prune

    weights = model.module.get_weights()

    # Make dataloader

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Prepare score buffers (match device of parameters)
    scores = {
        (m, name): torch.zeros_like(getattr(m, name), requires_grad=False)
        for (m, name) in weights
    }

    model.eval()

    if dist.get_rank() == 0:
        for i, (x, y_true) in enumerate(dataloader):
            if i >= num_batches: break
            x, y_true = x.to(device, non_blocking=True), y_true.to(device, non_blocking=True)

            model.zero_grad(set_to_none=True)
            with model.no_sync():  # prevents DDP all-reduce while still computing grads locally
                y_pred = model(x)  # assume standard forward; adapt if needed
                per_sample_mse = (y_pred - y_true).pow(2)
                if per_sample_mse.ndim > 2:
                    per_sample_mse = per_sample_mse.flatten(1)
                # Mean over features, sum over batch to avoid 1/B scaling
                loss = per_sample_mse.mean(1).sum()
                loss.backward()

            # Accumulate (grad * weight)^2
            with torch.no_grad():
                for (m, name) in weights:
                    p: torch.Tensor = getattr(m, name)
                    if p.grad is not None:
                        scores[(m, name)].add_((p.grad * p).pow(2))
            seen_samples += x.size(0)

    for k in scores:
        if seen_samples > 0:
            scores[k].div_(float(seen_samples))
        dist.broadcast(scores[k], src=0)

    return scores

def _compute_fisher_importance(model, device, dataset, batch_size=32, num_batches=64, **kwargs):
    """
    Compute second-order Fisher scores for unstructured pruning and return them in the
    format expected by prune.global_unstructured(..., importance_scores=...).

    Intended for execution in an initialized DDP environment. 
    
    Returns:
        (parameters_to_prune, importance_scores_dict)
        where importance_scores_dict maps (module, "weight") -> score tensor.
    """

    # Get parameters to prune

    weights = model.module.get_weights()

    # Make dataloader

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Prepare score buffers (match device of parameters)
    scores = {
        (m, name): torch.zeros_like(getattr(m, name), requires_grad=False)
        for (m, name) in weights
    }

    model.eval()

    if dist.get_rank() == 0:
        for i, (x, y_true) in enumerate(dataloader):
            if i >= num_batches: break
            x, y_true = x.to(device, non_blocking=True), y_true.to(device, non_blocking=True)

            model.zero_grad(set_to_none=True)
            with model.no_sync():  # prevents DDP all-reduce while still computing grads locally
                y_pred = model(x)  # assume standard forward; adapt if needed
                per_sample_mse = (y_pred - y_true).pow(2)
                if per_sample_mse.ndim > 2:
                    per_sample_mse = per_sample_mse.flatten(1)
                # Mean over features, sum over batch to avoid 1/B scaling
                loss = per_sample_mse.mean(1).sum()
                loss.backward()

            # Accumulate (grad * weight)^2
            with torch.no_grad():
                for (module, param) in weights:
                    p = getattr(module, param)
                    if p.grad is not None:
                        scores[(module, param)].add_(p.grad.pow(2))
            seen_samples += x.size(0)

    for (module, param) in scores:
        if seen_samples > 0:
            scores[(module, param)].div_(float(seen_samples))
        scores[(module, param)].mul_(getattr(module, param).pow(2))
        dist.broadcast(scores[(module, param)], src=0)

    return scores


def _compute_squisher_importance(model, device, optimizer_state, **kwargs):

    if optimizer_state is None: return None

    weights = model.module.get_weights()

    # Make a throwaway AdamW, load the provided state_dict onto it
    optimizer = AdamW(model.parameters())
    optimizer.load_state_dict(optimizer_state)

    scores = {
        (m, name): torch.zeros_like(getattr(m, name), device=device)
        for (m, name) in weights
    }

    if dist.get_rank() == 0:
        for (m, name) in weights:
            p = getattr(m, name)
            st = optimizer.state.get(p, None)
            if st is not None and "exp_avg_sq" in st:
                v = st["exp_avg_sq"]
                v = v.to(device, non_blocking=True)
                scores[(m, name)] = (p.detach() ** 2) * v

    for k in scores:
        dist.broadcast(scores[k], src=0)

    dist.barrier()

    return scores
