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

def _compute_taylor_importance(model, device, dataset, batch_size=4, num_batches=512, **kwargs):
    """
    Compute first-order Taylor scores for unstructured pruning and return them in the
    format expected by prune.global_unstructured(..., importance_scores=...).

    Intended for execution in an initialized DDP environment. All ranks contribute
    samples and scores are reduced across ranks.

    Returns:
        importance_scores_dict mapping (module, "weight") -> score tensor.
    """

    weights = model.module.get_weights()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    scores = {
        (m, name): torch.zeros_like(getattr(m, name), requires_grad=False)
        for (m, name) in weights
    }

    model.eval()

    seen_samples = 0
    for i, (x, y_true) in enumerate(dataloader):
        if i >= num_batches: break
        x, y_true = x.to(device, non_blocking=True), y_true.to(device, non_blocking=True)

        model.zero_grad(set_to_none=True)
        with model.no_sync():
            y_pred = model(x)
            per_sample_mse = (y_pred - y_true).pow(2)
            if per_sample_mse.ndim > 2:
                per_sample_mse = per_sample_mse.flatten(1)
            loss = per_sample_mse.mean(1).sum()
            loss.backward()

        # Accumulate |grad * weight| (first-order Taylor)
        with torch.no_grad():
            for (m, name) in weights:
                p: torch.Tensor = getattr(m, name)
                if p.grad is not None:
                    scores[(m, name)].add_((p.grad * p).abs())
        seen_samples += x.size(0)

    # Reduce across all ranks
    seen_samples_tensor = torch.tensor(seen_samples, device=device)
    dist.all_reduce(seen_samples_tensor, op=dist.ReduceOp.SUM)
    total_samples = seen_samples_tensor.item()

    for k in scores:
        dist.all_reduce(scores[k], op=dist.ReduceOp.SUM)
        if total_samples > 0:
            scores[k].div_(float(total_samples))

    return scores

def _compute_fisher_importance(model, device, dataset, batch_size=32, num_batches=128, **kwargs):
    """
    Compute Fisher information scores (diagonal approximation) for unstructured pruning.

    Fisher importance = E[grad²], which measures the curvature of the loss landscape
    with respect to each parameter. Higher values indicate parameters the loss is
    more sensitive to.

    Intended for execution in an initialized DDP environment. All ranks contribute
    samples and scores are reduced across ranks.

    Returns:
        importance_scores_dict mapping (module, "weight") -> score tensor.
    """

    weights = model.module.get_weights()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    scores = {
        (m, name): torch.zeros_like(getattr(m, name), requires_grad=False)
        for (m, name) in weights
    }

    model.eval()

    seen_samples = 0
    for i, (x, y_true) in enumerate(dataloader):
        if i >= num_batches: break
        x, y_true = x.to(device, non_blocking=True), y_true.to(device, non_blocking=True)

        model.zero_grad(set_to_none=True)
        with model.no_sync():
            y_pred = model(x)
            per_sample_mse = (y_pred - y_true).pow(2)
            if per_sample_mse.ndim > 2:
                per_sample_mse = per_sample_mse.flatten(1)
            loss = per_sample_mse.mean(1).sum()
            loss.backward()

        # Accumulate grad² (Fisher diagonal)
        with torch.no_grad():
            for (module, param) in weights:
                p = getattr(module, param)
                if p.grad is not None:
                    scores[(module, param)].add_(p.grad.pow(2))
        seen_samples += x.size(0)

    # Reduce across all ranks
    seen_samples_tensor = torch.tensor(seen_samples, device=device)
    dist.all_reduce(seen_samples_tensor, op=dist.ReduceOp.SUM)
    total_samples = seen_samples_tensor.item()

    for k in scores:
        dist.all_reduce(scores[k], op=dist.ReduceOp.SUM)
        if total_samples > 0:
            scores[k].div_(float(total_samples))

    return scores


def _compute_squisher_importance(model, device, optimizer_state, **kwargs):
    """
    Compute importance scores using Adam's second moment estimate (exp_avg_sq).

    This uses the optimizer's running estimate of E[grad²] as a proxy for Fisher
    information, multiplied by weight² to get a saliency-like score. This is
    essentially free if you already have a trained Adam optimizer state.

    Returns:
        importance_scores_dict mapping (module, "weight") -> score tensor.
    """

    if optimizer_state is None: return None

    weights = model.module.get_weights()

    # Make a throwaway AdamW, load the provided state_dict onto it
    optimizer = AdamW(model.parameters())
    optimizer.load_state_dict(optimizer_state)

    scores = {}

    for (m, name) in weights:
        p = getattr(m, name)
        st = optimizer.state.get(p, None)
        if st is not None and "exp_avg_sq" in st:
            v = st["exp_avg_sq"].to(device, non_blocking=True)
            scores[(m, name)] = (p.detach().pow(2)) * v
        else:
            # Fallback to magnitude if no optimizer state for this param
            scores[(m, name)] = p.detach().abs()

    # Broadcast from rank 0 to ensure consistency (optimizer state should be same
    # across ranks, but this guarantees it)
    for k in scores:
        dist.broadcast(scores[k], src=0)

    return scores
