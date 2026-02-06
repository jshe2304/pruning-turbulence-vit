import torch
from torch.utils.data import DataLoader


def compute_importance_scores(importance_metric, model, device, **kwargs):
    if importance_metric == 'taylor':
        return _compute_taylor_importance(model, device, **kwargs)
    elif importance_metric == 'fisher':
        return _compute_fisher_importance(model, device, **kwargs)
    else:
        return None


def _compute_taylor_importance(model, device, dataset, batch_size=4, num_batches=512, **kwargs):
    """
    Taylor importance: |weight * grad|, averaged over samples.
    Single-GPU version.
    """
    weights = model.get_weights()

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
        if i >= num_batches:
            break
        x, y_true = x.to(device, non_blocking=True), y_true.to(device, non_blocking=True)

        model.zero_grad(set_to_none=True)
        y_pred = model(x)
        per_sample_mse = (y_pred - y_true).pow(2)
        if per_sample_mse.ndim > 2:
            per_sample_mse = per_sample_mse.flatten(1)
        loss = per_sample_mse.mean(1).sum()
        loss.backward()

        with torch.no_grad():
            for (m, name) in weights:
                p = getattr(m, name)
                if p.grad is not None:
                    scores[(m, name)].add_((p.grad * p).abs())
        seen_samples += x.size(0)

    if seen_samples > 0:
        for k in scores:
            scores[k].div_(float(seen_samples))

    return scores


def _compute_fisher_importance(model, device, dataset, batch_size=32, num_batches=128, **kwargs):
    """
    Fisher importance (diagonal approximation): E[grad^2], averaged over samples.
    Single-GPU version.
    """
    weights = model.get_weights()

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
        if i >= num_batches:
            break
        x, y_true = x.to(device, non_blocking=True), y_true.to(device, non_blocking=True)

        model.zero_grad(set_to_none=True)
        y_pred = model(x)
        per_sample_mse = (y_pred - y_true).pow(2)
        if per_sample_mse.ndim > 2:
            per_sample_mse = per_sample_mse.flatten(1)
        loss = per_sample_mse.mean(1).sum()
        loss.backward()

        with torch.no_grad():
            for (m, name) in weights:
                p = getattr(m, name)
                if p.grad is not None:
                    scores[(m, name)].add_(p.grad.pow(2))
        seen_samples += x.size(0)

    if seen_samples > 0:
        for k in scores:
            scores[k].div_(float(seen_samples))

    return scores
