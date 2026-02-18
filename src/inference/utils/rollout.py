import torch

def n_step_rollout(model, ic, n=1, log_re=None):
    """Produce an n-step forward roll-out
    prediction.
    Args:
        model: trained pytorch model.
        ic: [B=1, C, T, X, Y] initial condition for prediction.
        n (int): number of steps to predict for.
        log_re: (B,) log Reynolds number tensor, or None.
    Returns:
        pred: [B=n, C, T, X, Y] n-step model prediction (time along dim=0)."""

    pred = []

    for _ in range(n):
        pred_temp, ic = single_step_rollout(model, ic, log_re=log_re)
        pred.append(pred_temp)

    pred = torch.cat(pred, dim=0)

    return pred

def single_step_rollout(model, ic, log_re=None):
    """Produce a single-step forward roll-out prediction.
    Args:
        model: trained pytorch model.
        ic: [B=1, C, T, X, Y] initial condition for prediction.
        log_re: (B,) log Reynolds number tensor, or None.
    Returns:
        pred: [B=1, C, T, X, Y] single-step model prediction.
        ic: [B=1, C, T, X, Y] initial condition for next time step.
    """

    with torch.no_grad():
        idx = torch.tensor([0], device=ic.device)
        out = model(ic, log_re) if log_re is not None else model(ic)
        if ic.shape[2] > 1:
            pred = torch.index_select(out, 2, index=idx)
            ic = torch.cat([pred, ic[:, :, :-1, :, :]], dim=2)
        else:
            pred = torch.index_select(out, 2, index=idx)
            ic = pred

    return pred, ic
