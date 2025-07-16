import torch.nn.utils.prune as prune

from .train import train

def prune_iterative(
    model, device, 
    train_dataset, validation_dataset, 
    n_prune_iterations, prune_amount,
    **finetune_config,
    ):

    for i in range(n_prune_iterations):

        # Prune

        prune.global_unstructured(
            model.get_parameters_to_prune(),
            pruning_method=prune.L1Unstructured,
            amount=prune_amount,
        )

        # Finetune

        train(
            model, device, 
            train_dataset, validation_dataset, 
            **finetune_config,
        )