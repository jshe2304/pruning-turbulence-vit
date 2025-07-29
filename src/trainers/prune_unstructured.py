import os
import torch
import torch.nn.utils.prune as prune

from .finetune import finetune

def prune_unstructured(
    model, device, 
    train_dataset, validation_dataset, 
    n_prune_iterations, prune_amount,
    logger=None,
    **finetune_config,
    ):
    """
    Prune the model using unstructured pruning. 

    Args:
        model: The model to prune
        device: The device to use
        train_dataset: The training dataset
        validation_dataset: The validation dataset
        n_prune_iterations: The number of pruning iterations
        prune_amount: The amount of pruning to apply
        logger: The wandb logger
        **finetune_config: Finetuning configuration (see `train.py`)
    """

    # Create pruned models directory

    checkpoint_dir = os.path.join(finetune_config['output_dir'], 'pruned_models')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Iterative pruning

    total_parameters = model.n_parameters()
    for i in range(n_prune_iterations):

        # Prune

        prune.global_unstructured(
            model.get_parameters_to_prune(),
            pruning_method=prune.L1Unstructured,
            amount=prune_amount, 
        )

        # Logging

        pruned_parameters = model.n_pruned_parameters()
        unpruned_parameters = total_parameters - pruned_parameters
        if logger is not None:
            logger.log(
                {
                    "unpruned_parameters": unpruned_parameters,
                    "fraction_unpruned": unpruned_parameters / total_parameters,
                },
                step=i * finetune_config['epochs']
            )

        # Finetune

        finetune(
            model, device, 
            train_dataset, validation_dataset, 
            **finetune_config,
            logger=logger
        )

        # Save model

        torch.save(
            model.state_dict(), 
            os.path.join(checkpoint_dir, f'iteration_{i}.pt')
        )
