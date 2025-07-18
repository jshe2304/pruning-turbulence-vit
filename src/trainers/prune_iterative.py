import torch.nn.utils.prune as prune

from .train import train

def prune_iterative(
    model, device, 
    train_dataset, validation_dataset, 
    n_prune_iterations, prune_amount,
    logger=None,
    **finetune_config,
    ):
    """
    Prune the model iteratively. 

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

    for i in range(n_prune_iterations):

        # Log

        unpruned_parameters, _ = model.n_unpruned_parameters()
        if logger is not None:
            logger.log({
                "unpruned_parameters": unpruned_parameters,
            })

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
            logger=logger
        )
