import os
import torch
import torch.nn.utils.prune as prune
from torch.optim import AdamW

from .finetune import cosine_finetune

def prune_unstructured(
    model, device, 
    train_dataset, validation_dataset, 
    prune_schedule, epoch_schedule,
    output_dir, logger=None,
    **finetune_config,
    ):
    """
    Prune the model using unstructured pruning. 

    Args:
        model: The model to prune
        device: The device to use
        train_dataset: The training dataset
        validation_dataset: The validation dataset
        prune_schedule: The schedule of pruning amounts
        epoch_schedule: The schedule of epochs to prune
        logger: The wandb logger
        **finetune_config: Finetuning configuration (see `finetune.py`)
    """

    local_rank = int(os.environ["LOCAL_RANK"])

    # Create pruned models directory (only on rank 0)

    if local_rank == 0:
        checkpoint_dir = os.path.join(output_dir, 'pruned_models')
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Iterative pruning

    total_parameters = model.module.n_parameters()
    epochs_completed = 0
    for p, epochs in zip(prune_schedule, epoch_schedule):

        # Prune

        prune.global_unstructured(
            model.module.get_parameters_to_prune(),
            pruning_method=prune.L1Unstructured,
            amount=p, 
        )

        # Optionally create persistent optimizer

        optimizer = None
        if not finetune_config['restart_optimizer']:
            optimizer = AdamW(model.module.parameters(), lr=finetune_config['lr'], weight_decay=finetune_config['weight_decay'])

        # Finetune

        cosine_finetune(
            model, device, 
            train_dataset, validation_dataset, 
            **finetune_config, 
            optimizer=optimizer, 
            epochs=epochs, 
            logger=logger,
        )

        # Logging

        if local_rank == 0:

            # Log losses to wandb

            if logger is not None:
                pruned_parameters = model.module.n_pruned_parameters()
                unpruned_parameters = total_parameters - pruned_parameters
                proportion_remaining = unpruned_parameters / total_parameters
                logger.log(
                    {
                        "unpruned_parameters": unpruned_parameters, 
                        "proportion_remaining": proportion_remaining,
                    },
                    step=epochs_completed
                )

            # Save model

            torch.save(
                model.module.state_dict(), 
                os.path.join(checkpoint_dir, f'{int(proportion_remaining * 100)}.pt')
            )

        epochs_completed += epochs
