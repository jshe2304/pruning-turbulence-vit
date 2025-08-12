import os
import torch
import torch.nn.utils.prune as prune
from torch.optim import AdamW

from .finetune import finetuners

def prune_unstructured(
    model, device, 
    train_dataset, validation_dataset, 
    prune_schedule, epoch_schedule,
    finetuner_type, 
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

    checkpoint_dir = None
    if local_rank == 0:
        checkpoint_dir = os.path.join(output_dir, 'pruned_models')
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Iterative pruning

    for iteration, (p, epochs) in enumerate(zip(prune_schedule, epoch_schedule)):

        # Prune

        prune.global_unstructured(
            model.module.get_parameters_to_prune(),
            pruning_method=prune.L1Unstructured,
            amount=p, 
        )

        # Load optimizer state from previous run (if any)

        if iteration == 0:
            optimizer_state = getattr(model.module, 'optimizer_state', None)
            delattr(model.module, 'optimizer_state')
            print('Loaded optimizer state.')

        # Optionally restart optimizer

        if finetune_config['restart_optimizer'] and p > 0:
            optimizer_state = None
        
        # Finetune
        
        optimizer_state = finetuners[finetuner_type](
            model, device, 
            train_dataset, validation_dataset, 
            **finetune_config, 
            optimizer_state=optimizer_state, 
            epochs=epochs, 
            logger=logger, 
            output_dir=checkpoint_dir,
            coast=(p == 0.)
        )

        # Logging

        if local_rank == 0:

            total_parameters = model.module.n_parameters()
            pruned_parameters = model.module.n_pruned_parameters()
            proportion_remaining = 1 - pruned_parameters / total_parameters

            torch.save(
                {
                    'model_state': model.module.state_dict(), 
                    'optimizer_state': optimizer_state
                }, 
                os.path.join(checkpoint_dir, f'{int(100 * proportion_remaining)}.tar')
            )
