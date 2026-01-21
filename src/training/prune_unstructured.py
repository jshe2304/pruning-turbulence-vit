import os
import torch
import torch.nn.utils.prune as prune
import torch.distributed as dist

from .finetune import finetune
from .utils import compute_importance_scores

def prune_unstructured(
    model, device, 
    optimizer_state,
    train_dataset, validation_dataset, 
    importance_metric, prune_schedule, epoch_schedule,  
    output_dir, logger=None,
    **finetune_config,
    ):
    """
    Prune the model using unstructured pruning. 

    Args:
        model: The model to prune
        device: The device to use
        optimizer_state: The optimizer state to use
        train_dataset: The training dataset
        validation_dataset: The validation dataset
        importance_metric: The importance metric to use
        prune_schedule: The schedule of pruning amounts
        epoch_schedule: The schedule of epochs to prune
        logger: The wandb logger
        **finetune_config: Finetuning configuration (see `finetune.py`)
    """

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # Create pruned models directory (only on rank 0)

    if local_rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    # Iterative pruning

    for p, epochs in zip(prune_schedule, epoch_schedule):

        # Prune

        importance_scores = compute_importance_scores(
            importance_metric, 
            model, device, 
            dataset=train_dataset,
            optimizer_state=optimizer_state,
        )

        pruning_method = prune.RandomUnstructured if importance_metric == 'random' else prune.L1Unstructured

        prune.global_unstructured(
            model.module.get_weights(),
            pruning_method=pruning_method, 
            importance_scores=importance_scores,
            amount=p, 
        )

        # Optionally restart optimizer

        if finetune_config['restart_optimizer'] and p > 0:
            optimizer_state = None 
        
        # Finetune
        
        optimizer_state = finetune(
            model, device, 
            optimizer_state, 
            train_dataset, validation_dataset, 
            **finetune_config, 
            epochs=epochs, 
            logger=logger, checkpoint_dir=output_dir
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
                os.path.join(output_dir, f'{(100 * proportion_remaining):.2f}.tar')
            )
