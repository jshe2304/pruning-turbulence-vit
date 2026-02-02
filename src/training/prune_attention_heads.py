'''

'''

import os
import torch

from .train import train
from .utils import prune_attention_head, num_pruned_heads

def prune_attention_heads(
    model, device, 
    optimizer_state,
    train_dataset, validation_dataset, 
    num_iterations,
    output_dir, logger=None,
    **finetune_config,
    ):
    """
    Iterative pruning of attention heads. 

    Args:
        model: The model to prune
        device: The device to use
        train_dataset: The training dataset
        validation_dataset: The validation dataset
        num_iterations: The number of iterations to prune
        logger: The wandb logger
        **finetune_config: Training configuration (see `train.py`)
    """

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # Create pruned models directory (only on rank 0)

    if local_rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    # Iterative pruning

    for iteration in range(num_iterations):

        prune_head = (
            iteration > 0 or # Not first iteration
            (iteration == 0 and optimizer_state is None) or # Starting from scratch
            (
                optimizer_state is not None and 
                optimizer_state['param_groups'][0]['lr'] <= finetune_config['early_stop_lr_threshold']
            ) # Checkpoint doesn't need finetuning
        )

        # Prune
        if prune_head:
            layer, head_index = prune_attention_head(model, train_dataset, device)
            print(f'Pruned head {head_index} in layer {layer}')
    
        # Optionally restart optimizer

        if finetune_config['restart_optimizer'] and prune_head:
            optimizer_state = None

        # Finetune
        
        optimizer_state = train(
            model, device,
            train_dataset, validation_dataset,
            output_dir=output_dir,
            optimizer_state=optimizer_state,
            save_best=False,
            logger=logger,
            **finetune_config,
        )

        # Logging

        if local_rank == 0:
            torch.save(
                {
                    'model_state': model.module.state_dict(), 
                    'optimizer_state': optimizer_state
                }, 
                os.path.join(output_dir, f'{num_pruned_heads(model)}.tar')
            )
