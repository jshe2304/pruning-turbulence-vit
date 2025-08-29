import os
import torch

from .finetune import finetune
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
        **finetune_config: Finetuning configuration (see `finetune.py`)
    """

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # Create pruned models directory (only on rank 0)

    if local_rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    # Iterative pruning

    for iteration in range(num_iterations):

        # Prune
        if iteration > 0:
            layer, head_index = prune_attention_head(model, train_dataset, device)
            print(f'Pruned head {head_index} in layer {layer}')
    
        # Optionally restart optimizer

        if finetune_config['restart_optimizer'] and iteration > 0:
            optimizer_state = None

        # Finetune
        
        optimizer_state = finetune(
            model, device, 
            optimizer_state, 
            train_dataset, validation_dataset, 
            **finetune_config, 
            logger=logger, checkpoint_dir=output_dir
        )

        # Logging

        if local_rank == 0 and iteration > 0:
            torch.save(
                {
                    'model_state': model.module.state_dict(), 
                    'optimizer_state': optimizer_state
                }, 
                os.path.join(output_dir, f'{num_pruned_heads(model)}.tar')
            )
