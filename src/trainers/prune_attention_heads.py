import os
import torch

from .finetune import finetuners
from .utils import prune_attention_head

def prune_attention_heads(
    model, device, 
    train_dataset, validation_dataset, 
    num_iterations,
    finetuner_type, 
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

    local_rank = int(os.environ["LOCAL_RANK"])

    # Create pruned models directory (only on rank 0)

    checkpoint_dir = None
    if local_rank == 0:
        checkpoint_dir = os.path.join(output_dir, 'pruned_models')
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Iterative pruning

    for iteration in range(num_iterations):

        # Prune

        layer, head_index = prune_attention_head(model, train_dataset, device)
        print(f'Pruned head {head_index} in layer {layer}')

        # Load optimizer state from previous run (if any)

        if iteration == 0:
            optimizer_state = getattr(model.module, 'optimizer_state', None)
            if optimizer_state is not None:
                delattr(model.module, 'optimizer_state')
            print('Loaded optimizer state.')

        # Optionally restart optimizer

        if finetune_config['restart_optimizer'] and iteration > 0:
            optimizer_state = None
        
        # Finetune
        
        optimizer_state = finetuners[finetuner_type](
            model, device, 
            train_dataset, validation_dataset, 
            **finetune_config, 
            optimizer_state=optimizer_state, 
            logger=logger, 
            output_dir=checkpoint_dir,
            coast=False
        )

        # Logging

        if local_rank == 0:
            torch.save(
                {
                    'model_state': model.module.state_dict(), 
                    'optimizer_state': optimizer_state
                }, 
                os.path.join(checkpoint_dir, f'{iteration}.tar')
            )
