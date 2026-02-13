import os
import torch
import torch.nn.utils.prune as prune
import torch.distributed as dist

from .train import train
from .utils import compute_importance_scores

def prune_unstructured(
    model, device,
    optimizer_state,
    train_dataset, validation_dataset,
    importance_metric, prune_schedule, epoch_schedule,
    checkpoint_dir, logger=None,
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
        checkpoint_dir: The directory to save checkpoints
        logger: The wandb logger
        **finetune_config: Training configuration (see `train.py`)
    """

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # Iterative pruning

    for iteration, (p, epochs) in enumerate(zip(prune_schedule, epoch_schedule)):

        if local_rank == 0:
            print(f"\n{'='*60}")
            print(f"PRUNING ITERATION {iteration + 1}/{len(prune_schedule)}")
            print(f"Pruning {p:.1%} of remaining weights, then finetuning for {epochs} epochs")
            print(f"{'='*60}")

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

        if local_rank == 0:
            total = model.module.n_parameters()
            pruned = model.module.n_pruned_parameters()
            print(f"Pruned {pruned:,} / {total:,} parameters ({pruned/total:.1%} sparsity)")

        # Optionally restart optimizer

        if finetune_config['restart_optimizer'] and p > 0:
            optimizer_state = None 
        
        # Finetune
        
        optimizer_state = train(
            model, device,
            train_dataset, validation_dataset,
            checkpoint_dir=checkpoint_dir,
            optimizer_state=optimizer_state,
            save_best=False,
            epochs=epochs,
            logger=logger,
            **finetune_config,
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
                os.path.join(checkpoint_dir, f'{(100 * proportion_remaining):.2f}.tar')
            )
