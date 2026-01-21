"""
Vision Transformer iterative pruning script. 
Here, we implement a prune-finetune-repeat loop on a pretrained model. 
Does not support distributed training. 

To run, pass in a path to a TOML config file as an argument. 
The TOML should contain the following sections:
- model
- train_dataset
- validation_dataset
- pruning
- finetuning
- output_dir
"""

import sys
import toml
import os
import wandb

import torch
import torch.distributed as dist
import torch.nn.utils.prune as prune
from torch.nn.parallel import DistributedDataParallel as DDP

from src.models.vision_transformer import ViT
from src.data.datasets import TimeSeriesDataset
from src.training.prune_unstructured import prune_unstructured

torch.autograd.set_detect_anomaly(True)

def get_masks(model):
    return [
        getattr(
            module, param + '_mask', 
            torch.ones_like(getattr(module, param))
        )
        for module, param in model.get_weights()
    ]

def bake_masks(model):
    for module, param in model.get_weights():
        if hasattr(module, param + '_mask'):
            prune.remove(module, param)

def main(config: dict):

    # Set up distributed training

    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Initialize wandb

    logger = wandb.init(
        project="turbulence-vit-prune",
        group=config['pruning']['importance_metric'],
        id=config.get('wandb_id', None), 
        config=config,
        resume='allow'
    ) if local_rank == 0 else None

    # Adjust batch size for distributed training

    config['finetuning']['batch_size'] //= world_size

    # Unpack state dict

    state_dict = torch.load(config['checkpoint_file'], map_location=device, weights_only=False)
    optimizer_state = state_dict.pop('optimizer_state', None)
    model_state_dict = state_dict.pop('model_state', state_dict)

    # Initialize model

    model = ViT(**config['model']).to(device)
    model.load_state_dict(model_state_dict)
    masks = get_masks(model) # get masks
    bake_masks(model) # bake in masks to remove buffers
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    for mask, (module, param) in zip(masks, model.module.get_weights()):
        prune.custom_from_mask(module, param, mask) # re-apply masks after DDP
    del masks

    # Adjust target steps for rollout losses

    config['train_dataset']['target_step'] *= config['finetuning']['num_rollout_steps']
    config['validation_dataset']['target_step'] *= config['finetuning']['num_rollout_steps']

    # Initialize datasets

    train_dataset = TimeSeriesDataset(**config['train_dataset'])
    validation_dataset = TimeSeriesDataset(**config['validation_dataset'])

    # Prune model

    prune_unstructured(
        model, device, 
        optimizer_state,
        train_dataset, validation_dataset, 
        **config['pruning'], 
        **config['finetuning'], 
        output_dir=config['output_dir'],
        logger=logger
    )

    # Clean up

    if local_rank == 0 and logger is not None: 
        logger.finish()

    dist.destroy_process_group()

if __name__ == '__main__':

    config_path = sys.argv[1]
    config = toml.load(config_path)

    main(config)
