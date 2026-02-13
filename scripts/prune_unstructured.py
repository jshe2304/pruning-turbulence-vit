"""
Vision Transformer iterative pruning script.
Here, we implement a prune-finetune-repeat loop on a pretrained model.

To run, pass in:
- A path to a TOML pruning config file to start a new pruning run
  (config must specify checkpoint_file pointing to trained model)
- A path to an output directory to resume from a previous pruning run
"""

import sys
import toml
import os
import wandb

import torch
import torch.distributed as dist
import torch.nn.utils.prune as prune
from torch.nn.parallel import DistributedDataParallel as DDP

from src.models import create_model
from src.data.py2d_dataset import Py2DDataset
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

    # Create output and checkpoint directories (only rank 0 process)

    output_dir = config['output_dir']
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    if local_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize wandb

    logger = wandb.init(
        project="turbulence-vit-prune",
        group=config['pruning']['importance_metric'],
        id=config.get('wandb_id', None),
        config=config,
        resume='allow'
    ) if local_rank == 0 else None

    # Save config with wandb_id (only for new runs, rank 0 only)

    if 'wandb_id' not in config: 
        config['wandb_id'] = logger.id
        config_path = os.path.join(output_dir, 'config.toml')
        with open(config_path, 'w') as f:
            toml.dump(config, f)

    if world_size > 1: dist.barrier()

    # Adjust batch size for distributed training

    config['finetuning']['batch_size'] //= world_size

    # Unpack state dict

    state_dict = torch.load(config['checkpoint_file'], map_location=device, weights_only=False)
    optimizer_state = state_dict.pop('optimizer_state', None)
    model_state_dict = state_dict.pop('model_state', state_dict)

    # Initialize model

    model = create_model(**config['model']).to(device)
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

    train_dataset = Py2DDataset(**config['train_dataset'])
    validation_dataset = Py2DDataset(**config['validation_dataset'])

    # Prune model

    prune_unstructured(
        model, device,
        optimizer_state,
        train_dataset, validation_dataset,
        checkpoint_dir=checkpoint_dir,
        logger=logger,
        **config['pruning'],
        **config['finetuning']
    )

    # Clean up logger

    if local_rank == 0 and logger is not None: 
        logger.finish()

    dist.destroy_process_group()

if __name__ == '__main__':

    path = sys.argv[1]

    # Start new pruning run from a config file
    if path.endswith('.toml'):
        prune_config = toml.load(path)

        checkpoint_path = prune_config['checkpoint_file']
        assert os.path.isfile(checkpoint_path)
        model_dir = os.path.dirname(os.path.dirname(checkpoint_path))
        model_config_path = os.path.join(model_dir, 'config.toml')
        model_config = toml.load(model_config_path)

        del model_config['training']
        config = model_config | prune_config
    # Resume pruning run from existing prune run directory
    elif os.path.isdir(path):
        config = toml.load(os.path.join(path, 'config.toml'))
        config['checkpoint_file'] = os.path.join(path, 'checkpoints/last.tar')
        config['pruning']['prune_schedule'].insert(0, 0.0)
    else:
        raise ValueError(f"Invalid path: {path}")

    main(config)
