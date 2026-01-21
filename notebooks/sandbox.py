"""
Single-process DDP reproduction script for pruning + rollout training.

This mirrors the critical parts of script_prune_unstructured.py and trainers, but
keeps WORLD_SIZE=1 and runs a short training loop to try reproducing in-place
autograd errors under pruning reparametrization.

Usage:
    python repro_prune_unstructured_ddp.py /abs/path/to/config.toml --max-batches 4
"""

import os
import sys

import toml

import torch
import torch.distributed as dist
import torch.nn.utils.prune as prune
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Ensure local imports work regardless of CWD
sys.path.append(os.path.dirname(__file__))

from models.vision_transformer import ViT  # noqa: E402
from data.datasets import TimeSeriesDataset  # noqa: E402

def init_single_process_ddp() -> tuple[torch.device, int]:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"

    dist.init_process_group(
        backend=backend,
        init_method="env://",
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(device)

    return device, local_rank

def get_masks(model):
    masks = [
        getattr(
            module, param + '_mask', 
            torch.ones_like(getattr(module, param))
        )
        for module, param in model.get_weights()
    ]
    
    return masks

def bake_masks(model):
    for module, param in model.get_weights():
        prune.remove(module, param)

def main() -> None:
    #torch.autograd.set_detect_anomaly(True)

    config_path = '/glade/u/home/jshen/pruning-turbulence-vit/src/config/prune/l1_rollout.toml'
    config = toml.load(config_path)
    config["checkpoint_file"] = '/glade/derecho/scratch/jshen/prune_base/l1_rollout/checkpoints/94.11.tar'

    device, local_rank = init_single_process_ddp()

    state_dict = torch.load(config["checkpoint_file"], map_location=device, weights_only=False)
    model_state_dict = state_dict.pop("model_state", state_dict)
    optimizer_state = state_dict.pop("optimizer_state", None)
    optimizer_state = None
    
    # Initialize model
    model = ViT(**config["model"]).to(device)
    model.load_state_dict(model_state_dict)
    masks = get_masks(model)
    bake_masks(model)
    
    # Wrap with DDP
    model = DDP(
        model,
        device_ids=[local_rank] if device.type == "cuda" else None,
        output_device=local_rank if device.type == "cuda" else None,
    )

    for mask, (module, param) in zip(masks, model.module.get_weights()):
        prune.custom_from_mask(module, param, mask)

    print(model.module.n_pruned_parameters())

    # Prune model
    prune.global_unstructured(
        model.module.get_weights(),
        pruning_method=prune.L1Unstructured,
        importance_scores=None,
        amount=0.06,
    )

    # Adjust dataset target_step for rollout
    config["train_dataset"]["target_step"] *= config["finetuning"]["num_rollout_steps"]

    # Data
    train_dataset = TimeSeriesDataset(**config["train_dataset"])
    sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=2, sampler=sampler, num_workers=4)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['finetuning']['lr'], weight_decay=config['finetuning']['weight_decay'])
    if optimizer_state is not None: optimizer.load_state_dict(optimizer_state)

    # Short training loop to attempt to trigger autograd in-place errors
    sampler.set_epoch(0)
    model.train()
    for i, (ic, target) in enumerate(train_loader):
        if i == 4: break
        ic, target = ic.to(device), target.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Mirror train_one_epoch rollout behavior
        for _ in range(config['finetuning']['num_rollout_steps']):
            y_pred = model(ic)
            prev_ic = ic[:, :, :-1, :, :].contiguous()
            ic = torch.cat([y_pred, prev_ic], dim=2)

        loss = torch.nn.functional.mse_loss(y_pred, target)
        loss.backward()
        print(loss.item())
        optimizer.step()

    # Cleanup
    if dist.is_initialized(): dist.destroy_process_group()

if __name__ == "__main__":
    main()


