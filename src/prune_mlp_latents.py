"""
A training loop following the lottery ticket pruning strategy. 
Here, we prune an individual attention head each iteration. 
See https://arxiv.org/pdf/1903.01611

1. Train a model for k epochs
2. Prune the model
3. Reinitialize to epoch k's weights
4. Repeat 1-3 until the model is pruned to the desired size
"""

import sys
import toml

import torch
from torch.optim import AdamW
from models.vit import ViT

from utils.data import TimeSeriesDataset
from utils.training import sample_loss, train_one_epoch

########################
# Iterative pruning loop
########################

def get_least_significant_mlp_latent(model, train_dataset, device):
    """
    Identify the least significant MLP latent dimension. 
    """

    min_loss = float('inf')
    layer, latent_index = None, None

    for l in range(len(model.transformer_blocks)):
        for i in range(model.transformer_blocks[l].attn.n_heads):

            latent_mask = model.transformer_blocks[l].mlp.latent_mask

            # Check if the latent is already pruned

            if latent_mask[i]: continue

            # Prune the latent, evaluate the loss, and unprune

            latent_mask[i] = False
            loss = sample_loss(
                model, train_dataset, 
                n_samples=4096, batch_size=128, 
                device=device
            )
            latent_mask[i] = True

            # Update the minimum loss/latent

            if loss < min_loss:
                min_loss, layer, latent_index = loss, l, i

    return layer, latent_index

train_losses = []
validation_losses = []

for prune_iteration in range(8):

    # Retrain

    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-7)
    for epoch in range(EPOCHS):
        train_one_epoch(model, train_dataloader, optimizer, device=DEVICE)

    # Log losses

    train_losses.append(sample_loss(model, train_dataset.data, device=DEVICE))
    validation_losses.append(sample_loss(model, validation_dataset.data, device=DEVICE))

    # Identify least signficant MLP latent dimension

    layer, latent_index = get_least_significant_mlp_latent(model, train_dataset, validation_dataset, device)
    # Prune and reinitialize

    model.transformer_blocks[layer].mlp.latent_mask[latent_index] = False

    model.load_state_dict(initialization)


if __name__ == '__main__':

    # Load config

    config_path = sys.argv[1]
    config = toml.load(config_path)

    # Device

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_gpus = torch.cuda.device_count()

    # Initialize model

    model = ViT(**config['model'])

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=list(range(n_gpus)))
    model.to(device)

    # Initialize data

    train_dataset = TimeSeriesDataset(**config['train_dataset'])
    validation_dataset = TimeSeriesDataset(**config['validation_dataset'])

    # Initialization train loop

    train()

    initialization = model.state_dict()

    # Train

    train(
        model, device, 
        train_dataset, validation_dataset, 
        **config['training']
    )