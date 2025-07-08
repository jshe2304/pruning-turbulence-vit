"""
A training loop following the lottery ticket pruning strategy. 
Here, we prune an individual attention head each iteration. 
See https://arxiv.org/pdf/1903.01611

1. Train a model for k epochs
2. Prune the model
3. Reinitialize to epoch k's weights
4. Repeat 1-3 until the model is pruned to the desired size
"""

import torch
from torch.optim import AdamW
from models.vit import ViT

from utils.data import load_data
from utils.train import train_one_epoch, sample_loss

# Device

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training parameters

EPOCHS = 100
INIT_EPOCHS = 10

# Data

TRAIN_DATADIR = '/scratch/midway3/jshe/2d_turbulence'
VALIDATION_DATADIR = '/scratch/midway3/jshe/2d_turbulence_validation'

###########
# Load data
###########

train_dataset = load_data(TRAIN_DATADIR)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)

validation_dataset = load_data(VALIDATION_DATADIR)
validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=32)

#######
# Model
#######

model = ViT(
    img_shape=(1, 256, 256),
    patch_shape=(1, 16, 16),
).to(DEVICE)

########################
# Initial training loop
########################

optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-7)
for epoch in range(INIT_EPOCHS):
    train_one_epoch(model, train_dataloader, optimizer, device=DEVICE)

initialization = model.state_dict()

########################
# Iterative pruning loop
########################

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

    min_loss = float('inf')
    layer, latent_index = None, None

    for l in range(len(model.transformer_blocks)):
        for i in range(model.transformer_blocks[l].attn.n_heads):

            latent_mask = model.transformer_blocks[l].mlp.latent_mask

            if latent_mask[i]: continue

            latent_mask[i] = False
            loss = sample_loss(model, train_dataset.data, device=DEVICE)
            latent_mask[i] = True

            if loss < min_loss:
                min_loss = loss
                layer, latent_index = l, i

    # Prune and reinitialize

    model.transformer_blocks[layer].mlp.latent_mask[latent_index] = False

    model.load_state_dict(initialization)
