"""
A basic training script for a vision transformer model, implementing:
- Learning rate warmup
- Learning rate plateau
- AdamW optimizer
- Weight decay
"""

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.vision_transformer import ViT

from utils.data import load_data

########
# Config
########

device = 'cuda' if torch.cuda.is_available() else 'cpu'

######
# Data
######

datadir = '/scratch/midway3/jshe/2d_turbulence'
dataloader = load_data(datadir)

#######
# Model
#######

model = ViT(
    img_shape=(2, 256, 256),
    patch_shape=(2, 16, 16),
    in_channels=2,
    d_encoder_embed=192,
    n_encoder_layers=4,
    n_encoder_heads=4,
    d_decoder_embed=96,
    n_decoder_layers=4,
    n_decoder_heads=4,
).to(device)

###########################
# Optimizers and schedulers
###########################

optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-7)
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.001, total_iters=3
)
plateau_scheduler = ReduceLROnPlateau(
    optimizer, factor=0.2, patience=5, mode='min'
)

##########
# Training
##########

train_losses = []
validation_losses = []

for epoch in range(1000):
    model.train()
    for img, target in dataloader:
        optimizer.zero_grad()

        pred = model(img.to(device))
        loss = torch.nn.functional.mse_loss(pred, target.to(device))
        loss.backward()
        optimizer.step()

        if epoch < 3: warmup_scheduler.step()
        plateau_scheduler.step(validation_losses[-1])

    validation_losses.append(torch.randn(1))
    train_losses.append(torch.randn(1))
