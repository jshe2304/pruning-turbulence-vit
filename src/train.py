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

from models.vit import ViT

from utils.data import get_time_series_dataloader
from utils.training import sample_loss

########
# Config
########

device = 'cuda'

log_dir = '/glade/derecho/scratch/jshen/turbulence_vit_logs/'
checkpoint_dir = '/glade/derecho/scratch/jshen/turbulence_vit_checkpoints/'

######
# Data
######

data_dir = '/glade/derecho/scratch/dpatel/2DTurbData/results/Re5000_fkx0fky4_r0.1_b20/NoSGS/NX256/dt0.0002_IC1'

train_frames = list(range(200000, 210001)) + list(range(995000, 1004998))
validation_frames = list(range(310000, 312001))

train_dataset, train_dataloader = get_time_series_dataloader(data_dir, train_frames, 3, 64)
validation_dataset, validation_dataloader = get_time_series_dataloader(data_dir, validation_frames, 3, 64)

#######
# Model
#######

model = ViT(
    in_channels=2,
    d_embed=192,
    n_heads=8, 
    n_layers=8, 
    img_shape=(256, 256), 
    patch_shape=(4, 4), 
)
model = torch.nn.DataParallel(model)
model = model.to(device)

###########################
# Optimizers and schedulers
###########################

optimizer = AdamW(model.parameters(), lr=0.0005, weight_decay=1e-7)

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
    for img, target in train_dataloader:
        optimizer.zero_grad()

        pred = model(img)
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        optimizer.step()

    if epoch < 3: warmup_scheduler.step()
    plateau_scheduler.step(validation_losses[-1])

    train_losses.append(sample_loss(model, train_dataset))
    validation_losses.append(sample_loss(model, validation_dataset))

    with open(log_dir + 'train_losses.txt', 'a') as f:
        f.write(f'{train_losses[-1]}\n')
    with open(log_dir + 'validation_losses.txt', 'a') as f:
        f.write(f'{validation_losses[-1]}\n')

    if epoch % 25 == 0:
        torch.save(model.state_dict(), checkpoint_dir + f'checkpoint_{epoch}.pt')
