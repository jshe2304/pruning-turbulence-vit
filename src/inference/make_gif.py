import os

import imageio
from io import BytesIO

import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from .rollout import n_step_rollout

def make_gif(model, test_dataset, output_file, time_per_frame=0.01, device='cpu'):

    # Make test dataloader

    dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=2)
    batch = next(iter(dataloader))

    inputs, targets = batch[0].to(device, dtype=torch.float32), batch[1].to(device, dtype=torch.float32)
    ic = inputs[0].unsqueeze(dim=0)
    n_steps = inputs.shape[0]

    pred = n_step_rollout(model, ic, n=n_steps, train_tendencies=False)

    # Extract fields from rollout

    pred = pred.transpose(-1,-2).squeeze().detach().cpu().numpy()
    pred_u = pred[:, 0]
    pred_v = pred[:, 1]

    targets = targets.transpose(-1,-2).squeeze().detach().cpu().numpy()
    tar_u = targets[:, 0]
    tar_v = targets[:, 1]

    frames = []
    for t in range(pred.shape[0]):

        print(t)

        fig, axs = plt.subplots(2, 2, figsize=(4, 4), sharex=True, sharey=True)

        axs[0, 0].imshow(pred_u[t], cmap='bwr', vmin=-5, vmax=5, aspect='equal')
        axs[0, 0].set_title('Predicted $u$')

        axs[0, 1].imshow(tar_u[t], cmap='bwr', vmin=-5, vmax=5, aspect='equal')
        axs[0, 1].set_title('Target $u$')

        axs[1, 0].imshow(pred_v[t], cmap='bwr', vmin=-5, vmax=5, aspect='equal')
        axs[1, 0].set_title('Predicted $v$')

        axs[1, 1].imshow(tar_v[t], cmap='bwr', vmin=-5, vmax=5, aspect='equal')
        axs[1, 1].set_title('Target $v$')

        for ax in axs.flatten(): 
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(f't = {t}')

        # Add png to frames

        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))
        buf.close()

    # Make sure parent directory is created then save file

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    imageio.mimsave(output_file, frames, duration=time_per_frame)