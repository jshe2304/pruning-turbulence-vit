import os

import numpy as np

from torch.utils.data import DataLoader

from .rollout import single_step_rollout

def make_inference(model, dataset, inference_length, output_dir, device):

    # Make output directory

    os.makedirs(output_dir, exist_ok=True)

    # Get initial condition

    dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)
    input, _ = next(iter(dataloader))
    ic = input[0].unsqueeze(dim=0).to(device)

    # Emulate and save

    for i in range(inference_length):
        pred, ic = single_step_rollout(model, ic, train_tendencies=False)

        pred = pred.clone().transpose(-1,-2).squeeze().detach().cpu().numpy()

        # Unnormalize data
        pred[0,:] = (pred[0,:]  * dataset.std[0]) + dataset.mean[0]
        pred[1,:] = (pred[1,:]  * dataset.std[1]) + dataset.mean[1]

        output_file = os.path.join(output_dir, f'{i}.npy')
        np.save(output_file, pred)
