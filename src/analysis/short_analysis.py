import numpy as np
import torch
from .metrics import get_rmse, get_acc, get_spectra
from .rollout import n_step_rollout

from torch.utils.data import DataLoader

def perform_short_analysis(model, test_dataset, climo_dataset, max_leadtime, n_ensembles, device):
    """
    Perform short-run analyses. 
    Returns a dictionary of results. 
    """

    # Make test dataloader

    dataloader = DataLoader(test_dataset, batch_size=max_leadtime, shuffle=False, num_workers=2)

    # Make climo data

    climo_dataloader = DataLoader(climo_dataset, batch_size=len(climo_dataset), shuffle=False, num_workers=2)
    climo_field, _ = next(iter(climo_dataloader))
    climo_field = climo_field.transpose(-1, -2).squeeze().detach().cpu().numpy()
    climo_u = climo_field[:,0].mean(axis=0)
    climo_v = climo_field[:,1].mean(axis=0)

    # Compute metrics

    results = {}

    rmse_u, rmse_u_per, rmse_v, rmse_v_per = [], [], [], []
    acc_u, acc_u_per, acc_v, acc_v_per = [], [], [], []
    spectra_list, spectra_tar_list, wavenumbers_list = [], [], None

    for i, batch in enumerate(dataloader):
        if i > n_ensembles: break

        print(f'Starting iteration {i}...', end='')
        
        inputs, targets = batch[0].to(device, dtype=torch.float32), batch[1].to(device, dtype=torch.float32)
        ic = inputs[0].unsqueeze(dim=0)
        n_steps = inputs.shape[0]

        pred = n_step_rollout(model, ic, n=n_steps, train_tendencies=False)
        per_pred = inputs[0].repeat(n_steps, 1, 1, 1, 1)[:,:,0,:,:]

        # Move to CPU numpy

        pred = pred.transpose(-1,-2).squeeze().detach().cpu().numpy()
        per_pred = per_pred.transpose(-1,-2).squeeze().detach().cpu().numpy()
        targets = targets.transpose(-1,-2).squeeze().detach().cpu().numpy()

        pred_u = pred[:, 0]
        pred_v = pred[:, 1]
        per_pred_u = per_pred[:, 0]
        per_pred_v = per_pred[:, 1]
        tar_u = targets[:, 0]
        tar_v = targets[:, 1]

        # Unnormalize

        pred_u = (pred_u * test_dataset.input_std[0] + test_dataset.input_mean[0])
        pred_v = (pred_v * test_dataset.input_std[1] + test_dataset.input_mean[1])
        tar_u = (tar_u * test_dataset.label_std[0] + test_dataset.label_mean[0])
        tar_v = (tar_v * test_dataset.label_std[1] + test_dataset.label_mean[1])
        
        rmse_u.append(get_rmse(tar_u, pred_u, climo=climo_u))
        rmse_u_per.append(get_rmse(tar_u, per_pred_u, climo=climo_u))
        rmse_v.append(get_rmse(tar_v, pred_v, climo=climo_v))
        rmse_v_per.append(get_rmse(tar_v, per_pred_v, climo=climo_v))

        acc_u.append(get_acc(tar_u, pred_u, climo_u))
        acc_u_per.append(get_acc(tar_u, per_pred_u, climo_u))
        acc_v.append(get_acc(tar_v, pred_v, climo_v))
        acc_v_per.append(get_acc(tar_v, per_pred_v, climo_v))

        spectra_temp, wavenumbers = get_spectra(pred_u, pred_v)
        spectra_tar_temp, _ = get_spectra(tar_u, tar_v)
        spectra_list.append(spectra_temp)
        spectra_tar_list.append(spectra_tar_temp)
        if wavenumbers_list is None:
            wavenumbers_list = wavenumbers

        print(f'done.')

    # RMSE

    results['rmse_u'] = np.stack(rmse_u, axis=0)
    results['rmse_v'] = np.stack(rmse_v, axis=0)
    results['rmse_u_per'] = np.stack(rmse_u_per, axis=0)
    results['rmse_v_per'] = np.stack(rmse_v_per, axis=0)

    # Acc

    results['acc_u'] = np.stack(acc_u, axis=0)
    results['acc_v'] = np.stack(acc_v, axis=0)
    results['acc_u_per'] = np.stack(acc_u_per, axis=0)
    results['acc_v_per'] = np.stack(acc_v_per, axis=0)

    # Spectra

    results['spectra'] = np.mean(spectra_list, axis=0)
    results['spectra_tar'] = np.mean(spectra_tar_list, axis=0)
    results['wavenumbers'] = wavenumbers_list

    return results
