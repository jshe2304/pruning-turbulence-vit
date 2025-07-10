import sys
import os
import logging

import torch
import torch.distributed as dist
from ruamel.yaml import YAML
import yaml
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_dir = os.path.join(parent_dir, 'src')
sys.path.append(parent_dir)
sys.path.append(src_dir)

from src.models.vision_transformer import ViT
from src.utils.data_loaders import get_dataloader

from analysis.short_analysis import perform_short_analysis
from analysis.io_utils import load_params

logging.basicConfig(level=logging.INFO)

def main(config):
    """
    Main function to run analysis on trained model.
    """
    # Extract values from config
    dataset_params = config["dataset_params"]
    short_analysis_params = config["short_analysis_params"]

    # Dataset-related parameters
    run_num = dataset_params["run_num"]
    root_dir = os.path.join(dataset_params["root_dir"], run_num)
    model_filename = dataset_params["model_filename"]
    params_filename = dataset_params["params_filename"]

    # Short analysis parameters
    rmse = short_analysis_params["rmse"]
    acc = short_analysis_params["acc"]
    spectra = short_analysis_params["spectra"]
    test_length_short = short_analysis_params["analysis_length"]
    num_tests = short_analysis_params["num_ensembles"]
    test_file_start_idx = short_analysis_params["test_file_start_idx"]
    test_length_climo = short_analysis_params["test_length_climo"]

    # Read in params file
    params_fp = os.path.join(root_dir, params_filename)
    train_params = load_params(params_fp)
    logging.info(f'params: {train_params}')

    # Initiate model and load weights
    model_fp = os.path.join(root_dir, model_filename)
    model = ViT(
        img_size=train_params["img_size"],
        patch_size=train_params["patch_size"],
        num_frames=train_params["num_frames"],
        tubelet_size=train_params["tubelet_size"],
        in_chans=train_params["in_chans"],
        encoder_embed_dim=train_params["encoder_embed_dim"],
        encoder_depth=train_params["encoder_depth"],
        encoder_num_heads=train_params["encoder_num_heads"],
        decoder_embed_dim=train_params["decoder_embed_dim"],
        decoder_depth=train_params["decoder_depth"],
        decoder_num_heads=train_params["decoder_num_heads"],
        mlp_ratio=train_params["mlp_ratio"],
        num_out_frames=train_params["num_out_frames"],
        patch_recovery=train_params["patch_recovery"]
        )

    ckpt_temp = torch.load(model_fp, map_location=torch.device('cpu'))['model_state']
    ckpt = {key[7:]: val for key, val in ckpt_temp.items()}
    model.load_state_dict(ckpt)
    model.eval()

    # Directory to saved emulated data and analysis
    save_dir = os.path.join(root_dir, 'data')  # or create a separate directory if desired
    analysis_dir = os.path.join(root_dir, 'analysis')
    plot_dir = os.path.join(root_dir, 'plots')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Initiate dataloaders
    test_file_range = (test_file_start_idx, test_file_start_idx+(test_length_short*num_tests*train_params["target_step"])-1)
    dataloader, dataset = get_dataloader(data_dir=train_params["data_dir"],
                                        file_range=test_file_range,
                                        target_step=train_params["target_step"],
                                        train_tendencies=train_params["train_tendencies"],
                                        batch_size=test_length_short,
                                        train=False,
                                        stride=train_params["target_step"],
                                        distributed=dist.is_initialized(),
                                        num_frames=train_params["num_frames"],
                                        num_out_frames=train_params["num_out_frames"],
                                        num_workers=2,
                                        pin_memory=train_params["pin_memory"])

    # Perform short analysis
    if rmse or acc or spectra:

        # Dataloader to calculate climatology
        test_file_range = (test_file_start_idx, test_file_start_idx+test_length_climo)
        dataloader_climo, dataset_climo = get_dataloader(data_dir=train_params["data_dir"],
                                        file_range=test_file_range,
                                        target_step=1,
                                        train_tendencies=train_params["train_tendencies"],
                                        batch_size=test_length_climo,
                                        train=False,
                                        stride=1,
                                        distributed=dist.is_initialized(),
                                        num_frames=1, #params["num_frames"],
                                        num_out_frames=1, #params["num_out_frames"],
                                        num_workers=2,
                                        pin_memory=train_params["pin_memory"])

        climo_data, _ = next(iter(dataloader_climo))
        print(f'climo_data.shape: {climo_data.shape}')

        climo_data = climo_data.transpose(-1, -2).squeeze().detach().cpu().numpy()                     # [B=n_steps, C, X, Y]
        climo_u = climo_data[:,0].mean(axis=0)                                       # [X, Y]
        climo_v = climo_data[:,1].mean(axis=0) 

        results_short = perform_short_analysis(model, dataloader, dataset, climo_u, climo_v, short_analysis_params, train_params, dataset_params)
        print('short analysis performed')

        if short_analysis_params["save_short_analysis"]:
            print('saving short analysis results')
            save_fp = os.path.join(analysis_dir, 'emulate')
            os.makedirs(save_fp, exist_ok=True)
            np.savez(os.path.join(save_fp, 'short_analysis.npz'), **results_short)

    else:
        print('No short analysis requested')

    return results_short

if __name__ == "__main__":
    # Load the configuration file

    config_filename = str(sys.argv[1])
    with open("config/" + config_filename, "r") as f:
        config = yaml.safe_load(f)

    # Pass the entire config dictionary to main
    results_short = main(config)
