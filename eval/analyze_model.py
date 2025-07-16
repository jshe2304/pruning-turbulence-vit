import os, re, sys, logging

import torch
import torch.distributed as dist
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_dir = os.path.join(parent_dir, 'src')
sys.path.append(parent_dir)
sys.path.append(src_dir)

from src.models.vit import ViT
from src.utils.data import get_time_series_dataloader

from analysis.short_analysis import perform_short_analysis
from analysis.io_utils import load_params, get_npy_files
from analysis.rollout import single_step_rollout

logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_count = torch.cuda.device_count()

ROOT_DIR = '/home/jason/data/weather_data/weather_data/run_1'
MODEL_FILENAME = 'checkpoint_100.pt'
PARAMS_FILENAME = 'params.json'
DATA_DIR = '/home/jason/data/weather_data/weather_data/run_1/data'
ANALYSIS_DIR = '/home/jason/data/weather_data/weather_data/run_1/analysis'

test_length_short = 100
num_tests = 10
test_file_start_idx = 350000
test_length_climo = 1000

def main():
    """
    Main function to run analysis on trained model.
    """

    # Read in params file
    train_params = load_params(PARAMS_FILENAME)

    # Initiate model and load weights
    model = ViT(
        in_channels=2,
        d_embed=128,
        n_heads=8, 
        n_layers=8, 
        img_shape=(256, 256), 
        patch_shape=(4, 4), 
    )
    model.load_state_dict(torch.load(MODEL_FILENAME, map_location=torch.device(device)))
    model.eval()
    model.to(device)

    # Directory to saved emulated data and analysis
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    # Initiate dataloaders
    test_frames = (test_file_start_idx, test_file_start_idx+(test_length_short*(num_tests+2)*train_params["target_step"])-1)
    dataloader, dataset = get_time_series_dataloader(
        data_dir=DATA_DIR,
        frames=test_frames,
        target_offset=3,
        batch_size=test_length_short,
    )

    # Dataloader to calculate climatology
    test_frames = list(range(test_file_start_idx, test_file_start_idx+test_length_climo))
    _, climo_dataloader = get_time_series_dataloader(
        data_dir=DATA_DIR,
        frames=test_frames,
        target_offset=1,
        batch_size=test_length_climo
    )
    climo_data, _ = next(iter(climo_dataloader))

    climo_data = climo_data.transpose(-1, -2).squeeze().detach().cpu().numpy()
    climo_u = climo_data[:,0].mean(axis=0)
    climo_v = climo_data[:,1].mean(axis=0)

    results_short = perform_short_analysis(
        model, 
        dataloader, dataset, 
        climo_u, climo_v, 
        device
    )

    # Save results

    save_fp = os.path.join(analysis_dir, 'emulate')
    os.makedirs(save_fp, exist_ok=True)
    np.savez(os.path.join(save_fp, 'short_analysis.npz'), **results_short)

    return

if __name__ == "__main__":
    main()