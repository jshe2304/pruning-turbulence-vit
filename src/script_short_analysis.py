import sys
import os
import toml
import copy

import numpy as np
import torch

from models.vision_transformer import ViT
from data.datasets import TimeSeriesDataset

from inference.short_analysis import perform_short_analysis

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(config: dict):

    # Create model
    
    model = ViT(**config['model'])
    state_dict = torch.load(config['checkpoint_file'], map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)

    # Load data

    test_dataset = TimeSeriesDataset(**config['test_dataset']) 
    climo_dataset = TimeSeriesDataset(**config['climo_dataset'])

    # Compute metrics

    results = perform_short_analysis(
        model, 
        test_dataset, climo_dataset, 
        **config['analysis'], 
        device=device
    )

    # Store results

    np.savez(config['output_file'], **results)

if __name__ == "__main__":
    
    # Load config

    config_path = sys.argv[1]
    config = toml.load(config_path)

    # If analyzing one model

    if 'checkpoint_file' in config and 'output_file' in config:
        main(config)

    # If analyzing many models

    if 'checkpoint_dir' in config and 'output_dir' in config:

        # Make sure output directory exists

        os.makedirs(config['output_dir'], exist_ok=True)

        # Loop through state dict files

        for file in os.listdir(config['checkpoint_dir']):
            fname, ext = os.path.splitext(file)
            if not (ext == '.pt' or ext == '.tar'):
                continue

            # Create single model config file for processing

            this_config = copy.deepcopy(config)
            this_config['checkpoint_file'] = os.path.join(config['checkpoint_dir'], fname + ext)
            this_config['output_file'] = os.path.join(config['output_dir'], fname + '_metrics.npz')

            main(this_config)