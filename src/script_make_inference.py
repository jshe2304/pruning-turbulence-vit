import sys
import os
import toml
import copy

import torch

from models.vision_transformer import ViT
from data.datasets import TimeSeriesDataset

from inference.make_inference import make_inference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(config: dict):

    # Create model
    
    model = ViT(**config['model'])
    state_dict = torch.load(config['checkpoint_file'], map_location=device, weights_only=False)
    optimizer_state = state_dict.pop('optimizer_state', None)
    model_state_dict = state_dict.pop('model_state', state_dict)
    model.load_state_dict(model_state_dict)
    model.to(device)

    # Load data

    dataset = TimeSeriesDataset(**config['dataset']) 

    # Save inference

    make_inference(
        model, dataset, 
        **config['inference'], 
        output_dir=config['output_dir'], 
        device=device
    )

if __name__ == "__main__":
    
    # Load config

    config_path = sys.argv[1]
    config = toml.load(config_path)

    # If analyzing one model

    if 'checkpoint_file' in config and 'output_dir' in config:
        print(config)
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
            this_config['output_dir'] = os.path.join(config['output_dir'], fname)

            main(this_config)