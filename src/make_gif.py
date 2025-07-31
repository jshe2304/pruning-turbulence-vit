import sys
import os
import toml
import copy

import torch

from models.vision_transformer import ViT
from data.datasets import TimeSeriesDataset

from analysis.make_gif import make_gif

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(config: dict):

    # Initiate and load model
    
    model = ViT(**config['model'])
    state_dict = torch.load(config['checkpoint_file'], map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)

    # Load test data

    test_dataset = TimeSeriesDataset(**config['test_dataset']) 
    
    # Make gif

    make_gif(model, test_dataset, config['output_file'], device=device)

if __name__ == "__main__":
    
    # Load config

    config_path = sys.argv[1]
    config = toml.load(config_path)

    main(config)
