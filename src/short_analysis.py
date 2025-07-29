import sys
import toml

import numpy as np
import torch

from models.vision_transformer import ViT
from data.dataloaders import TurbulenceDataset

from analysis.short_analysis import perform_short_analysis

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    # Load config

    config_path = sys.argv[1]
    config = toml.load(config_path)

    # Initiate model
    
    model = ViT(**config['model'])

    # Load checkpoint

    state_dict = torch.load(config['checkpoint_file'], map_location=device, weights_only=False)

    # Check if state dict includes non-model states

    if config['checkpoint_file'].endswith('.tar'):
        state_dict = state_dict['model_state']

    # Check if state dict is distributed
    
    try:
        model.load_state_dict(state_dict)
    except:
        state_dict = {
            k.replace("module.", ""): v
            for k, v in state_dict.items()
        }
        model.load_state_dict(state_dict)

    # Send loaded model to device
    
    model.to(device)

    # Load data

    test_dataset = TurbulenceDataset(**config['test_dataset']) 
    climo_dataset = TurbulenceDataset(**config['climo_dataset'])

    # Compute metrics

    results = perform_short_analysis(
        model, 
        test_dataset, climo_dataset, 
        **config['analysis'], 
        device=device
    )

    # Store results

    np.savez(config['output_file'], **results)
