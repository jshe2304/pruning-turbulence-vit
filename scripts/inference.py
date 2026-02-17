"""
Vision Transformer inference script.

To run, pass in:
- A path to a TOML inference config file
- A path to a checkpoint file (.tar or .pt) to infer from trained/pruned model

When passing a checkpoint file, the script will:
1. Infer the model directory from the checkpoint path
2. Load the model config from <model_dir>/config.toml
3. Use the inference config from the command line TOML file

The inference TOML config should contain:
- dataset: The dataset to use for initial conditions
- inference: The inference parameters (inference_length, chunk_size)
- output_dir: Where to save the inference results
"""

import sys
import os
import toml
import copy

import torch

from src.models import create_model
from src.data.py2d_dataset import Py2DDataset
from src.data.multi_py2d_dataset import MultiPy2DDataset

from src.inference.make_inference import make_inference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(config: dict):

    # Create model

    model = create_model(**config['model'])
    state_dict = torch.load(config['checkpoint_file'], map_location=device, weights_only=False)
    optimizer_state = state_dict.pop('optimizer_state', None)
    model_state_dict = state_dict.pop('model_state', state_dict)
    model.load_state_dict(model_state_dict)
    model.to(device)

    # Load data

    Dataset = MultiPy2DDataset if 'data_dirs' in config['dataset'] else Py2DDataset
    dataset = Dataset(**config['dataset'])

    # Save inference

    make_inference(
        model, dataset,
        **config['inference'],
        output_dir=config['output_dir'],
        device=device
    )

if __name__ == "__main__":

    path = sys.argv[1]

    # Load inference config from TOML file
    if path.endswith('.toml'):
        inference_config = toml.load(path)

        # If checkpoint_file is provided in config, infer model config
        if 'checkpoint_file' in inference_config:
            checkpoint_path = inference_config['checkpoint_file']
            assert os.path.isfile(checkpoint_path), f"Checkpoint file not found: {checkpoint_path}"

            # Infer model directory from checkpoint path
            model_dir = os.path.dirname(os.path.dirname(checkpoint_path))
            model_config_path = os.path.join(model_dir, 'config.toml')

            # Load model config and merge with inference config
            if os.path.exists(model_config_path):
                model_config = toml.load(model_config_path)
                # Extract only model section, merge with inference config
                config = {'model': model_config['model']} | inference_config
            else:
                # Fallback: use model config from inference config if it exists
                config = inference_config
        else:
            config = inference_config

        # Run single model inference
        if 'checkpoint_file' in config and 'output_dir' in config:
            print(config)
            main(config)

        # Run inference on multiple models
        elif 'checkpoint_dir' in config and 'output_dir' in config:
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

    # Load from checkpoint file directly
    elif path.endswith('.tar') or path.endswith('.pt'):
        checkpoint_path = path
        assert os.path.isfile(checkpoint_path), f"Checkpoint file not found: {checkpoint_path}"

        # Infer model directory from checkpoint path
        model_dir = os.path.dirname(os.path.dirname(checkpoint_path))
        model_config_path = os.path.join(model_dir, 'config.toml')
        assert os.path.exists(model_config_path), f"Model config not found: {model_config_path}"

        # Load model config
        model_config = toml.load(model_config_path)

        # Extract inference-related configs from model config
        config = {
            'model': model_config['model'],
            'checkpoint_file': checkpoint_path,
        }

        # Use validation dataset for initial conditions if available
        if 'validation_dataset' in model_config:
            config['dataset'] = model_config['validation_dataset']
        elif 'train_dataset' in model_config:
            config['dataset'] = model_config['train_dataset']
        else:
            raise ValueError("No dataset config found in model config")

        # Set default inference parameters
        checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        config['output_dir'] = os.path.join(model_dir, 'inference', checkpoint_name)
        config['inference'] = {
            'inference_length': 10000,  # Default value
            'chunk_size': 1000,
        }

        print(f"Running inference from checkpoint: {checkpoint_path}")
        print(f"Output directory: {config['output_dir']}")
        print(f"Inference length: {config['inference']['inference_length']}")
        print(config)

        main(config)

    else:
        raise ValueError(f"Invalid path: {path}\nMust be either a .toml config file or a .tar/.pt checkpoint file")