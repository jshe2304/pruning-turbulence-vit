import sys
import toml

import torch

from src.inference.long_analysis import perform_long_analysis

if __name__ == "__main__":
    
    # Load config

    config_path = sys.argv[1]
    config = toml.load(config_path)

    results = perform_long_analysis(
        **config['analysis'], 
    )
