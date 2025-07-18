import os
import sys
import toml
import torch
import numpy as np

from models.vit import ViT
from data.datasets import TimeSeriesDataset
from torch.utils.data import DataLoader

from analysis.short_analysis import perform_short_analysis

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_from_checkpoint(model_config, checkpoint_path):
    """
    Load model from checkpoint following repository patterns.
    """
    model = ViT(**model_config)
    
    # Load checkpoint with proper key handling (as done in script_prune_iterative.py)
    state_dict = torch.load(checkpoint_path, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except:
        # Handle DDP model keys by removing "module." prefix
        state_dict = {
            k.replace("module.", ""): v
            for k, v in state_dict.items()
        }
        model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    return model

def create_dataset_and_loader(dataset_config, batch_size=32):
    """
    Create dataset and dataloader following repository patterns.
    """
    dataset = TimeSeriesDataset(**dataset_config)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    return dataset, dataloader

def run_short_analysis(config):
    """
    Run short analysis on trained model following repository API patterns.
    Always performs RMSE, ACC, and SPECTRA analyses.
    """
    # Extract main configuration sections
    model_config = config["model"]
    eval_config = config["evaluation"]
    train_dataset_config = config["train_dataset"]
    test_dataset_config = config["test_dataset"]
    
    # Load model
    checkpoint_path = model_config["checkpoint_path"]
    model = load_model_from_checkpoint(model_config, checkpoint_path)
    
    # Create test dataset and dataloader
    test_dataset, test_dataloader = create_dataset_and_loader(
        test_dataset_config, 
        batch_size=eval_config.get("batch_size", 32)
    )
    
    # Create climatology dataset if needed
    climo_config = eval_config.get("climatology", {})
    if climo_config:
        climo_dataset, climo_dataloader = create_dataset_and_loader(
            climo_config,
            batch_size=climo_config.get("batch_size", 1000)
        )
        
        # Calculate climatology baseline
        climo_batch = next(iter(climo_dataloader))
        climo_input, climo_target = climo_batch
        
        # Assuming 2-channel data (u, v velocity components)
        climo_data = climo_target.detach().cpu().numpy()  # [B, C, H, W]
        climo_u = climo_data[:, 0].mean(axis=0)  # [H, W]
        climo_v = climo_data[:, 1].mean(axis=0)  # [H, W]
    else:
        climo_u = climo_v = None
    
    # Prepare analysis parameters (always include RMSE, ACC, SPECTRA)
    short_analysis_params = {
        "rmse": True,
        "acc": True, 
        "spectra": True,
        **eval_config  # Include other evaluation parameters
    }
    
    # Perform short analysis
    results = perform_short_analysis(
        model=model,
        dataloader=test_dataloader,
        dataset=test_dataset,
        climo_u=climo_u,
        climo_v=climo_v,
        short_analysis_params=short_analysis_params,
        train_params=train_dataset_config,  # For compatibility
        dataset_params=test_dataset_config,  # For compatibility
        device=device
    )
    
    return results

def run_analysis_from_config(config_path):
    """
    Load TOML config and run analysis.
    """
    config = toml.load(config_path)
    return run_short_analysis(config)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_model.py <config_file.toml>")
        sys.exit(1)
        
    config_path = sys.argv[1]
    results = run_analysis_from_config(config_path)
    
    if results:
        print("Analysis completed successfully")
        print(f"Results contain: RMSE, ACC, SPECTRA")
        print(f"All results keys: {list(results.keys())}")
    else:
        print("Analysis failed")
