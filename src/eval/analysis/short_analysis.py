import numpy as np
import torch
from typing import Dict, List, Tuple, Optional

from analysis.metrics import get_rmse, get_acc, get_spectra
from analysis.rollout import n_step_rollout

def calculate_statistics(data_list: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Calculate comprehensive statistics for a list of metric arrays.
    
    Args:
        data_list: List of numpy arrays with shape [n_steps, ...]
        
    Returns:
        Dictionary containing various statistics
    """
    stacked_data = np.stack(data_list, axis=0)  # [n_ensembles, n_steps, ...]
    
    return {
        'raw': stacked_data,
        'mean': np.mean(stacked_data, axis=0),
        'std': np.std(stacked_data, axis=0),
        'median': np.quantile(stacked_data, 0.5, axis=0),
        'q25': np.quantile(stacked_data, 0.25, axis=0),
        'q75': np.quantile(stacked_data, 0.75, axis=0),
    }

def unnormalize_predictions(pred: np.ndarray, target: np.ndarray, dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unnormalize model predictions and targets using dataset statistics.
    
    Args:
        pred: Model predictions [n_steps, n_channels, H, W]
        target: Target values [n_steps, n_channels, H, W]
        dataset: Dataset object with normalization statistics
        
    Returns:
        Tuple of (unnormalized_pred, unnormalized_target)
    """
    pred_unnorm = pred.copy()
    target_unnorm = target.copy()
    
    # Unnormalize predictions
    for ch in range(pred.shape[1]):
        pred_unnorm[:, ch] = pred[:, ch] * dataset.input_std[ch] + dataset.input_mean[ch]
        target_unnorm[:, ch] = target[:, ch] * dataset.label_std[ch] + dataset.label_mean[ch]
    
    return pred_unnorm, target_unnorm

def process_single_batch(
    model: torch.nn.Module, 
    batch: Tuple[torch.Tensor, torch.Tensor], 
    device: torch.device,
    train_tendencies: bool,
    dataset,
    climo_u: Optional[np.ndarray] = None,
    climo_v: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Process a single batch through the model and calculate all metrics.
    
    Args:
        model: Trained PyTorch model
        batch: Tuple of (inputs, targets)
        device: Torch device
        train_tendencies: Whether model was trained on tendencies
        dataset: Dataset object with normalization statistics
        climo_u: Climatology for u component
        climo_v: Climatology for v component
        
    Returns:
        Dictionary containing RMSE, ACC, and spectra results for this batch
    """
    inputs, targets = batch[0].to(device, dtype=torch.float32), batch[1].to(device, dtype=torch.float32)
    ic = inputs[0].unsqueeze(dim=0)
    n_steps = inputs.shape[0]

    # Generate predictions
    with torch.no_grad():
        pred = n_step_rollout(model, ic, n=n_steps, train_tendencies=train_tendencies)
        
        # Create persistence baseline (just repeat first frame)
        persistence = inputs[0].repeat(n_steps, 1, 1, 1, 1)[:, :, 0, :, :]

    # Convert to numpy and move to CPU
    pred_np = pred.transpose(-1, -2).squeeze().detach().cpu().numpy()
    persistence_np = persistence.transpose(-1, -2).squeeze().detach().cpu().numpy()
    targets_np = targets.transpose(-1, -2).squeeze().detach().cpu().numpy()

    # Unnormalize data
    pred_unnorm, targets_unnorm = unnormalize_predictions(pred_np, targets_np, dataset)
    persistence_unnorm, _ = unnormalize_predictions(persistence_np, targets_np, dataset)

    # Extract u and v components
    pred_u, pred_v = pred_unnorm[:, 0], pred_unnorm[:, 1]
    persistence_u, persistence_v = persistence_unnorm[:, 0], persistence_unnorm[:, 1]
    target_u, target_v = targets_unnorm[:, 0], targets_unnorm[:, 1]

    # Calculate metrics
    results = {}
    
    # RMSE metrics
    results['rmse_u'] = get_rmse(target_u, pred_u, climo=climo_u)
    results['rmse_u_persistence'] = get_rmse(target_u, persistence_u, climo=climo_u)
    results['rmse_v'] = get_rmse(target_v, pred_v, climo=climo_v)
    results['rmse_v_persistence'] = get_rmse(target_v, persistence_v, climo=climo_v)
    
    # Accuracy metrics
    results['acc_u'] = get_acc(target_u, pred_u, climo_u)
    results['acc_u_persistence'] = get_acc(target_u, persistence_u, climo_u)
    results['acc_v'] = get_acc(target_v, pred_v, climo_v)
    results['acc_v_persistence'] = get_acc(target_v, persistence_v, climo_v)
    
    # Spectral analysis
    spectra_pred, wavenumbers = get_spectra(pred_u, pred_v)
    spectra_target, _ = get_spectra(target_u, target_v)
    results['spectra_pred'] = spectra_pred
    results['spectra_target'] = spectra_target
    results['wavenumbers'] = wavenumbers
    
    return results


def aggregate_batch_results(batch_results: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """
    Aggregate results from multiple batches into final statistics.
    
    Args:
        batch_results: List of dictionaries containing results from each batch
        
    Returns:
        Dictionary containing aggregated statistics
    """
    # Collect all metrics by type
    metrics = {
        'rmse_u': [r['rmse_u'] for r in batch_results],
        'rmse_u_persistence': [r['rmse_u_persistence'] for r in batch_results],
        'rmse_v': [r['rmse_v'] for r in batch_results],
        'rmse_v_persistence': [r['rmse_v_persistence'] for r in batch_results],
        'acc_u': [r['acc_u'] for r in batch_results],
        'acc_u_persistence': [r['acc_u_persistence'] for r in batch_results],
        'acc_v': [r['acc_v'] for r in batch_results],
        'acc_v_persistence': [r['acc_v_persistence'] for r in batch_results],
        'spectra_pred': [r['spectra_pred'] for r in batch_results],
        'spectra_target': [r['spectra_target'] for r in batch_results],
    }
    
    # Calculate statistics for each metric
    final_results = {}
    
    # Process RMSE and ACC metrics with full statistics
    for metric_name in ['rmse_u', 'rmse_u_persistence', 'rmse_v', 'rmse_v_persistence',
                       'acc_u', 'acc_u_persistence', 'acc_v', 'acc_v_persistence']:
        stats = calculate_statistics(metrics[metric_name])
        for stat_name, stat_value in stats.items():
            final_results[f'{metric_name}_{stat_name}'] = stat_value
    
    # Process spectral metrics (just take mean across ensembles)
    final_results['spectra_pred'] = np.mean(metrics['spectra_pred'], axis=0)
    final_results['spectra_target'] = np.mean(metrics['spectra_target'], axis=0)
    final_results['wavenumbers'] = batch_results[0]['wavenumbers']  # Same for all batches
    
    return final_results


def perform_short_analysis(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    dataset,
    climo_u: Optional[np.ndarray],
    climo_v: Optional[np.ndarray],
    short_analysis_params: Dict,
    train_params: Dict,
    dataset_params: Dict,
    device: torch.device
) -> Dict[str, np.ndarray]:
    """
    Perform comprehensive short-term analysis including RMSE, accuracy, and spectral analysis.
    
    Args:
        model: Trained PyTorch model
        dataloader: DataLoader for test data
        dataset: Dataset object with normalization statistics
        climo_u: Climatology for u component [H, W]
        climo_v: Climatology for v component [H, W]
        short_analysis_params: Analysis configuration parameters
        train_params: Training parameters (for compatibility)
        dataset_params: Dataset parameters (for compatibility)
        device: PyTorch device
        
    Returns:
        Dictionary containing analysis results with statistics
    """

    batch_results = []
    num_ensembles = short_analysis_params.get("num_ensembles", 10)
    train_tendencies = train_params.get("train_tendencies", False)
    
    # Process each batch
    for i, batch in enumerate(dataloader):
        if i >= num_ensembles:
            break
            
        batch_result = process_single_batch(
            model=model,
            batch=batch,
            device=device,
            train_tendencies=train_tendencies,
            dataset=dataset,
            climo_u=climo_u,
            climo_v=climo_v
        )
        batch_results.append(batch_result)
    
    # Aggregate results across all batches
    final_results = aggregate_batch_results(batch_results)

    return final_results
