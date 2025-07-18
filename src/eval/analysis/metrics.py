"""
Metrics functions for turbulence model evaluation.

This module contains the core metrics functions used in short analysis:
- RMSE (Root Mean Square Error)
- ACC (Spatial Correlation/Accuracy)  
- Spectral analysis
"""

import numpy as np
from typing import Tuple, Optional

def get_rmse(y: np.ndarray, y_hat: np.ndarray, climo: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate Root Mean Square Error with optional climatology baseline.
    
    Args:
        y: Ground truth data [n_steps, H, W]
        y_hat: Predicted data [n_steps, H, W]
        climo: Climatology baseline [H, W]. If None, uses zeros.
        
    Returns:
        RMSE values [n_steps]
    """
    if climo is None:
        climo = np.zeros((y.shape[-2], y.shape[-1]))

    y_anom = y - climo
    y_hat_anom = y_hat - climo
    err = (y_anom - y_hat_anom) ** 2
    err = np.mean(err, axis=(-1, -2))
    rmse = np.sqrt(err)
    
    return rmse


def get_acc(y: np.ndarray, y_hat: np.ndarray, climo: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate spatial correlation (accuracy) with optional climatology baseline.
    
    Args:
        y: Ground truth data [n_steps, H, W]
        y_hat: Predicted data [n_steps, H, W]
        climo: Climatology baseline [H, W]. If None, uses zeros.
        
    Returns:
        Correlation coefficients [n_steps]
    """
    if climo is None:
        climo = np.zeros((y.shape[-2], y.shape[-1]))

    corr = []
    for i in range(y.shape[0]):
        y_i = y[i] - climo
        y_hat_i = y_hat[i] - climo
        corr.append(np.corrcoef(y_i.flatten(), y_hat_i.flatten())[1, 0])

    return np.array(corr)


def spectrum_zonal_average_2d(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate zonally averaged spectrum for 2D flow variables.

    Args:
        u: 2D velocity component [H, W]
        v: 2D velocity component [H, W]

    Returns:
        Tuple of (energy_spectrum, wavenumbers)
    """
    if u.ndim != 2 or v.ndim != 2:
        raise ValueError("Input flow variables must be 2D")
    if u.shape[0] != u.shape[1] or v.shape[0] != v.shape[1]:
        raise ValueError("Flow variables must be square matrices")

    n = u.shape[0]

    # FFT of velocities along the first dimension
    u_hat = np.fft.rfft(u, axis=1) / n
    v_hat = np.fft.rfft(v, axis=1) / n

    # Account for negative wavenumbers
    u_hat[1:] = 2 * u_hat[1:]
    v_hat[1:] = 2 * v_hat[1:]

    # Energy spectrum
    energy_spectrum = np.mean(np.abs(u_hat) ** 2, axis=0)
    wavenumbers = np.linspace(0, n // 2, n // 2 + 1)

    return energy_spectrum, wavenumbers


def get_spectra(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate energy spectra for velocity field time series.
    
    Args:
        u: U velocity component [n_steps, H, W]
        v: V velocity component [n_steps, H, W]
        
    Returns:
        Tuple of (spectra [n_steps, n_wavenumbers], wavenumbers)
    """
    spectra = []
    wavenumbers = None
    
    for i in range(u.shape[0]):
        energy_spectrum, wn = spectrum_zonal_average_2d(u[i], v[i])
        spectra.append(energy_spectrum)
        if wavenumbers is None:
            wavenumbers = wn

    # Ensure wavenumbers is not None
    if wavenumbers is None:
        raise ValueError("No wavenumbers calculated")

    return np.stack(spectra, axis=0), wavenumbers