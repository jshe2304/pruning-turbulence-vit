import numpy as np

def get_rmse(y, y_hat, climo=None):
    
    if climo is None:
        climo = np.zeros((y.shape[-2], y.shape[-1]))

    y_anom = y - climo
    y_hat_anom = y_hat - climo
    err = (y_anom - y_hat_anom) ** 2
    err = np.mean(err, axis=(-1, -2))
    rmse = np.sqrt(err)
    
    return rmse

def get_acc(y, y_hat, climo=None):
    """
    Args:
        y, y_hat: [B=n_steps, X, Y]
    """

    if climo is None:
        climo = np.zeros((y.shape[-2], y.shape[-1]))

    corr = []
    for i in range(y.shape[0]):
        y_i = y[i] - climo
        y_hat_i = y_hat[i] - climo
        #acc = (
        #        np.sum(y_i * y_hat_i) /
        #        np.sqrt(
        #            np.sum(y_i ** 2) * np.sum(y_hat_i ** 2)
        #            )
        #        )
        #corr.append(acc)
        corr.append(np.corrcoef(y_i.flatten(), y_hat_i.flatten())[1, 0])

    return np.array(corr)

def _spectrum_zonal_average_2D_FHIT(U,V):
    """
    Zonal averaged spectrum for 2D flow variables

    Args:
    U: 2D square matrix, velocity
    V: 2D square matrix, velocity

    Returns:
    E_hat: 1D array
    wavenumber: 1D array
    """

    # Check input shape
    if U.ndim != 2 and V.ndim != 2:
        raise ValueError("Input flow variable is not 2D. Please input 2D matrix.")
    if U.shape[0] != U.shape[1] and V.shape[0] != V.shape[1]:
        raise ValueError("Dimension mismatch for flow variable. Flow variable should be a square matrix.")

    N_LES = U.shape[0]

    # fft of velocities along the first dimension
    U_hat = np.fft.rfft(U, axis=1)/ N_LES  #axis=1
    V_hat = np.fft.rfft(V, axis=1)/ N_LES  #axis=1

    U_hat[1:] = 2*U_hat[1:] # Multiply by 2 to account for the negative wavenumbers
    V_hat[1:] = 2*V_hat[1:] 

    # Energy
    E_hat = U_hat

    # Average over the second dimension
    # Multiplying by 2 to account for the negative wavenumbers
    E_hat = np.mean(np.abs(E_hat) ** 2, axis=0) #axis=0
    wavenumbers = np.linspace(0, N_LES//2, N_LES//2+1)

    return E_hat, wavenumbers

def get_spectra(U, V):
    """`
    Args:
        U, V: [B=n_steps, X, Y]
    Returns:
        spectra: [B=n_steps, k]
    """
    
    spectra = []
    for i in range(U.shape[0]):
        E_hat, wavenumbers = _spectrum_zonal_average_2D_FHIT(U[i], V[i])
        spectra.append(E_hat)

    spectra = np.stack(spectra, axis=0)

    return spectra, wavenumbers
