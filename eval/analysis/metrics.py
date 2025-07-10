import numpy as np
from scipy.stats import gaussian_kde

from py2d.initialize import initialize_wavenumbers_rfft2, gridgen
from py2d.derivative import derivative
from py2d.convert import Omega2Psi, Psi2UV

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

def spectrum_zonal_average_2D_FHIT(U,V):
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
  #E_hat = 0.5 * U_hat * np.conj(U_hat) + 0.5 * V_hat * np.conj(V_hat)
  E_hat = U_hat

  # Average over the second dimension
  # Multiplying by 2 to account for the negative wavenumbers
  E_hat = np.mean(np.abs(E_hat), axis=0) #axis=0
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
        E_hat, wavenumbers = spectrum_zonal_average_2D_FHIT(U[i], V[i])
        spectra.append(E_hat)

    spectra = np.stack(spectra, axis=0)

    return spectra, wavenumbers


# def get_zonal_PCA(zdata, n_comp=1):
#     """
#     Compute PCA of zonally-averaged fields.
#     Args:
#         data: [B=n_steps, X, Y] np.array of data
#     Returns:
#         pcs: [B, n_comp]
#         eofs: [n_comp, X]
#     """

#     # Zonally average data
#     print(f'zdata.shape: {zdata.shape}')

#     # initiate PCA
#     pca = PCA(n_components=n_comp)

#     pcs = pca.fit_transform(zdata)      # [B, n_comp]
#     eofs = pca.components_              # [n_comp, X]
#     print(f'pcs.shape: {pcs.shape}')
#     print(f'eofs.shape: {eofs.shape}')

#     return pcs, eofs

def manual_eof(X_demeaned, n_comp=1):
    # Step 1: Demean the data
    # X_demeaned = X - np.mean(X, axis=0)
    # Step 2: Covariance matrix
    C = np.cov(X_demeaned, rowvar=False)
    # Step 3: Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    # Step 4: Sort by descending eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    # First three EOFs and PCs
    EOFs_manual = eigenvectors[:, :n_comp]
    PCs_manual = X_demeaned @ EOFs_manual

    # Calculate the explained variance
    total_variance = np.sum(eigenvalues)
    explained_variance = eigenvalues / total_variance

    return EOFs_manual, PCs_manual, np.flip(explained_variance)[:3]

# Method 2: Manual calculation of EOFs and PCs using SVD
def manual_svd_eof(X_demeaned):

    # Step 2: Apply SVD on the demeaned data
    # U: left singular vectors (PCs)
    # s: singular values
    # Vt: right singular vectors (EOFs, transposed)
    U, s, Vt = np.linalg.svd(X_demeaned, full_matrices=False)
    
    # Step 3: Extract the first three EOFs and PCs
    EOFs_svd = Vt.T    # EOFs and transpose to shape (N, 3)
    PCs_svd = U * s # Scale U by the singular values to get PCs
    
    # Step 4: Calculate explained variance
    total_variance = np.sum(s ** 2)
    explained_variance = (s ** 2) / total_variance
    
    return EOFs_svd, PCs_svd, explained_variance

def get_div(U, V):
    """
    Args:
        U: [B=n_steps, X, Y] 
        V: [B=n_steps, X, Y]
    Returns:
        div: [B,] divergence vs time
    """
   
    Lx, Ly = 2*np.pi, 2*np.pi
    Nx, Ny = U.shape[1], U.shape[2]
    Lx, Ly, X, Y, dx, dy = gridgen(Lx, Ly, Nx, Ny, INDEXING='ij')

    Kx, Ky, Kabs, Ksq, invKsq = initialize_wavenumbers_rfft2(Nx, Ny, Lx, Ly, INDEXING='ij')

    div = []
    for i in range(U.shape[0]):
        Dx = derivative(U[i,:,:], [0,1], Kx, Ky, spectral=False) #[1,0]
        Dy = derivative(V[i,:,:], [1,0], Kx, Ky, spectral=False) #[0,1]
        div.append(np.mean(np.abs(Dx + Dy)))

    return np.array(div)

def divergence(U, V):
    """
    Args:
        U: [X, Y] 
        V: [X, Y]
    Returns:
        div: [X,Y] divergence vs time
    """
   
    Lx, Ly = 2*np.pi, 2*np.pi
    Nx, Ny = U.shape[0], U.shape[1]
    Lx, Ly, X, Y, dx, dy = gridgen(Lx, Ly, Nx, Ny, INDEXING='ij')

    Kx, Ky, Kabs, Ksq, invKsq = initialize_wavenumbers_rfft2(Nx, Ny, Lx, Ly, INDEXING='ij')

    Ux = derivative(U.T, [1,0], Kx, Ky, spectral=False) #[1,0]
    Vy = derivative(V.T, [0,1], Kx, Ky, spectral=False) #[0,1]
    div = Ux + Vy

    return div

def PDF_compute(data, bw_factor=1):
    data_arr = np.array(data).flatten()
    del data

    # Calculate mean and standard deviation
    data_mean, data_std = np.mean(data_arr), np.std(data_arr)

    # Define bins within 10 standard deviations from the mean, but also limit them within the range of the data
    bin_max = np.min(np.abs([np.min(data_arr), np.max(data_arr)]))
    bin_min = -bin_max
    bins = np.linspace(bin_min, bin_max, 100)

    print('PDF Clculation')
    print('bin min', bin_min)
    print('bin max', bin_max)
    print('data Shape', data_arr.shape)
    print('data mean', data_mean)
    print('data_std', data_std)
    print('Total nans', np.sum(np.isnan(data_arr)))

    # Compute PDF using Scipy
    bw1 = bw_factor*(data_arr.shape[0])**(-1/5) # custom bw method scott method n**(-1/5)
    kde = gaussian_kde(data_arr, bw_method=bw1)

    # # Define a range over which to evaluate the density
    data_bins = bins
    bw_scott = kde.factor
    # # Evaluate the density over the range
    data_pdf = kde.evaluate(data_bins)

    return data_mean, data_std, data_pdf, data_bins, bw_scott

def return_period(data, dt=1, bins=50, bin_range=None):
    '''
    Return period for a time series data
    Inverse of the exceedance probability
    '''
    # Compute histogram frequencies and bin edges
    freq, bin_edges = np.histogram(data, bins=bins, range=bin_range)
    # Compute bin centers
    bins_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    # Compute cumulative frequencies from the highest bin downwards
    freq_exceedance = np.cumsum(freq[::-1])[::-1]
    # Total number of data points
    total_data_points = len(data)
    # Calculate exceedance probabilities
    prob_exceedance = freq_exceedance / total_data_points
    # Avoid division by zero in return period calculation
    prob_exceedance = np.clip(prob_exceedance, 1e-14, 1)
    # Calculate return periods
    return_periods = dt / prob_exceedance
    return return_periods, prob_exceedance, bins_centers
