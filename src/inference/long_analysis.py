import os
import numpy as np
from statsmodels.tsa.stattools import acf

from py2d.initialize import initialize_wavenumbers_rfft2, gridgen
from py2d.spectra import spectrum_angled_average, spectrum_zonal_average

from .utils.long_metrics import manual_eof, divergence, PDF_compute
from .utils.io_utils import get_npy_files, frame_generator

def perform_long_analysis(
    inference_dir: str,
    output_dir: str,
    img_size: int,

    # analysis flags
    analysis_length: int,
    temporal_mean: bool,
    zonal_mean: bool,
    zonal_eof_pc: bool,
    div: bool,
    return_period: bool,
    return_period_anomaly: bool,
    PDF_U: bool,
    PDF_Omega: bool,
    spectra: bool,
    eof_ncomp: int,
    PC_autocorr_nlags: int,
):
    """
    Perform long-run analysis on emulator predictions.

    Args
        inference_dir : Directory containing emulator `.npy` files to analyze.
        output_dir : Directory where analysis outputs will be saved.
        img_size : Number of grid points along each spatial dimension.
        analysis_length : Maximum number of emulator frames to process.
        temporal_mean : Compute and save temporal mean fields if True.
        zonal_mean : Compute and save zonal mean time series if True.
        zonal_eof_pc : Compute zonal EOFs and principal components if True.
        div : Compute and save divergence metric if True.
        return_period : Compute and save max/min extremes if True.
        return_period_anomaly : Compute extremes of anomalies relative to climatology if True.
        PDF_U : Compute PDF of U field across all snapshots if True.
        PDF_Omega : Compute PDF of vorticity (Omega) field if True.
        spectra : Compute and save angular and zonal spectra if True.
        eof_ncomp : Number of EOF components to compute.
        PC_autocorr_nlags : Number of lags for EOF principal-component autocorrelation.
    """

    Lx, Ly = 2*np.pi, 2*np.pi
    Nx = img_size
    Lx, Ly, X, Y, dx, dy = gridgen(Lx, Ly, Nx, Nx, INDEXING='ij')

    Kx, Ky, Kabs, Ksq, invKsq = initialize_wavenumbers_rfft2(Nx, Nx, Lx, Ly, INDEXING='ij')

    # Load emulator prediction files
    files = get_npy_files(inference_dir)
    print(f"Number of saved predicted .npy files: {len(files)}")

    # Loading climatology for return period anomaly calculation
    if return_period_anomaly:
        try:
            data = np.load(os.path.join(output_dir, 'temporal_mean.npz'))
            U_sample_mean_climatology = data['U_sample_mean']
            V_sample_mean_climatology = data['V_sample_mean']
            Omega_sample_mean_climatology = data['Omega_sample_mean']
        except FileNotFoundError:
            print(f"File not found: {os.path.join(output_dir, 'temporal_mean.npz')}")
            print("Skipping return period anomaly calculation.")
            return_period_anomaly = False

    os.makedirs(output_dir, exist_ok=True)

    U_mean_temp = np.zeros((Nx, Nx))
    V_mean_temp = np.zeros((Nx, Nx))
    Omega_mean_temp = np.zeros((Nx, Nx))

    U_zonal, V_zonal, Omega_zonal = [], [], []
    div_arr = []
    U_max, U_min, V_max, V_min, Omega_max, Omega_min = [], [], [], [], [], []
    U_max_anom_arr, U_min_anom_arr, V_max_anom_arr, V_min_anom_arr, Omega_max_anom_arr, Omega_min_anom_arr = [], [], [], [], [], []
    spectra_U_angular_avg_arr, spectra_V_angular_avg_arr, spectra_Omega_angular_avg_arr = [], [], []
    spectra_U_zonal_avg_arr, spectra_V_zonal_avg_arr, spectra_Omega_zonal_avg_arr = [], [], []

    Omega_arr = []
    U_arr = []

    frame_gen = frame_generator(files, inference_dir, Kx, Ky)

    total_files_analyzed = 0

    for U, V, Omega in frame_gen:
        if total_files_analyzed >= analysis_length:
            print('break after analyzing # files ', total_files_analyzed)
            break
        total_files_analyzed = total_files_analyzed+1

        if temporal_mean:
            U_mean_temp += U
            V_mean_temp += V
            Omega_mean_temp += Omega

        if spectra:

            ## Angular Averaged Spectra
            U_abs_hat = np.sqrt(np.fft.fft2(U)*np.conj(np.fft.fft2(U)))
            V_abs_hat = np.sqrt(np.fft.fft2(V)*np.conj(np.fft.fft2(V)))
            Omega_abs_hat = np.sqrt(np.fft.fft2(Omega)*np.conj(np.fft.fft2(Omega)))

            spectra_U_temp, wavenumber_angular_avg = spectrum_angled_average(U_abs_hat, spectral=True)
            spectra_V_temp, wavenumber_angular_avg = spectrum_angled_average(V_abs_hat, spectral=True)
            spectra_Omega_temp, wavenumber_angular_avg = spectrum_angled_average(Omega_abs_hat, spectral=True)

            spectra_U_angular_avg_arr.append(spectra_U_temp)
            spectra_V_angular_avg_arr.append(spectra_V_temp)
            spectra_Omega_angular_avg_arr.append(spectra_Omega_temp)

            ## Zonal Spectra
            spectra_U_temp, wavenumber_zonal_avg = spectrum_zonal_average(U.T)
            spectra_V_temp, wavenumber_zonal_avg = spectrum_zonal_average(V.T)
            spectra_Omega_temp, wavenumber_zonal_avg = spectrum_zonal_average(Omega.T)

            spectra_U_zonal_avg_arr.append(spectra_U_temp)
            spectra_V_zonal_avg_arr.append(spectra_V_temp)
            spectra_Omega_zonal_avg_arr.append(spectra_Omega_temp)

        if zonal_mean or zonal_eof_pc:
            U_zonal_temp = np.mean(U, axis=1)
            V_zonal_temp = np.mean(V, axis=1)
            Omega_zonal_temp = np.mean(Omega, axis=1)
            U_zonal.append(U_zonal_temp)
            V_zonal.append(V_zonal_temp)
            Omega_zonal.append(Omega_zonal_temp)

        if div:
            div_temp = divergence(U, V)
            div_arr.append(np.mean(np.abs(div_temp)))

        if return_period:

            U_max.append(np.max(U))
            U_min.append(np.min(U))
            V_max.append(np.max(V))
            V_min.append(np.min(V))
            Omega_max.append(np.max(Omega))
            Omega_min.append(np.min(Omega))

        if return_period_anomaly:

            U_anom = U - U_sample_mean_climatology
            V_anom = V - V_sample_mean_climatology
            Omega_anom = Omega - Omega_sample_mean_climatology

            U_max_anom_arr.append(np.max(U_anom))
            U_min_anom_arr.append(np.min(U_anom))
            V_max_anom_arr.append(np.max(V_anom))
            V_min_anom_arr.append(np.min(V_anom))
            Omega_max_anom_arr.append(np.max(Omega_anom))
            Omega_min_anom_arr.append(np.min(Omega_anom))


        # Calculating PDF will may need large memory
        if PDF_U:
            U_arr.append(U)

        if PDF_Omega:
            Omega_arr.append(Omega)

    if temporal_mean:

        U_mean = U_mean_temp/total_files_analyzed
        V_mean = V_mean_temp/total_files_analyzed
        Omega_mean = Omega_mean_temp/total_files_analyzed

        np.savez(os.path.join(output_dir, 'temporal_mean.npz'), U_sample_mean=U_mean, V_sample_mean=V_mean, Omega_sample_mean=Omega_mean, )

    if spectra:

        spectra_U_angular_avg = np.mean(spectra_U_angular_avg_arr, axis=0)
        spectra_V_angular_avg = np.mean(spectra_V_angular_avg_arr, axis=0)
        spectra_Omega_angular_avg = np.mean(spectra_Omega_angular_avg_arr, axis=0)

        spectra_U_zonal_avg = np.mean(spectra_U_zonal_avg_arr, axis=0)
        spectra_V_zonal_avg = np.mean(spectra_V_zonal_avg_arr, axis=0)
        spectra_Omega_zonal_avg = np.mean(spectra_Omega_zonal_avg_arr, axis=0)

        np.savez(os.path.join(output_dir, 'spectra.npz'),
            spectra_U_angular_avg=spectra_U_angular_avg, spectra_V_angular_avg=spectra_V_angular_avg, spectra_Omega_angular_avg=spectra_Omega_angular_avg, wavenumber_angular_avg=wavenumber_angular_avg,
            spectra_U_zonal_avg=spectra_U_zonal_avg, spectra_V_zonal_avg=spectra_V_zonal_avg, spectra_Omega_zonal_avg=spectra_Omega_zonal_avg, wavenumber_zonal_avg=wavenumber_zonal_avg,
        )

    if zonal_eof_pc or zonal_mean:

        U_zonal_mean = np.mean(U_zonal, axis=0)
        Omega_zonal_mean = np.mean(Omega_zonal, axis=0)
        V_zonal_mean = np.mean(V_zonal, axis=0)

        np.savez(os.path.join(output_dir, 'zonal_mean.npz'), U_zonal_mean=U_zonal_mean, Omega_zonal_mean=Omega_zonal_mean, V_zonal_mean=V_zonal_mean, )

        if zonal_eof_pc:

            U_zonal_anom = np.array(U_zonal) - U_zonal_mean
            EOF_U, PC_U, exp_var_U = manual_eof(U_zonal_anom, eof_ncomp)

            PC_acf_U = []
            n_lags = PC_autocorr_nlags

            for i in range(eof_ncomp):
                acf_i, confint_i = acf(PC_U[:, i], nlags=n_lags, alpha=0.5)
                PC_acf_U.append({"acf": acf_i, "confint": confint_i})

            Omega_zonal_anom = np.array(Omega_zonal) - Omega_zonal_mean
            EOF_Omega, PC_Omega, exp_var_Omega = manual_eof(Omega_zonal_anom, eof_ncomp)

            PC_acf_Omega = []
            for i in range(eof_ncomp):
                acf_i, confint_i = acf(PC_Omega[:, i], nlags=n_lags, alpha=0.5)
                PC_acf_Omega.append({"acf": acf_i, "confint": confint_i})

            # ## Scikit-learn

            # pca = PCA(n_components=long_analysis_params["eof_ncomp"])
            # PC_U_sklearn = pca.fit_transform(U_zonal_anom)
            # EOF_U_sklearn = pca.components_.T
            # expvar_U_sklearn = pca.explained_variance_ratio_

            # pca = PCA(n_components=long_analysis_params["eof_ncomp"])
            # PC_Omega_sklearn = pca.fit_transform(Omega_zonal_anom)
            # EOF_Omega_sklearn = pca.components_.T
            # expvar_Omega_sklearn = pca.explained_variance_ratio_

            # ## SVD
            # EOF_U_svd, PC_U_svd, expvar_U_svd = manual_svd_eof(U_zonal_anom)
            # EOF_Omega_svd, PC_Omega_svd, expvar_Omega_svd = manual_svd_eof(Omega_zonal_anom)

            # np.savez(os.path.join(output_dir, 'zonal_eof_pc.npz'), EOF_U=EOF_U, PC_U=PC_U, exp_var_U=exp_var_U, EOF_Omega=EOF_Omega, PC_Omega=PC_Omega, exp_var_Omega=exp_var_Omega, EOF_U_sklearn=EOF_U_sklearn, PC_U_sklearn=PC_U_sklearn, expvar_U_sklearn=expvar_U_sklearn, EOF_Omega_sklearn=EOF_Omega_sklearn, PC_Omega_sklearn=PC_Omega_sklearn, expvar_Omega_sklearn=expvar_Omega_sklearn, EOF_U_svd=EOF_U_svd, PC_U_svd=PC_U_svd, expvar_U_svd=expvar_U_svd, EOF_Omega_svd=EOF_Omega_svd, PC_Omega_svd=PC_Omega_svd, expvar_Omega_svd=expvar_Omega_svd)
            np.savez(os.path.join(output_dir, 'zonal_eof_pc.npz'), U_eofs=EOF_U, U_pc=PC_U, U_expvar=exp_var_U, U_pc_acf=PC_acf_U, Omega_eofs=EOF_Omega, Omega_PC=PC_Omega, Omega_expvar=exp_var_Omega, Omega_pc_acf=PC_acf_Omega)

    if div:
        div_arr = np.array(div_arr, dtype=np.float32) # Torch Emulator data is float32
        np.save(os.path.join(output_dir, 'div'), div_arr)

    if return_period:
        np.savez(os.path.join(output_dir, 'extremes.npz'), U_max_arr=np.asarray(U_max), U_min_arr=np.asarray(U_min), V_max_arr=np.asarray(V_max), V_min_arr=np.asarray(V_min), Omega_max_arr=np.asarray(Omega_max), Omega_min_arr=np.asarray(Omega_min))

    if return_period_anomaly:
        np.savez(os.path.join(output_dir, 'extremes_anom.npz'), U_max_arr=np.asarray(U_max_anom_arr), U_min_arr=np.asarray(U_min_anom_arr), V_max_arr=np.asarray(V_max_anom_arr), V_min_arr=np.asarray(V_min_anom_arr), Omega_max_arr=np.asarray(Omega_max_anom_arr), Omega_min_arr=np.asarray(Omega_min_anom_arr), )

    if PDF_U:
        U_arr = np.array(U_arr)
        U_mean, U_std, U_pdf, U_bins, bw_scott = PDF_compute(U_arr)
        np.savez(os.path.join(output_dir, 'PDF_U.npz'), bw_scott=bw_scott, U_mean=U_mean, U_std=U_std, U_pdf=U_pdf, U_bins=U_bins)

    if PDF_Omega:
        Omega_arr = np.array(Omega_arr)
        Omega_mean, Omega_std, Omega_pdf, Omega_bins, bw_scott = PDF_compute(Omega_arr)
        np.savez(os.path.join(output_dir, 'PDF_Omega.npz'), Omega_mean=Omega_mean, Omega_std=Omega_std, Omega_pdf=Omega_pdf, Omega_bins=Omega_bins, bw_scott=bw_scott)

