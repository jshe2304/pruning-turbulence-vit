import os

import numpy as np
from scipy.io import loadmat

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize

from py2d.initialize import initialize_wavenumbers_rfft2
from py2d.convert import Omega2Psi, Psi2UV

class TimeSeriesDataset(Dataset):
    def __init__(self, data_dir, frame_ranges, target_offset): 
        """
        Args:
            data_dir: Path to the data directory
            frame_ranges: List of tuples indicating frame ranges
            target_offset: Offset for target frame indices
        """

        self.data_dir = data_dir
        self.target_offset = target_offset

        # Expand frames into a list of indices

        self.frames = []
        for frame_range in frame_ranges: self.frames.extend(range(*frame_range))

        # Load mean/std for normalization

        stats_dir = os.path.join(data_dir, 'stats')
        mean = np.load(os.path.join(stats_dir, 'mean_full_field.npy')).tolist()
        std  = np.load(os.path.join(stats_dir, 'std_full_field.npy')).tolist()
        self.normalize = Normalize(mean, std)

    def __len__(self):
        return len(self.frames) - self.target_offset

    def __getitem__(self, i: int):

        input_frame = self._load_and_norm(self.frames[i + self.target_offset])
        target_frame = self._load_and_norm(self.frames[i])

        return input_frame, target_frame

    def _load_and_norm(self, file_num: int) -> torch.Tensor:
        """
        Load and normalize a frame from the data directory

        Args:
            file_num: Frame index

        Returns:
            Normalized frame as a tensor
        """

        file_path = os.path.join(self.data_dir, 'data', f"{file_num}.mat")

        omega = loadmat(file_path)['Omega']
        uv = self._omega_to_uv(omega)
        uv = self.normalize(uv)
        return uv

    def _omega_to_uv(self, omega: np.ndarray) -> torch.Tensor:
        """
        Convert omega to uv

        Args:
            omega: Omega vorticity field

        Returns:
            U,V velocity fields as a tensor
        """
        nx, ny = omega.shape
        Kx, Ky, _, _, invKsq = initialize_wavenumbers_rfft2(nx, ny, 2*np.pi, 2*np.pi, INDEXING='ij')
        psi = Omega2Psi(omega, invKsq)
        u, v = Psi2UV(psi, Kx, Ky)
        return torch.tensor(np.stack([u, v]), dtype=torch.float32)
