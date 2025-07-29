import os

import numpy as np
from scipy.io import loadmat

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from py2d.initialize import initialize_wavenumbers_rfft2
from py2d.convert import Omega2Psi, Psi2UV

class TimeSeriesDataset(Dataset):
    def __init__(self, data_dir, frame_ranges, target_offset, **kwargs): 
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

        input_frame = self._load_and_norm(self.frames[i])
        target_frame = self._load_and_norm(self.frames[i + self.target_offset])

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

class MultiFrameTimeSeriesDataset(Dataset):
    def __init__(self, data_dir, frame_ranges, target_offset, num_frames=1, **kwargs): 
        """
        Args:
            data_dir: Path to the data directory
            frame_ranges: List of tuples indicating frame ranges
            target_offset: Offset for target frame indices
            num_frames: Number of consecutive frames to load for input
        """

        self.data_dir = data_dir
        self.target_offset = target_offset
        self.num_frames = num_frames

        # Expand frames into a list of indices
        self.frames = []
        for frame_range in frame_ranges: 
            self.frames.extend(range(*frame_range))

        # Load mean/std for normalization
        stats_dir = os.path.join(data_dir, 'stats')
        mean = np.load(os.path.join(stats_dir, 'mean_full_field.npy')).tolist()
        std  = np.load(os.path.join(stats_dir, 'std_full_field.npy')).tolist()
        self.normalize = Normalize(mean, std)

    def __len__(self):
        return len(self.frames) - self.target_offset - (self.num_frames - 1)

    def __getitem__(self, i: int):
        # Load multiple input frames
        input_frames = []
        for frame_idx in range(self.num_frames):
            frame = self._load_and_norm(self.frames[i + frame_idx])
            input_frames.append(frame)
        
        # Stack frames along temporal dimension: [C, T, H, W]
        input_sequence = torch.stack(input_frames, dim=1)  # [C=2, T=num_frames, H, W]
        
        # Load target frame
        target_frame = self._load_and_norm(self.frames[i + self.num_frames - 1 + self.target_offset])

        return input_sequence, target_frame

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
