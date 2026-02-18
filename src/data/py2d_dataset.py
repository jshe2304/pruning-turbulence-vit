import os
import math
import re as regex

import numpy as np
from scipy.io import loadmat

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from py2d.initialize import initialize_wavenumbers_rfft2
from py2d.convert import Omega2Psi, Psi2UV

class Py2DDataset(Dataset):
    def __init__(self, data_dir, frame_range, stride, target_step, input_step, num_frames=1, **kwargs): 
        """
        Args:
            data_dir: Path to the data directory
            file_range: List of tuples indicating frame ranges
            target_step: Frames between the last input and target
            stride: Number of frames between inputs
            num_frames: Number of consecutive frames to load for input
        """

        self.data_dir = data_dir
        self.target_step = target_step
        self.input_step = input_step
        self.stride = stride
        self.num_frames = num_frames

        # Parse Reynolds number from data_dir
        match = regex.search(r'Re(\d+)', data_dir)
        self.log_re = torch.tensor(math.log(float(match.group(1))), dtype=torch.float32)

        # Expand frames into a list of indices
        if type(frame_range[0]) is not list: frame_range = [frame_range]
        self.frames = []
        for start_frame, end_frame in frame_range: 
            self.frames.extend(range(start_frame + (num_frames - 1), end_frame, stride))

        # Load mean/std for normalization

        self.mean = np.load(os.path.join(data_dir, 'stats/mean_full_field.npy')).tolist()
        self.std  = np.load(os.path.join(data_dir, 'stats/std_full_field.npy')).tolist()
        self.normalize = Normalize(self.mean, self.std)

        # Make sure all necessary frames are in the data directory

        existing_frames = set(
            int(f.rsplit('.', 1)[0]) 
            for f in os.listdir(os.path.join(data_dir, 'data')) 
            if f.endswith('.mat')
        )

        self.frames = [
            f for f in self.frames 
            if (f in existing_frames and f + self.target_step in existing_frames)
        ]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i: int):
        
        # Load and stack multiple input frames
        input_frames = torch.stack([
            self._load_and_norm(self.frames[i] - t * self.input_step)
            for t in range(self.num_frames)
        ], dim=1)

        # Load target frame
        target_frame = self._load_and_norm(self.frames[i] + self.target_step).unsqueeze(1)

        return input_frames, self.log_re, target_frame

    def _load_and_norm(self, file_num: int) -> torch.Tensor:
        file_path = os.path.join(self.data_dir, f'data/{file_num}.mat')
        omega = loadmat(file_path)['Omega']
        uv = self._omega_to_uv(omega)
        uv = self.normalize(uv)
        return uv

    def _omega_to_uv(self, omega: np.ndarray) -> torch.Tensor:
        nx, ny = omega.shape
        Kx, Ky, _, _, invKsq = initialize_wavenumbers_rfft2(
            nx, ny,
            2*np.pi, 2*np.pi,
            INDEXING='ij'
        )
        psi = Omega2Psi(omega, invKsq)
        u, v = Psi2UV(psi, Kx, Ky)
        return torch.tensor(np.stack([u, v]), dtype=torch.float32)
