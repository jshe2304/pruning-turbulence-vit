import re
import bisect

import torch
from torch.utils.data import ConcatDataset

from src.data.py2d_dataset import Py2DDataset


class MultiPy2DDataset(ConcatDataset):
    def __init__(self, data_dirs, frame_range, stride, target_step, input_step, num_frames=1, **kwargs):
        """
        Dataset that loads from multiple directories (one per Reynolds number).
        Wraps multiple Py2DDataset instances via ConcatDataset.

        Args:
            data_dirs: List of directory paths, each containing data for one Reynolds number
            frame_range: List of [start, end] or list of such lists
            stride: Number of frames between consecutive samples
            target_step: Frames between last input and target
            input_step: Frames between consecutive input frames
            num_frames: Number of consecutive input frames
        """
        self.reynolds_numbers = []
        for d in data_dirs:
            match = re.search(r'Re(\d+)', d)
            assert match, f"Could not parse Reynolds number from {d}"
            self.reynolds_numbers.append(float(match.group(1)))

        datasets = [
            Py2DDataset(d, frame_range, stride, target_step, input_step, num_frames, **kwargs)
            for d in data_dirs
        ]
        super().__init__(datasets)

    def __getitem__(self, i: int):
        ds_idx = bisect.bisect_right(self.cumulative_sizes, i)
        sample_idx = i if ds_idx == 0 else i - self.cumulative_sizes[ds_idx - 1]

        input_frames, target_frame = self.datasets[ds_idx][sample_idx]
        reynolds = torch.tensor(self.reynolds_numbers[ds_idx], dtype=torch.float32)

        return input_frames, target_frame, reynolds
