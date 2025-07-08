import torch
import numpy as np
from torch.utils.data import TensorDataset
import os


class TimeSeriesDataset(TensorDataset):
    """
    Dataset for time series data. 
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        """
        Returns the current and next frame in the time series. 
        """
        if index >= self.data.shape[0] - 1:
            raise IndexError('Index out of range')
        
        return self.data[index], self.data[index + 1]

def load_data(datadir):
    """
    Loads data from provided directory. 
    """

    n_frames = 1000

    data = torch.zeros(n_frames, 256, 256)
    for i in range(n_frames):
        data[i] = np.load(os.path.join(datadir, f'{i}.npy'))

    return TimeSeriesDataset(data)