import os
import numpy as np
from scipy.io import loadmat

from py2d.convert import UV2Omega, Omega2UV

# def save_numpy_data(filepath, data):
#     """Save numpy data to file."""
#     np.save(filepath, data)

# def load_numpy_data(filepath):
#     """Load numpy data from file if exists."""
#     if os.path.exists(filepath):
#         return np.load(filepath)
#     return None

# def get_npy_files(folder_path):
#     # List all .npy files in the folder
#     npy_files = [file for file in os.listdir(folder_path) if file.endswith('.npy')]
    
#     # Sort the files numerically based on their numeric part
#     npy_files.sort(key=lambda x: int(x.split('.')[0]))
    
#     return npy_files

def get_mat_files_in_range(data_dir, file_range):
    """
    Retrieves .mat file names within the specified range or ranges.

    Args:
        data_dir (str): Path to the directory containing .mat files.
        file_range (list): A single [start, end] list or a list of such lists.

    Returns:
        list: List of .mat file names within the specified range(s).
    """
    def in_any_range(number, ranges):
        """Check if number falls in any of the given ranges."""
        return any(start <= number <= end for start, end in ranges)

    # Normalize to list of ranges
    if not file_range or not isinstance(file_range[0], list):
        ranges = [file_range]  # Single range case
    else:
        ranges = file_range    # List of ranges

    all_files = os.listdir(data_dir)
    filtered_files = []

    for file_name in all_files:
        if file_name.endswith('.mat'):
            try:
                number = int(os.path.splitext(file_name)[0])
                if in_any_range(number, ranges):
                    filtered_files.append(file_name)
            except ValueError:
                pass  # Skip files without numeric names

    filtered_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    return filtered_files

def frame_generator(files, inference_dir, Kx, Ky):
    """
    Yields (U, V, Omega) one timestep at a time.
    - For 'emulate', each file is a .npy chunk of shape (N_chunk, 2, H, W)
    - For 'train'/'truth', each file is a .mat with Omega
    """
    for fname in files:
        chunk = np.load(os.path.join(inference_dir, fname))   # only this chunk in memory

        if chunk.ndim == 3:
            U, V = chunk[0], chunk[1]
            Omega = UV2Omega(U.T, V.T, Kx, Ky, spectral=False).T
            yield U.astype(np.float32), V.astype(np.float32), Omega.astype(np.float32)
        elif chunk.ndim == 4:
            for frame in chunk:
                U, V = frame[0], frame[1]
                Omega = UV2Omega(U.T, V.T, Kx, Ky, spectral=False).T
                yield U.astype(np.float32), V.astype(np.float32), Omega.astype(np.float32)
        
def get_npy_files(folder_path):
    # List all .npy files in the folder
    npy_files = [file for file in os.listdir(folder_path) if file.endswith('.npy')]
    
    # Sort the files numerically based on their numeric part
    npy_files.sort(key=lambda x: int(x.split('.')[0]))
    
    return npy_files
