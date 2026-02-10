"""Compute per-channel (u, v) mean, std, min, and max for a data directory.

Saves mean_full_field.npy, std_full_field.npy, min_full_field.npy, and
max_full_field.npy into a stats/ subfolder.

Usage:
    python compute_stats.py /path/to/Re5000_.../dt0.0002_IC1
    python compute_stats.py dir1 dir2 dir3
    python compute_stats.py dir1 --sample-every 50
"""

import argparse
import os

import numpy as np
from scipy.io import loadmat

from py2d.initialize import initialize_wavenumbers_rfft2
from py2d.convert import Omega2Psi, Psi2UV


def compute_stats(data_dir, sample_every=100):
    data_path = os.path.join(data_dir, "data")
    stats_path = os.path.join(data_dir, "stats")
    os.makedirs(stats_path, exist_ok=True)

    files = sorted([f for f in os.listdir(data_path) if f.endswith(".mat")])
    sampled = files[::sample_every]
    print(f"{data_dir}: {len(files)} frames, using {len(sampled)} for stats")

    # Streaming mean/std to avoid storing all frames in memory
    n = 0
    u_sum, v_sum = 0.0, 0.0
    u_sum_sq, v_sum_sq = 0.0, 0.0
    u_min, v_min = np.inf, np.inf
    u_max, v_max = -np.inf, -np.inf
    Kx = None

    for fn in sampled:
        omega = loadmat(os.path.join(data_path, fn))["Omega"]
        nx, ny = omega.shape
        if Kx is None:
            Kx, Ky, _, _, invKsq = initialize_wavenumbers_rfft2(
                nx, ny, 2 * np.pi, 2 * np.pi, INDEXING="ij"
            )
        psi = Omega2Psi(omega, invKsq)
        u, v = Psi2UV(psi, Kx, Ky)
        n += u.size
        u_sum += u.sum()
        v_sum += v.sum()
        u_sum_sq += (u ** 2).sum()
        v_sum_sq += (v ** 2).sum()
        u_min = min(u_min, u.min())
        v_min = min(v_min, v.min())
        u_max = max(u_max, u.max())
        v_max = max(v_max, v.max())

    u_mean, v_mean = u_sum / n, v_sum / n
    mean = np.array([u_mean, v_mean])
    std = np.array([
        np.sqrt(u_sum_sq / n - u_mean ** 2),
        np.sqrt(v_sum_sq / n - v_mean ** 2),
    ])

    mins = np.array([u_min, v_min])
    maxs = np.array([u_max, v_max])

    np.save(os.path.join(stats_path, "mean_full_field.npy"), mean)
    np.save(os.path.join(stats_path, "std_full_field.npy"), std)
    np.save(os.path.join(stats_path, "min_full_field.npy"), mins)
    np.save(os.path.join(stats_path, "max_full_field.npy"), maxs)
    print(f"  mean={mean}, std={std}, min={mins}, max={maxs}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dirs", nargs="+", help="Data directories (containing data/ subfolder)")
    parser.add_argument("--sample-every", type=int, default=100, help="Sample every Nth frame")
    args = parser.parse_args()

    for d in args.data_dirs:
        compute_stats(d, args.sample_every)


if __name__ == "__main__":
    main()
