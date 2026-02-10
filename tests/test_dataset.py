"""
Tests for Py2DDataset and MultiPy2DDataset.

Py2DDataset tests use synthetic data in tmp_path.
MultiPy2DDataset tests use real data on scratch.
"""

import os

import numpy as np
import pytest
import torch
from scipy.io import savemat

from src.data.py2d_dataset import Py2DDataset
from src.data.multi_py2d_dataset import MultiPy2DDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IMG_SIZE = 64  # grid resolution for synthetic vorticity fields

DATA_ROOT = "/glade/derecho/scratch/jshen/2DTurbData/results"
SUB = "NoSGS/NX256/dt0.0002_IC1"

RE_DIRS = [
    os.path.join(DATA_ROOT, "Re2000_fkx0fky4_r0.1_b20", SUB),
    os.path.join(DATA_ROOT, "Re3000_fkx0fky4_r0.1_b20", SUB),
    os.path.join(DATA_ROOT, "Re4000_fkx0fky4_r0.1_b20", SUB),
]

# Common dataset kwargs for multi-Re tests (small frame range for speed)
MULTI_RE_KWARGS = dict(frame_range=[1000, 1100], stride=10, target_step=1, input_step=1)

data_available = pytest.mark.skipif(
    not all(os.path.isdir(os.path.join(d, "data")) for d in RE_DIRS),
    reason="Real data directories not available",
)


def _make_data_dir(root, name, frame_numbers):
    """Create a fake data directory with .mat files and stats/."""
    d = os.path.join(root, name)
    data_dir = os.path.join(d, "data")
    stats_dir = os.path.join(d, "stats")
    os.makedirs(data_dir)
    os.makedirs(stats_dir)

    for fn in frame_numbers:
        omega = np.random.randn(IMG_SIZE, IMG_SIZE)
        savemat(os.path.join(data_dir, f"{fn}.mat"), {"Omega": omega})

    # 2-channel (u, v) stats
    np.save(os.path.join(stats_dir, "mean_full_field.npy"), np.zeros(2))
    np.save(os.path.join(stats_dir, "std_full_field.npy"), np.ones(2))

    return d


# ---------------------------------------------------------------------------
# Py2DDataset
# ---------------------------------------------------------------------------

class TestPy2DDataset:

    def test_len(self, tmp_path):
        frames = list(range(0, 20))
        d = _make_data_dir(tmp_path, "Re5000_fkx0fky4_r0.1_b20", frames)
        ds = Py2DDataset(d, frame_range=[0, 20], stride=2, target_step=1, input_step=1, num_frames=1)
        # frames 0..18 by stride 2 = [0,2,4,6,8,10,12,14,16,18], each needs frame+1 to exist
        assert len(ds) == 10

    def test_getitem_shapes(self, tmp_path):
        frames = list(range(0, 10))
        d = _make_data_dir(tmp_path, "Re5000_fkx0fky4_r0.1_b20", frames)
        ds = Py2DDataset(d, frame_range=[0, 10], stride=1, target_step=1, input_step=1, num_frames=2)
        inp, tgt = ds[0]
        assert inp.shape == (2, 2, IMG_SIZE, IMG_SIZE)   # [C, num_frames, H, W]
        assert tgt.shape == (2, 1, IMG_SIZE, IMG_SIZE)    # [C, 1, H, W]

    def test_missing_target_frame_excluded(self, tmp_path):
        # frames 0-4 exist, target_step=10 means no valid samples
        frames = list(range(0, 5))
        d = _make_data_dir(tmp_path, "Re5000_fkx0fky4_r0.1_b20", frames)
        ds = Py2DDataset(d, frame_range=[0, 5], stride=1, target_step=10, input_step=1, num_frames=1)
        assert len(ds) == 0

    def test_multi_range(self, tmp_path):
        frames = list(range(0, 30))
        d = _make_data_dir(tmp_path, "Re5000_fkx0fky4_r0.1_b20", frames)
        ds = Py2DDataset(d, frame_range=[[0, 10], [20, 30]], stride=1, target_step=1, input_step=1, num_frames=1)
        # range [0,10) -> 0..9, all have target f+1 <= 29 => 10
        # range [20,30) -> 20..29, frame 29 needs 30 which doesn't exist => 9
        assert len(ds) == 19


# ---------------------------------------------------------------------------
# MultiPy2DDataset
# ---------------------------------------------------------------------------

@data_available
class TestMultiPy2DDataset:

    def test_len_combines_dirs(self):
        ds = MultiPy2DDataset(RE_DIRS, **MULTI_RE_KWARGS)
        # [1000, 1100) stride 10 = frames 1000,1010,...,1090 = 10 per dir, 3 dirs = 30
        assert len(ds) == 30

    def test_getitem_returns_three_tuple(self):
        ds = MultiPy2DDataset(RE_DIRS, **MULTI_RE_KWARGS)
        result = ds[0]
        assert len(result) == 3

    def test_getitem_shapes(self):
        ds = MultiPy2DDataset(RE_DIRS, **MULTI_RE_KWARGS, num_frames=2)
        inp, re, tgt = ds[0]
        assert inp.shape == (2, 2, 256, 256)
        assert tgt.shape == (2, 1, 256, 256)
        assert re.shape == ()
        assert re.dtype == torch.float32

    def test_no_nans(self):
        ds = MultiPy2DDataset(RE_DIRS, **MULTI_RE_KWARGS)
        inp, re, tgt = ds[0]
        assert not torch.isnan(inp).any()
        assert not torch.isnan(tgt).any()

    def test_reynolds_number_parsing(self):
        ds = MultiPy2DDataset(RE_DIRS, **MULTI_RE_KWARGS)
        assert ds.reynolds_numbers == [2000.0, 3000.0, 4000.0]

    def test_reynolds_values_in_samples(self):
        ds = MultiPy2DDataset(RE_DIRS, **MULTI_RE_KWARGS)
        _, re_first, _ = ds[0]
        _, re_last, _ = ds[len(ds) - 1]
        assert re_first.item() == 2000.0
        assert re_last.item() == 4000.0

    def test_samples_ordered_by_directory(self):
        ds = MultiPy2DDataset(RE_DIRS, **MULTI_RE_KWARGS)
        per_dir = len(ds) // len(RE_DIRS)
        for i in range(per_dir):
            _, re_val, _ = ds[i]
            assert re_val.item() == 2000.0
        for i in range(per_dir, 2 * per_dir):
            _, re_val, _ = ds[i]
            assert re_val.item() == 3000.0
        for i in range(2 * per_dir, len(ds)):
            _, re_val, _ = ds[i]
            assert re_val.item() == 4000.0

    def test_empty_when_no_valid_frames(self):
        # Frame range way beyond available data
        ds = MultiPy2DDataset(RE_DIRS[:1], frame_range=[999999, 999999], stride=1, target_step=1, input_step=1)
        assert len(ds) == 0

    def test_subset_dirs(self):
        ds2 = MultiPy2DDataset(RE_DIRS[:2], **MULTI_RE_KWARGS)
        ds3 = MultiPy2DDataset(RE_DIRS, **MULTI_RE_KWARGS)
        assert len(ds3) > len(ds2)
