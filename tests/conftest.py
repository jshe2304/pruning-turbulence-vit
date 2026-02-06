import pytest
import torch


# ---------------------------------------------------------------------------
# Small model configs for fast CPU tests
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_vit_config():
    return dict(
        img_size=32,
        patch_size=4,
        embed_dim=32,
        depth=2,
        num_heads=2,
        num_frames=2,
    )


@pytest.fixture
def vit_config():
    return dict(
        img_size=32,
        patch_size=4,
        num_frames=2,
        tubelet_size=2,
        in_chans=2,
        encoder_embed_dim=32,
        encoder_depth=2,
        encoder_num_heads=2,
        decoder_embed_dim=32,
        decoder_depth=2,
        decoder_num_heads=2,
        mlp_ratio=4.,
        num_out_frames=1,
    )


# ---------------------------------------------------------------------------
# Dummy inputs matching the configs above
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_input_simple_vit(simple_vit_config):
    """B=2, C=2, T=num_frames, H=img_size, W=img_size"""
    B = 2
    C = 2
    T = simple_vit_config["num_frames"]
    H = W = simple_vit_config["img_size"]
    return torch.randn(B, C, T, H, W)


@pytest.fixture
def dummy_input_vit(vit_config):
    """B=2, C=in_chans, T=num_frames, H=img_size, W=img_size"""
    B = 2
    C = vit_config["in_chans"]
    T = vit_config["num_frames"]
    H = W = vit_config["img_size"]
    return torch.randn(B, C, T, H, W)
