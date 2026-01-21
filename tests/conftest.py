"""Shared pytest fixtures for testing."""

import pytest
import torch

from src.models.vision_transformer import ViT, Block
from src.models.modules.attention import Attention
from src.models.modules.mlp import MLP
from src.models.modules.embeddings import PatchEmbed


@pytest.fixture
def tiny_model_config():
    """Minimal model configuration for fast CPU testing.

    Key constraints:
    - embed_dim must be divisible by 16 for positional encoding (sin-cos)
    - embed_dim must be divisible by num_heads for attention
    - img_size must be divisible by patch_size
    - num_frames must be divisible by tubelet_size
    """
    return {
        "img_size": 32,
        "patch_size": 8,
        "num_frames": 2,
        "tubelet_size": 2,
        "in_chans": 2,
        "encoder_embed_dim": 32,
        "encoder_depth": 1,
        "encoder_num_heads": 2,
        "decoder_embed_dim": 32,
        "decoder_depth": 1,
        "decoder_num_heads": 2,
        "mlp_ratio": 4.0,
        "num_out_frames": 1,
        "patch_recovery": "subpixel_conv",
    }


@pytest.fixture
def tiny_model(tiny_model_config):
    """Tiny ViT model instance for testing."""
    model = ViT(**tiny_model_config)
    model.eval()
    return model


@pytest.fixture
def tiny_input(tiny_model_config):
    """Random input tensor matching model's expected input shape.

    Shape: (B=2, C=in_chans, T=num_frames, H=img_size, W=img_size)
    """
    batch_size = 2
    return torch.randn(
        batch_size,
        tiny_model_config["in_chans"],
        tiny_model_config["num_frames"],
        tiny_model_config["img_size"],
        tiny_model_config["img_size"],
    )


@pytest.fixture
def tiny_target(tiny_model_config):
    """Random target tensor matching model's expected output shape.

    Shape: (B=2, C=in_chans, T=num_out_frames, H=img_size, W=img_size)
    """
    batch_size = 2
    return torch.randn(
        batch_size,
        tiny_model_config["in_chans"],
        tiny_model_config["num_out_frames"],
        tiny_model_config["img_size"],
        tiny_model_config["img_size"],
    )


@pytest.fixture
def tiny_attention():
    """Tiny Attention module for testing."""
    return Attention(embed_dim=32, num_heads=2, qkv_bias=True)


@pytest.fixture
def tiny_mlp():
    """Tiny MLP module for testing."""
    return MLP(in_features=32, hidden_features=128, out_features=32)


@pytest.fixture
def tiny_patch_embed(tiny_model_config):
    """Tiny PatchEmbed module for testing."""
    return PatchEmbed(
        img_size=tiny_model_config["img_size"],
        patch_size=tiny_model_config["patch_size"],
        num_frames=tiny_model_config["num_frames"],
        tubelet_size=tiny_model_config["tubelet_size"],
        in_chans=tiny_model_config["in_chans"],
        embed_dim=tiny_model_config["encoder_embed_dim"],
    )


@pytest.fixture
def tiny_block():
    """Tiny Block (attention + MLP) for testing."""
    return Block(embed_dim=32, num_heads=2, mlp_ratio=4)


@pytest.fixture
def attention_input():
    """Input tensor for attention testing.

    Shape: (batch_size=2, n_tokens=16, embed_dim=32)
    """
    return torch.randn(2, 16, 32)
