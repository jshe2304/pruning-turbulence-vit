"""Tests for the PatchEmbed module."""

import pytest
import torch

from src.models.modules.embeddings import PatchEmbed


class TestPatchEmbedForward:
    """Tests for PatchEmbed forward pass."""

    def test_forward_pass_runs(self, tiny_patch_embed, tiny_input):
        """Verify forward pass completes without error."""
        with torch.no_grad():
            output = tiny_patch_embed(tiny_input)
        assert output is not None

    def test_output_shape(self, tiny_patch_embed, tiny_input):
        """Verify output has shape (B, num_patches, embed_dim)."""
        with torch.no_grad():
            output = tiny_patch_embed(tiny_input)

        batch_size = tiny_input.shape[0]
        num_patches = tiny_patch_embed.num_patches
        embed_dim = tiny_patch_embed.proj.out_channels

        expected_shape = (batch_size, num_patches, embed_dim)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

    def test_output_no_nan(self, tiny_patch_embed, tiny_input):
        """Verify output contains no NaN values."""
        with torch.no_grad():
            output = tiny_patch_embed(tiny_input)
        assert not torch.isnan(output).any(), "Output contains NaN values"


class TestPatchEmbedBackward:
    """Tests for PatchEmbed backward pass."""

    def test_backward_pass_runs(self, tiny_patch_embed, tiny_input):
        """Verify backward pass completes without error."""
        tiny_input.requires_grad_(True)
        output = tiny_patch_embed(tiny_input)
        loss = output.sum()
        loss.backward()

    def test_gradients_exist(self, tiny_patch_embed, tiny_input):
        """Verify all parameters have gradients after backward pass."""
        tiny_patch_embed.zero_grad()
        output = tiny_patch_embed(tiny_input)
        loss = output.sum()
        loss.backward()

        for name, param in tiny_patch_embed.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"


class TestPatchEmbedNumPatches:
    """Tests for num_patches calculation."""

    def test_num_patches_calculation(self, tiny_patch_embed, tiny_model_config):
        """Verify num_patches is calculated correctly."""
        img_size = tiny_model_config["img_size"]
        patch_size = tiny_model_config["patch_size"]
        num_frames = tiny_model_config["num_frames"]
        tubelet_size = tiny_model_config["tubelet_size"]

        # grid_size = (T // tubelet, H // patch, W // patch)
        expected_grid = (
            num_frames // tubelet_size,  # 2 // 2 = 1
            img_size // patch_size,  # 32 // 8 = 4
            img_size // patch_size,  # 32 // 8 = 4
        )
        expected_num_patches = expected_grid[0] * expected_grid[1] * expected_grid[2]  # 1 * 4 * 4 = 16

        assert tiny_patch_embed.grid_size == expected_grid
        assert tiny_patch_embed.num_patches == expected_num_patches

    def test_various_patch_sizes(self):
        """Verify num_patches for different patch/image sizes."""
        configs = [
            {"img_size": 64, "patch_size": 8, "num_frames": 4, "tubelet_size": 2, "expected": 2 * 8 * 8},
            {"img_size": 32, "patch_size": 4, "num_frames": 2, "tubelet_size": 1, "expected": 2 * 8 * 8},
            {"img_size": 128, "patch_size": 16, "num_frames": 2, "tubelet_size": 2, "expected": 1 * 8 * 8},
        ]

        for config in configs:
            patch_embed = PatchEmbed(
                img_size=config["img_size"],
                patch_size=config["patch_size"],
                num_frames=config["num_frames"],
                tubelet_size=config["tubelet_size"],
                in_chans=2,
                embed_dim=32,
            )
            assert patch_embed.num_patches == config["expected"], (
                f"Expected {config['expected']}, got {patch_embed.num_patches}"
            )


class TestPatchEmbedConfig:
    """Tests for PatchEmbed configuration options."""

    def test_with_norm_layer(self):
        """Verify PatchEmbed works with norm layer."""
        import torch.nn as nn

        patch_embed = PatchEmbed(
            img_size=32,
            patch_size=8,
            num_frames=2,
            tubelet_size=2,
            in_chans=2,
            embed_dim=32,
            norm_layer=nn.LayerNorm,
        )

        x = torch.randn(2, 2, 2, 32, 32)
        with torch.no_grad():
            output = patch_embed(x)

        assert output.shape == (2, 16, 32)

    def test_without_norm_layer(self):
        """Verify PatchEmbed works without norm layer."""
        patch_embed = PatchEmbed(
            img_size=32,
            patch_size=8,
            num_frames=2,
            tubelet_size=2,
            in_chans=2,
            embed_dim=32,
            norm_layer=None,
        )

        x = torch.randn(2, 2, 2, 32, 32)
        with torch.no_grad():
            output = patch_embed(x)

        assert output.shape == (2, 16, 32)

    def test_flatten_option(self):
        """Verify PatchEmbed flatten option works."""
        # With flatten=False, output should not be reshaped
        patch_embed = PatchEmbed(
            img_size=32,
            patch_size=8,
            num_frames=2,
            tubelet_size=2,
            in_chans=2,
            embed_dim=32,
            flatten=False,
        )

        x = torch.randn(2, 2, 2, 32, 32)
        with torch.no_grad():
            output = patch_embed(x)

        # Without flatten: (B, embed_dim, T', H', W')
        assert output.shape == (2, 32, 1, 4, 4)
