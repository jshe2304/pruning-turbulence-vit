"""Tests for the MLP module."""

import pytest
import torch
import torch.nn as nn

from src.models.modules.mlp import MLP


class TestMLPForward:
    """Tests for MLP forward pass."""

    def test_forward_pass_runs(self, tiny_mlp, attention_input):
        """Verify forward pass completes without error."""
        with torch.no_grad():
            output = tiny_mlp(attention_input)
        assert output is not None

    def test_output_shape(self, tiny_mlp, attention_input):
        """Verify output shape matches expected dimensions."""
        with torch.no_grad():
            output = tiny_mlp(attention_input)
        # MLP has same in_features and out_features (32)
        assert output.shape == attention_input.shape

    def test_output_no_nan(self, tiny_mlp, attention_input):
        """Verify output contains no NaN values."""
        with torch.no_grad():
            output = tiny_mlp(attention_input)
        assert not torch.isnan(output).any(), "Output contains NaN values"

    def test_output_no_inf(self, tiny_mlp, attention_input):
        """Verify output contains no infinite values."""
        with torch.no_grad():
            output = tiny_mlp(attention_input)
        assert not torch.isinf(output).any(), "Output contains infinite values"


class TestMLPBackward:
    """Tests for MLP backward pass."""

    def test_backward_pass_runs(self, tiny_mlp, attention_input):
        """Verify backward pass completes without error."""
        attention_input.requires_grad_(True)
        output = tiny_mlp(attention_input)
        loss = output.sum()
        loss.backward()

    def test_gradients_exist(self, tiny_mlp, attention_input):
        """Verify all parameters have gradients after backward pass."""
        tiny_mlp.zero_grad()
        output = tiny_mlp(attention_input)
        loss = output.sum()
        loss.backward()

        for name, param in tiny_mlp.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"

    def test_gradients_no_nan(self, tiny_mlp, attention_input):
        """Verify gradients contain no NaN values."""
        tiny_mlp.zero_grad()
        output = tiny_mlp(attention_input)
        loss = output.sum()
        loss.backward()

        for name, param in tiny_mlp.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient"


class TestMLPConfig:
    """Tests for MLP configuration."""

    def test_different_dimensions(self):
        """Verify MLP works with different input/output dimensions."""
        configs = [
            {"in_features": 32, "hidden_features": 128, "out_features": 32},
            {"in_features": 64, "hidden_features": 256, "out_features": 64},
            {"in_features": 32, "hidden_features": 64, "out_features": 16},
        ]

        for config in configs:
            mlp = MLP(**config)
            x = torch.randn(2, 16, config["in_features"])
            with torch.no_grad():
                output = mlp(x)
            expected_shape = (2, 16, config["out_features"])
            assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

    def test_custom_activation(self):
        """Verify MLP works with custom activation functions."""
        for act_layer in [nn.GELU, nn.ReLU, nn.SiLU]:
            mlp = MLP(in_features=32, hidden_features=128, out_features=32, act_layer=act_layer)
            x = torch.randn(2, 16, 32)
            with torch.no_grad():
                output = mlp(x)
            assert output.shape == x.shape


class TestMLPExpansionRatio:
    """Tests for MLP expansion ratio behavior."""

    def test_mlp_ratio_4x(self):
        """Verify MLP with 4x expansion ratio (typical)."""
        embed_dim = 32
        mlp_ratio = 4
        mlp = MLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            out_features=embed_dim,
        )

        # Check hidden layer size
        assert mlp.fc1.out_features == 128  # 32 * 4

        x = torch.randn(2, 16, embed_dim)
        with torch.no_grad():
            output = mlp(x)
        assert output.shape == x.shape
