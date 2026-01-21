"""Tests for the Attention module."""

import pytest
import torch

from src.models.modules.attention import Attention


class TestAttentionForward:
    """Tests for Attention forward pass."""

    def test_forward_pass_runs(self, tiny_attention, attention_input):
        """Verify forward pass completes without error."""
        with torch.no_grad():
            output = tiny_attention(attention_input)
        assert output is not None

    def test_output_shape_matches_input(self, tiny_attention, attention_input):
        """Verify output shape matches input shape."""
        with torch.no_grad():
            output = tiny_attention(attention_input)
        assert output.shape == attention_input.shape

    def test_output_no_nan(self, tiny_attention, attention_input):
        """Verify output contains no NaN values."""
        with torch.no_grad():
            output = tiny_attention(attention_input)
        assert not torch.isnan(output).any(), "Output contains NaN values"

    def test_output_no_inf(self, tiny_attention, attention_input):
        """Verify output contains no infinite values."""
        with torch.no_grad():
            output = tiny_attention(attention_input)
        assert not torch.isinf(output).any(), "Output contains infinite values"


class TestAttentionBackward:
    """Tests for Attention backward pass."""

    def test_backward_pass_runs(self, tiny_attention, attention_input):
        """Verify backward pass completes without error."""
        attention_input.requires_grad_(True)
        output = tiny_attention(attention_input)
        loss = output.sum()
        loss.backward()

    def test_gradients_exist(self, tiny_attention, attention_input):
        """Verify all parameters have gradients after backward pass."""
        tiny_attention.zero_grad()
        output = tiny_attention(attention_input)
        loss = output.sum()
        loss.backward()

        for name, param in tiny_attention.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"

    def test_gradients_no_nan(self, tiny_attention, attention_input):
        """Verify gradients contain no NaN values."""
        tiny_attention.zero_grad()
        output = tiny_attention(attention_input)
        loss = output.sum()
        loss.backward()

        for name, param in tiny_attention.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient"


class TestAttentionConfig:
    """Tests for Attention configuration."""

    def test_embed_dim_divisibility(self):
        """Verify attention fails if embed_dim not divisible by num_heads."""
        with pytest.raises(AssertionError):
            Attention(embed_dim=33, num_heads=4)  # 33 not divisible by 4

    def test_various_num_heads(self):
        """Verify attention works with various num_heads configurations."""
        for num_heads in [1, 2, 4, 8]:
            embed_dim = 32  # Must be divisible by num_heads
            if embed_dim % num_heads == 0:
                attn = Attention(embed_dim=embed_dim, num_heads=num_heads)
                x = torch.randn(2, 16, embed_dim)
                with torch.no_grad():
                    output = attn(x)
                assert output.shape == x.shape

    def test_qkv_bias_options(self):
        """Verify attention works with and without QKV bias."""
        for qkv_bias in [True, False]:
            attn = Attention(embed_dim=32, num_heads=2, qkv_bias=qkv_bias)
            x = torch.randn(2, 16, 32)
            with torch.no_grad():
                output = attn(x)
            assert output.shape == x.shape

    def test_proj_bias_options(self):
        """Verify attention works with and without projection bias."""
        for proj_bias in [True, False]:
            attn = Attention(embed_dim=32, num_heads=2, proj_bias=proj_bias)
            x = torch.randn(2, 16, 32)
            with torch.no_grad():
                output = attn(x)
            assert output.shape == x.shape
