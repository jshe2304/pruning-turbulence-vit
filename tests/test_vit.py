"""Tests for the Vision Transformer model."""

import pytest
import torch
import torch.nn as nn


class TestForwardPass:
    """Tests for forward pass behavior."""

    def test_forward_pass_runs(self, tiny_model, tiny_input):
        """Verify forward pass completes without error."""
        with torch.no_grad():
            output = tiny_model(tiny_input)
        assert output is not None

    def test_forward_pass_output_shape(self, tiny_model, tiny_input, tiny_model_config):
        """Verify output has correct shape (B, C, T_out, H, W)."""
        with torch.no_grad():
            output = tiny_model(tiny_input)

        batch_size = tiny_input.shape[0]
        expected_shape = (
            batch_size,
            tiny_model_config["in_chans"],
            tiny_model_config["num_out_frames"],
            tiny_model_config["img_size"],
            tiny_model_config["img_size"],
        )
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

    def test_forward_pass_no_nan(self, tiny_model, tiny_input):
        """Verify output contains no NaN values."""
        with torch.no_grad():
            output = tiny_model(tiny_input)
        assert not torch.isnan(output).any(), "Output contains NaN values"

    def test_forward_pass_no_inf(self, tiny_model, tiny_input):
        """Verify output contains no infinite values."""
        with torch.no_grad():
            output = tiny_model(tiny_input)
        assert not torch.isinf(output).any(), "Output contains infinite values"

    def test_encoder_output_shape(self, tiny_model, tiny_input):
        """Verify encoder produces correct latent shape."""
        with torch.no_grad():
            latent = tiny_model.forward_encoder(tiny_input)

        batch_size = tiny_input.shape[0]
        num_patches = tiny_model.patch_embed.num_patches
        embed_dim = tiny_model.pos_embed.shape[-1]

        expected_shape = (batch_size, num_patches, embed_dim)
        assert latent.shape == expected_shape, f"Expected {expected_shape}, got {latent.shape}"

    def test_decoder_output_shape(self, tiny_model, tiny_input, tiny_model_config):
        """Verify decoder produces correct output shape."""
        with torch.no_grad():
            latent = tiny_model.forward_encoder(tiny_input)
            output = tiny_model.forward_decoder(latent)

        batch_size = tiny_input.shape[0]
        expected_shape = (
            batch_size,
            tiny_model_config["in_chans"],
            tiny_model_config["num_out_frames"],
            tiny_model_config["img_size"],
            tiny_model_config["img_size"],
        )
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

    def test_train_mode_runs(self, tiny_model, tiny_input):
        """Verify forward pass works in train mode."""
        tiny_model.train()
        output = tiny_model(tiny_input, train=True)
        assert output is not None


class TestBackwardPass:
    """Tests for backward pass behavior."""

    def test_backward_pass_runs(self, tiny_model, tiny_input, tiny_target):
        """Verify backward pass completes without error."""
        tiny_model.train()
        output = tiny_model(tiny_input)
        loss = tiny_model.forward_loss(tiny_target, output)
        loss.backward()
        # If we get here, backward pass succeeded

    def test_gradients_exist(self, tiny_model, tiny_input, tiny_target):
        """Verify all parameters have gradients after backward pass."""
        tiny_model.train()
        tiny_model.zero_grad()

        output = tiny_model(tiny_input)
        loss = tiny_model.forward_loss(tiny_target, output)
        loss.backward()

        for name, param in tiny_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"

    def test_gradients_not_all_zero(self, tiny_model, tiny_input, tiny_target):
        """Verify gradients are non-zero (model is learning)."""
        tiny_model.train()
        tiny_model.zero_grad()

        output = tiny_model(tiny_input)
        loss = tiny_model.forward_loss(tiny_target, output)
        loss.backward()

        has_nonzero_grad = False
        for name, param in tiny_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if param.grad.abs().sum() > 0:
                    has_nonzero_grad = True
                    break

        assert has_nonzero_grad, "All gradients are zero"

    def test_gradients_no_nan(self, tiny_model, tiny_input, tiny_target):
        """Verify gradients contain no NaN values."""
        tiny_model.train()
        tiny_model.zero_grad()

        output = tiny_model(tiny_input)
        loss = tiny_model.forward_loss(tiny_target, output)
        loss.backward()

        for name, param in tiny_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient"

    def test_gradients_no_inf(self, tiny_model, tiny_input, tiny_target):
        """Verify gradients contain no infinite values."""
        tiny_model.train()
        tiny_model.zero_grad()

        output = tiny_model(tiny_input)
        loss = tiny_model.forward_loss(tiny_target, output)
        loss.backward()

        for name, param in tiny_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.isinf(param.grad).any(), f"Parameter {name} has infinite gradient"

    def test_loss_decreases_with_steps(self, tiny_model, tiny_input, tiny_target):
        """Verify optimizer steps reduce loss (basic learning check)."""
        tiny_model.train()

        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=0.01)

        # Initial forward pass
        output1 = tiny_model(tiny_input)
        loss1 = tiny_model.forward_loss(tiny_target, output1).item()

        # Multiple backward + step iterations for reliable convergence
        for _ in range(5):
            optimizer.zero_grad()
            output = tiny_model(tiny_input)
            loss = tiny_model.forward_loss(tiny_target, output)
            loss.backward()
            optimizer.step()

        # Final forward pass
        with torch.no_grad():
            output2 = tiny_model(tiny_input)
            loss2 = tiny_model.forward_loss(tiny_target, output2).item()

        assert loss2 < loss1, f"Loss did not decrease: {loss1:.6f} -> {loss2:.6f}"


class TestSaveLoad:
    """Tests for model save/load functionality."""

    def test_state_dict_save_load(self, tiny_model, tiny_input):
        """Verify state dict roundtrip preserves model outputs."""
        tiny_model.eval()

        # Get output before save
        with torch.no_grad():
            output_before = tiny_model(tiny_input)

        # Save and load state dict
        state_dict = tiny_model.state_dict()
        tiny_model.load_state_dict(state_dict)

        # Get output after load
        with torch.no_grad():
            output_after = tiny_model(tiny_input)

        assert torch.allclose(output_before, output_after, atol=1e-6), "Outputs differ after state dict roundtrip"

    def test_state_dict_keys(self, tiny_model):
        """Verify state dict contains expected keys."""
        state_dict = tiny_model.state_dict()

        # Check for essential components
        expected_prefixes = [
            "patch_embed",
            "pos_embed",
            "encoder_blocks",
            "norm",
            "decoder_embed",
            "decoder_pos_embed",
            "decoder_blocks",
            "decoder_norm",
            "patchrecovery",
        ]

        for prefix in expected_prefixes:
            matching_keys = [k for k in state_dict.keys() if k.startswith(prefix)]
            assert len(matching_keys) > 0, f"No keys found with prefix '{prefix}'"


class TestModelProperties:
    """Tests for model properties and utilities."""

    def test_n_parameters(self, tiny_model):
        """Verify n_parameters returns positive count."""
        n_params = tiny_model.n_parameters()
        assert n_params > 0, "Model should have trainable parameters"

    def test_n_pruned_parameters_initially_zero(self, tiny_model):
        """Verify n_pruned_parameters is 0 for unpruned model."""
        n_pruned = tiny_model.n_pruned_parameters()
        assert n_pruned == 0, "Unpruned model should have 0 pruned parameters"

    def test_get_weights_returns_list(self, tiny_model):
        """Verify get_weights returns list of weight tuples."""
        weights = tiny_model.get_weights()
        assert isinstance(weights, list), "get_weights should return a list"
        assert len(weights) > 0, "get_weights should return non-empty list"
        for item in weights:
            assert isinstance(item, tuple), "Each item should be a tuple"
            assert len(item) == 2, "Each tuple should have 2 elements"


class TestLossFunctions:
    """Tests for loss function behavior."""

    def test_forward_loss_returns_scalar(self, tiny_model, tiny_input, tiny_target):
        """Verify forward_loss returns a scalar tensor."""
        with torch.no_grad():
            output = tiny_model(tiny_input)
            loss = tiny_model.forward_loss(tiny_target, output)

        assert loss.ndim == 0, "Loss should be a scalar"
        assert loss.item() >= 0, "MSE loss should be non-negative"

    def test_forward_loss_with_weights(self, tiny_model, tiny_input, tiny_target):
        """Verify weighted loss works correctly."""
        with torch.no_grad():
            output = tiny_model(tiny_input)
            weights = torch.ones_like(tiny_target)
            loss = tiny_model.forward_loss(tiny_target, output, weights=weights)

        assert loss.ndim == 0, "Loss should be a scalar"

    def test_spectral_loss_returns_scalar(self, tiny_model, tiny_input, tiny_target):
        """Verify spectral_loss returns a scalar tensor."""
        with torch.no_grad():
            output = tiny_model(tiny_input)
            loss = tiny_model.spectral_loss(tiny_target, output, weight=1.0, threshold_wavenumber=2)

        assert loss.ndim == 0, "Spectral loss should be a scalar"
