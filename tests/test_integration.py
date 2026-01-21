"""Integration tests for training loops and data processing."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TestMiniTrainingLoop:
    """Tests for basic training loop functionality."""

    def test_mini_training_loop(self, tiny_model, tiny_model_config):
        """Verify a mini training loop completes successfully."""
        tiny_model.train()

        # Create synthetic dataset
        batch_size = 2
        num_samples = 4
        inputs = torch.randn(
            num_samples,
            tiny_model_config["in_chans"],
            tiny_model_config["num_frames"],
            tiny_model_config["img_size"],
            tiny_model_config["img_size"],
        )
        targets = torch.randn(
            num_samples,
            tiny_model_config["in_chans"],
            tiny_model_config["num_out_frames"],
            tiny_model_config["img_size"],
            tiny_model_config["img_size"],
        )

        dataset = TensorDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=0.001)

        # Train for 2 epochs
        num_epochs = 2
        losses = []

        for epoch in range(num_epochs):
            for batch_inputs, batch_targets in dataloader:
                optimizer.zero_grad()
                outputs = tiny_model(batch_inputs)
                loss = tiny_model.forward_loss(batch_targets, outputs)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        # Verify training completed
        assert len(losses) == num_epochs * (num_samples // batch_size)
        # Verify loss values are reasonable
        assert all(l >= 0 for l in losses), "Loss values should be non-negative"
        assert all(not torch.isnan(torch.tensor(l)) for l in losses), "Loss should not be NaN"

    def test_training_reduces_loss(self, tiny_model, tiny_model_config):
        """Verify training reduces loss over epochs."""
        tiny_model.train()

        # Fixed synthetic data (same for all epochs)
        inputs = torch.randn(
            4,
            tiny_model_config["in_chans"],
            tiny_model_config["num_frames"],
            tiny_model_config["img_size"],
            tiny_model_config["img_size"],
        )
        targets = torch.randn(
            4,
            tiny_model_config["in_chans"],
            tiny_model_config["num_out_frames"],
            tiny_model_config["img_size"],
            tiny_model_config["img_size"],
        )

        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=0.01)

        # Compute initial loss
        with torch.no_grad():
            initial_outputs = tiny_model(inputs)
            initial_loss = tiny_model.forward_loss(targets, initial_outputs).item()

        # Train for several steps
        for _ in range(10):
            optimizer.zero_grad()
            outputs = tiny_model(inputs)
            loss = tiny_model.forward_loss(targets, outputs)
            loss.backward()
            optimizer.step()

        # Compute final loss
        with torch.no_grad():
            final_outputs = tiny_model(inputs)
            final_loss = tiny_model.forward_loss(targets, final_outputs).item()

        assert final_loss < initial_loss, f"Loss should decrease: {initial_loss:.6f} -> {final_loss:.6f}"

class TestRolloutTraining:
    """Tests for multi-step autoregressive rollout training."""

    def test_rollout_training_single_step(self, tiny_model, tiny_model_config):
        """Verify single-step rollout training works."""
        tiny_model.train()

        # Input: 2 frames, output: 1 frame
        inputs = torch.randn(
            2,
            tiny_model_config["in_chans"],
            tiny_model_config["num_frames"],
            tiny_model_config["img_size"],
            tiny_model_config["img_size"],
        )
        targets = torch.randn(
            2,
            tiny_model_config["in_chans"],
            tiny_model_config["num_out_frames"],
            tiny_model_config["img_size"],
            tiny_model_config["img_size"],
        )

        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=0.001)

        # Single step rollout (equivalent to standard forward)
        optimizer.zero_grad()
        outputs = tiny_model(inputs)
        loss = tiny_model.forward_loss(targets, outputs)
        loss.backward()
        optimizer.step()

        assert loss.item() >= 0

    def test_rollout_training_multi_step(self, tiny_model, tiny_model_config):
        """Verify multi-step rollout training with autoregressive prediction."""
        tiny_model.train()

        num_rollout_steps = 2
        batch_size = 2

        # Initial input
        current_input = torch.randn(
            batch_size,
            tiny_model_config["in_chans"],
            tiny_model_config["num_frames"],
            tiny_model_config["img_size"],
            tiny_model_config["img_size"],
        )

        # Target for each rollout step
        targets = [
            torch.randn(
                batch_size,
                tiny_model_config["in_chans"],
                tiny_model_config["num_out_frames"],
                tiny_model_config["img_size"],
                tiny_model_config["img_size"],
            )
            for _ in range(num_rollout_steps)
        ]

        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=0.001)
        optimizer.zero_grad()

        total_loss = 0.0

        # Autoregressive rollout
        for step in range(num_rollout_steps):
            output = tiny_model(current_input)
            loss = tiny_model.forward_loss(targets[step], output)
            total_loss = total_loss + loss

            # Prepare input for next step (shift by output)
            # New input = [last frame of current input, output]
            # For simplicity, just use output repeated
            current_input = torch.cat(
                [current_input[:, :, 1:, :, :], output],
                dim=2,
            )

        total_loss.backward()
        optimizer.step()

        assert total_loss.item() >= 0


class TestPatchifyUnpatchify:
    """Tests for patchify/unpatchify roundtrip."""

    def test_patchify_unpatchify_roundtrip(self, tiny_model, tiny_model_config):
        """Verify patchify and unpatchify are inverses."""
        # Create a tensor matching output shape
        img = torch.randn(
            2,
            tiny_model_config["in_chans"],
            tiny_model_config["num_out_frames"],
            tiny_model_config["img_size"],
            tiny_model_config["img_size"],
        )

        # Patchify then unpatchify
        patched = tiny_model.patchify(img)
        reconstructed = tiny_model.unpatchify(patched)

        assert torch.allclose(img, reconstructed, atol=1e-6), "Patchify/unpatchify should be lossless"

    def test_patchify_output_shape(self, tiny_model, tiny_model_config):
        """Verify patchify produces correct shape."""
        img = torch.randn(
            2,
            tiny_model_config["in_chans"],
            tiny_model_config["num_out_frames"],
            tiny_model_config["img_size"],
            tiny_model_config["img_size"],
        )

        patched = tiny_model.patchify(img)

        # Expected: (B, L, D) where L = num_patches, D = patch elements
        batch_size = img.shape[0]
        num_patches = tiny_model.patch_embed.num_patches
        patch_size = tiny_model.patch_embed.patch_size[0]
        num_out_frames = tiny_model_config["num_out_frames"]
        in_chans = tiny_model_config["in_chans"]
        patch_dim = num_out_frames * patch_size * patch_size * in_chans

        expected_shape = (batch_size, num_patches, patch_dim)
        assert patched.shape == expected_shape, f"Expected {expected_shape}, got {patched.shape}"


class TestGradientFlow:
    """Tests for gradient flow through the entire model."""

    def test_gradients_flow_through_all_blocks(self, tiny_model, tiny_input, tiny_target):
        """Verify gradients flow through encoder and decoder blocks."""
        tiny_model.train()
        tiny_model.zero_grad()

        output = tiny_model(tiny_input)
        loss = tiny_model.forward_loss(tiny_target, output)
        loss.backward()

        # Check encoder blocks
        for i, block in enumerate(tiny_model.encoder_blocks):
            for name, param in block.named_parameters():
                if param.requires_grad:
                    assert param.grad is not None, f"Encoder block {i} {name} has no gradient"
                    assert param.grad.abs().sum() > 0, f"Encoder block {i} {name} has zero gradient"

        # Check decoder blocks
        for i, block in enumerate(tiny_model.decoder_blocks):
            for name, param in block.named_parameters():
                if param.requires_grad:
                    assert param.grad is not None, f"Decoder block {i} {name} has no gradient"
                    assert param.grad.abs().sum() > 0, f"Decoder block {i} {name} has zero gradient"

    def test_patch_embed_receives_gradients(self, tiny_model, tiny_input, tiny_target):
        """Verify patch embedding layer receives gradients."""
        tiny_model.train()
        tiny_model.zero_grad()

        output = tiny_model(tiny_input)
        loss = tiny_model.forward_loss(tiny_target, output)
        loss.backward()

        for name, param in tiny_model.patch_embed.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Patch embed {name} has no gradient"
                assert param.grad.abs().sum() > 0, f"Patch embed {name} has zero gradient"


class TestBatchSizeVariation:
    """Tests for different batch sizes."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_various_batch_sizes(self, tiny_model, tiny_model_config, batch_size):
        """Verify model works with different batch sizes."""
        inputs = torch.randn(
            batch_size,
            tiny_model_config["in_chans"],
            tiny_model_config["num_frames"],
            tiny_model_config["img_size"],
            tiny_model_config["img_size"],
        )

        with torch.no_grad():
            output = tiny_model(inputs)

        expected_shape = (
            batch_size,
            tiny_model_config["in_chans"],
            tiny_model_config["num_out_frames"],
            tiny_model_config["img_size"],
            tiny_model_config["img_size"],
        )
        assert output.shape == expected_shape
