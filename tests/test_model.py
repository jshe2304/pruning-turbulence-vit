"""
Smoke tests for model construction and forward pass.

Catches: shape mismatches, bad config keys, broken init, wrong output dims.
Run before submitting any training job.
"""

import pytest
import torch
import torch.nn.utils.prune as prune

from src.models.vit import ViT
from src.models.encoder_decoder_vit import EncoderDecoderViT


# ---------------------------------------------------------------------------
# ViT (encoder-only)
# ---------------------------------------------------------------------------

class TestViT:

    def test_forward_shape(self, simple_vit_config, dummy_input_simple_vit):
        model = ViT(**simple_vit_config)
        out = model(dummy_input_simple_vit)
        B = dummy_input_simple_vit.shape[0]
        C = 2
        H = W = simple_vit_config["img_size"]
        assert out.shape == (B, C, 1, H, W)

    def test_deterministic_with_seed(self, simple_vit_config, dummy_input_simple_vit):
        torch.manual_seed(0)
        m1 = ViT(**simple_vit_config)
        out1 = m1(dummy_input_simple_vit)

        torch.manual_seed(0)
        m2 = ViT(**simple_vit_config)
        out2 = m2(dummy_input_simple_vit)

        assert torch.allclose(out1, out2)

    def test_backward(self, simple_vit_config, dummy_input_simple_vit):
        model = ViT(**simple_vit_config)
        out = model(dummy_input_simple_vit)
        loss = out.sum()
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_parameter_count_positive(self, simple_vit_config):
        model = ViT(**simple_vit_config)
        assert model.n_parameters() > 0

    def test_save_load_roundtrip(self, simple_vit_config, dummy_input_simple_vit, tmp_path):
        model = ViT(**simple_vit_config)
        out_before = model(dummy_input_simple_vit)

        path = tmp_path / "ckpt.tar"
        torch.save({"model_state": model.state_dict()}, path)

        model2 = ViT(**simple_vit_config)
        ckpt = torch.load(path, weights_only=True)
        model2.load_state_dict(ckpt["model_state"])
        out_after = model2(dummy_input_simple_vit)

        assert torch.allclose(out_before, out_after)

    def test_load_ddp_checkpoint(self, simple_vit_config, dummy_input_simple_vit, tmp_path):
        """Checkpoints saved under DDP have a 'module.' prefix — loading should strip it."""
        model = ViT(**simple_vit_config)
        state = {"module." + k: v for k, v in model.state_dict().items()}

        path = tmp_path / "ddp_ckpt.tar"
        torch.save({"model_state": state}, path)

        model2 = ViT(**simple_vit_config)
        ckpt = torch.load(path, weights_only=True)
        model2.load_state_dict(ckpt["model_state"])
        out = model2(dummy_input_simple_vit)
        assert out.shape[0] == dummy_input_simple_vit.shape[0]

    def test_load_pruned_checkpoint(self, simple_vit_config, dummy_input_simple_vit, tmp_path):
        """Checkpoints with pruning masks should reload correctly."""
        model = ViT(**simple_vit_config)
        # Apply pruning to a weight
        prune.l1_unstructured(model.blocks[0].attn.qkv, name="weight", amount=0.3)
        out_before = model(dummy_input_simple_vit)

        path = tmp_path / "pruned_ckpt.tar"
        torch.save({"model_state": model.state_dict()}, path)

        model2 = ViT(**simple_vit_config)
        ckpt = torch.load(path, weights_only=True)
        model2.load_state_dict(ckpt["model_state"])
        out_after = model2(dummy_input_simple_vit)

        assert torch.allclose(out_before, out_after)

    def test_get_weights_not_empty(self, simple_vit_config):
        model = ViT(**simple_vit_config)
        weights = model.get_weights()
        assert len(weights) > 0
        for module, name in weights:
            assert hasattr(module, name)


# ---------------------------------------------------------------------------
# Encoder-decoder ViT
# ---------------------------------------------------------------------------

class TestEncoderDecoderViT:

    def test_forward_shape(self, vit_config, dummy_input_vit):
        model = EncoderDecoderViT(**vit_config)
        out = model(dummy_input_vit)
        B = dummy_input_vit.shape[0]
        C = vit_config["in_chans"]
        H = W = vit_config["img_size"]
        num_out = vit_config["num_out_frames"]
        assert out.shape == (B, C, num_out, H, W)

    def test_backward(self, vit_config, dummy_input_vit):
        model = EncoderDecoderViT(**vit_config)
        out = model(dummy_input_vit)
        loss = out.sum()
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_save_load_roundtrip(self, vit_config, dummy_input_vit, tmp_path):
        model = EncoderDecoderViT(**vit_config)
        out_before = model(dummy_input_vit)

        path = tmp_path / "ckpt.tar"
        torch.save({"model_state": model.state_dict()}, path)

        model2 = EncoderDecoderViT(**vit_config)
        ckpt = torch.load(path, weights_only=True)
        model2.load_state_dict(ckpt["model_state"])
        out_after = model2(dummy_input_vit)

        assert torch.allclose(out_before, out_after)

    def test_patchify_unpatchify_roundtrip(self, vit_config):
        model = EncoderDecoderViT(**vit_config)
        B, C = 2, vit_config["in_chans"]
        T = vit_config["num_out_frames"]
        H = W = vit_config["img_size"]
        imgs = torch.randn(B, C, T, H, W)
        assert torch.allclose(model.unpatchify(model.patchify(imgs)), imgs)

    def test_forward_loss_shape(self, vit_config):
        model = EncoderDecoderViT(**vit_config)
        B, C = 2, vit_config["in_chans"]
        T = vit_config["num_out_frames"]
        H = W = vit_config["img_size"]
        img = torch.randn(B, C, T, H, W)
        pred = torch.randn(B, C, T, H, W)
        loss = model.forward_loss(img, pred)
        assert loss.shape == ()  # scalar
        assert loss.item() > 0
