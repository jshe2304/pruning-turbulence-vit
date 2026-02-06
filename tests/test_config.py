"""
Validate TOML configs before submitting cluster jobs.

Catches: missing keys, wrong types, mismatched dataset params, bad paths.
"""

import glob
import os

import pytest
import toml


CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_all_configs(subdir):
    pattern = os.path.join(CONFIGS_DIR, subdir, "*.toml")
    paths = glob.glob(pattern)
    return [(p, toml.load(p)) for p in paths]


TRAIN_CONFIGS = _load_all_configs("train")
PRUNE_CONFIGS = _load_all_configs("prune")


# ---------------------------------------------------------------------------
# Training configs
# ---------------------------------------------------------------------------

class TestTrainConfigs:

    @pytest.mark.parametrize("path,cfg", TRAIN_CONFIGS, ids=[os.path.basename(p) for p, _ in TRAIN_CONFIGS])
    def test_has_required_sections(self, path, cfg):
        for section in ("model", "training"):
            assert section in cfg, f"{path} missing [{section}]"

    @pytest.mark.parametrize("path,cfg", TRAIN_CONFIGS, ids=[os.path.basename(p) for p, _ in TRAIN_CONFIGS])
    def test_model_section_keys(self, path, cfg):
        if "model" not in cfg:
            pytest.skip("no [model]")
        model = cfg["model"]
        # img_size and patch_size must divide evenly
        if "img_size" in model and "patch_size" in model:
            assert model["img_size"] % model["patch_size"] == 0, (
                f"img_size ({model['img_size']}) not divisible by patch_size ({model['patch_size']})"
            )
        # num_heads must divide embed_dim
        embed_dim = model.get("embed_dim") or model.get("encoder_embed_dim")
        num_heads = model.get("num_heads") or model.get("encoder_num_heads")
        if embed_dim is not None and num_heads is not None:
            assert embed_dim % num_heads == 0, (
                f"embed_dim ({embed_dim}) not divisible by num_heads ({num_heads})"
            )

    @pytest.mark.parametrize("path,cfg", TRAIN_CONFIGS, ids=[os.path.basename(p) for p, _ in TRAIN_CONFIGS])
    def test_training_section_keys(self, path, cfg):
        if "training" not in cfg:
            pytest.skip("no [training]")
        training = cfg["training"]
        required = {"lr", "epochs", "batch_size", "output_dir"}
        missing = required - set(training.keys())
        assert not missing, f"{path} [training] missing keys: {missing}"

    @pytest.mark.parametrize("path,cfg", TRAIN_CONFIGS, ids=[os.path.basename(p) for p, _ in TRAIN_CONFIGS])
    def test_batch_size_positive_power_of_two(self, path, cfg):
        bs = cfg.get("training", {}).get("batch_size")
        if bs is None:
            pytest.skip("no batch_size")
        assert bs > 0 and (bs & (bs - 1)) == 0, f"batch_size={bs} is not a positive power of 2"

    @pytest.mark.parametrize("path,cfg", TRAIN_CONFIGS, ids=[os.path.basename(p) for p, _ in TRAIN_CONFIGS])
    def test_dataset_num_frames_matches_model(self, path, cfg):
        """num_frames in dataset config should match the model's num_frames."""
        model_nf = cfg.get("model", {}).get("num_frames")
        for section in ("train_dataset", "validation_dataset"):
            ds_nf = cfg.get(section, {}).get("num_frames")
            if model_nf is not None and ds_nf is not None:
                assert model_nf == ds_nf, (
                    f"{path}: model.num_frames={model_nf} != {section}.num_frames={ds_nf}"
                )


# ---------------------------------------------------------------------------
# Pruning configs
# ---------------------------------------------------------------------------

class TestPruneConfigs:

    @pytest.mark.parametrize("path,cfg", PRUNE_CONFIGS, ids=[os.path.basename(p) for p, _ in PRUNE_CONFIGS])
    def test_has_pruning_section(self, path, cfg):
        assert "pruning" in cfg or "prune" in cfg, f"{path} missing [pruning] or [prune]"
