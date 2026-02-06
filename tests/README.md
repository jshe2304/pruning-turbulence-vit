# Tests

Run all tests from the project root:

```bash
python -m pytest tests/ -v
```

## test_compile.py

Verifies all code parses and imports without errors.

### test_script_compiles
- Checks each `scripts/*.py` file compiles via `py_compile`

### test_src_module_imports
- Checks every module under `src/` can be imported via `importlib`

## test_config.py

Validates TOML config files before cluster job submission.

### TestTrainConfigs
- `test_has_required_sections` -- `[model]` and `[training]` sections exist
- `test_model_section_keys` -- `img_size % patch_size == 0` and `embed_dim % num_heads == 0`
- `test_training_section_keys` -- required keys (`lr`, `epochs`, `batch_size`, `output_dir`) are present
- `test_batch_size_positive_power_of_two` -- batch size is a positive power of 2
- `test_dataset_num_frames_matches_model` -- dataset `num_frames` matches model `num_frames`

### TestPruneConfigs
- `test_has_pruning_section` -- `[pruning]` or `[prune]` section exists

## test_model.py

Smoke tests for model construction, forward/backward passes, and checkpointing.

### TestSimpleViT
- `test_forward_shape` -- output shape matches expected `(B, C, 1, H, W)`
- `test_deterministic_with_seed` -- same seed produces identical outputs
- `test_backward` -- gradients are computed for all trainable parameters
- `test_parameter_count_positive` -- model has trainable parameters
- `test_save_load_roundtrip` -- checkpoint save/load reproduces outputs
- `test_load_ddp_checkpoint` -- DDP `module.` prefix is stripped on load
- `test_load_pruned_checkpoint` -- pruning masks survive save/load
- `test_get_weights_not_empty` -- `get_weights()` returns valid entries

### TestViT
- `test_forward_shape` -- output shape matches expected `(B, C, num_out_frames, H, W)`
- `test_backward` -- gradients are computed for all trainable parameters
- `test_save_load_roundtrip` -- checkpoint save/load reproduces outputs
- `test_patchify_unpatchify_roundtrip` -- `patchify` then `unpatchify` recovers original input
- `test_forward_loss_shape` -- `forward_loss` returns a positive scalar
