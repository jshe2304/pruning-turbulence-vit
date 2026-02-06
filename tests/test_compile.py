"""Verify that all scripts and source modules compile without errors."""

import py_compile
import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# ── Scripts ──────────────────────────────────────────────────────────────────

SCRIPTS = sorted((ROOT / "scripts").glob("*.py"))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_compiles(script):
    py_compile.compile(str(script), doraise=True)


# ── Source modules ───────────────────────────────────────────────────────────

SRC_MODULES = [
    "src.models",
    "src.models.simple_vit",
    "src.models.vit",
    "src.models.modules.attention",
    "src.models.modules.mlp",
    "src.models.modules.embeddings",
    "src.models.modules.positional_encodings",
    "src.models.modules.conv",
    "src.data.datasets",
    "src.training.train",
    "src.training.prune_unstructured",
    "src.training.prune_attention_heads",
    "src.training.distill",
    "src.training.utils.compute_loss",
    "src.training.utils.importance_scores",
    "src.training.utils.structured_pruning",
    "src.inference.make_inference",
    "src.inference.short_analysis",
    "src.inference.long_analysis",
    "src.inference.make_gif",
    "src.inference.utils.io_utils",
    "src.inference.utils.rollout",
    "src.inference.utils.short_metrics",
    "src.inference.utils.long_metrics",
    "src.inference.utils.plot_config",
]


@pytest.mark.parametrize("module", SRC_MODULES)
def test_src_module_imports(module):
    importlib.import_module(module)
