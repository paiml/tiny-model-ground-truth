"""Shared fixtures for parity tests.

These tests wrap the parity_check.py logic into pytest-discoverable
test functions. Each test maps to a falsifiable claim in CLAIMS.md.

conftest.py provides fixtures — for importable helpers, use:
  from helpers import MODELS, load_oracle, apr_run_json, ...
"""

import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"


def apr_available() -> bool:
    return shutil.which("apr") is not None


def models_available() -> bool:
    model_files = [
        "smollm-135m-int4.apr", "smollm-135m-int8.apr",
        "qwen2-0.5b-int4.apr", "qwen2-0.5b-int8.apr",
        "gpt2-124m-int4.apr", "gpt2-124m-int8.apr",
    ]
    return all((MODELS_DIR / f).exists() for f in model_files)


def llamacpp_available() -> bool:
    return shutil.which("llama-completion") is not None


def llamacpp_models_available() -> bool:
    model_files = [
        "smollm-135m-q8_0.gguf", "smollm-135m-q4_0.gguf",
        "qwen2-0.5b-q8_0.gguf", "qwen2-0.5b-q4_0.gguf",
        "gpt2-124m-q8_0.gguf", "gpt2-124m-q4_0.gguf",
    ]
    return all((MODELS_DIR / f).exists() for f in model_files)


def pytest_collection_modifyitems(config, items):
    """Auto-skip requires_apr/requires_llamacpp tests when deps are unavailable."""
    has_apr = apr_available()
    has_models = models_available()
    has_llamacpp = llamacpp_available()
    has_llamacpp_models = llamacpp_models_available()

    skip_apr = pytest.mark.skip(reason="apr CLI not in PATH")
    skip_models = pytest.mark.skip(reason="model files not found (run make convert)")
    skip_llamacpp = pytest.mark.skip(reason="llama-completion not in PATH")
    skip_llamacpp_models = pytest.mark.skip(
        reason="llama.cpp native GGUF models not found (run make convert-llamacpp)"
    )

    for item in items:
        markers = {m.name for m in item.iter_markers()}
        if "requires_apr" in markers:
            if not has_apr:
                item.add_marker(skip_apr)
            elif not has_models:
                item.add_marker(skip_models)
        if "requires_llamacpp" in markers:
            if not has_llamacpp:
                item.add_marker(skip_llamacpp)
            elif not has_llamacpp_models:
                item.add_marker(skip_llamacpp_models)
