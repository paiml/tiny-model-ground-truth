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


def pytest_collection_modifyitems(config, items):
    """Auto-skip requires_apr tests when apr or models are unavailable."""
    has_apr = apr_available()
    has_models = models_available()

    skip_apr = pytest.mark.skip(reason="apr CLI not in PATH")
    skip_models = pytest.mark.skip(reason="model files not found (run make convert)")

    for item in items:
        if "requires_apr" in {m.name for m in item.iter_markers()}:
            if not has_apr:
                item.add_marker(skip_apr)
            elif not has_models:
                item.add_marker(skip_models)
