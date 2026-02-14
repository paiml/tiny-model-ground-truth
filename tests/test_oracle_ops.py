"""Oracle-ops output schema validation tests (PMAT-003).

Validates that the 5 oracle-ops scripts produced valid JSON with
the required schema. Tests the output files, not the scripts themselves
(which require GPU/models).
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
OPS_DIR = ROOT / "oracle-ops"

OPERATIONS = ["quantize", "finetune", "merge", "convert", "prune"]

# Each operation has a known set of slugs that should have oracles
EXPECTED_SLUGS = {
    "quantize": ["smollm-135m"],
    "finetune": ["smollm-135m"],
    "merge": ["smollm-135m"],
    "convert": ["smollm-135m"],
    "prune": ["smollm-135m"],
}

# Common required keys across all operations (convert uses "format" instead of "method")
COMMON_KEYS = {"tool", "model", "slug"}

# Per-operation required keys (beyond common)
OPERATION_KEYS = {
    "quantize": {"method", "quantize_time_s", "param_bytes"},
    "finetune": {
        "method",
        "rank",
        "alpha",
        "target_modules",
        "trainable_params",
        "total_params",
        "trainable_pct",
        "train_time_s",
        "adapter_files",
    },
    "merge": {"method", "weight", "merge_time_s", "tensor_count"},
    "convert": {"format", "convert_time_s", "file_count", "total_bytes"},
    "prune": {"method", "target_sparsity", "actual_sparsity", "total_params", "prune_time_s"},
}


def _all_op_params():
    """Generate (operation, slug) pairs for parametrized tests."""
    params = []
    for op in OPERATIONS:
        for slug in EXPECTED_SLUGS[op]:
            params.append((op, slug))
    return params


OP_PARAMS = _all_op_params()


@pytest.mark.parametrize("op", OPERATIONS)
def test_ops_directory_exists(op):
    """Each operation must have an oracle-ops subdirectory."""
    assert (OPS_DIR / op).is_dir(), f"Missing directory: oracle-ops/{op}"


@pytest.mark.parametrize("op,slug", OP_PARAMS)
def test_ops_file_exists(op, slug):
    """Oracle-ops JSON file must exist for each expected model."""
    path = OPS_DIR / op / f"{slug}.json"
    assert path.exists(), f"Missing oracle-ops file: {path}"


@pytest.mark.parametrize("op,slug", OP_PARAMS)
def test_ops_valid_json(op, slug):
    """Oracle-ops files must be valid JSON."""
    path = OPS_DIR / op / f"{slug}.json"
    data = json.loads(path.read_text())
    # Top-level is a list with one entry
    assert isinstance(data, list), f"Expected list, got {type(data).__name__}"
    assert len(data) >= 1, "Expected at least one entry"
    assert isinstance(data[0], dict), f"Expected dict entry, got {type(data[0]).__name__}"


@pytest.mark.parametrize("op,slug", OP_PARAMS)
def test_ops_common_keys(op, slug):
    """All oracle-ops entries must have common metadata keys."""
    path = OPS_DIR / op / f"{slug}.json"
    data = json.loads(path.read_text())
    entry = data[0]
    missing = COMMON_KEYS - set(entry.keys())
    assert not missing, f"Missing common keys in {op}/{slug}: {missing}"


@pytest.mark.parametrize("op,slug", OP_PARAMS)
def test_ops_operation_specific_keys(op, slug):
    """Each operation must have its required specific keys."""
    path = OPS_DIR / op / f"{slug}.json"
    data = json.loads(path.read_text())
    entry = data[0]
    required = OPERATION_KEYS[op]
    missing = required - set(entry.keys())
    assert not missing, f"Missing keys in {op}/{slug}: {missing}"


@pytest.mark.parametrize("op,slug", OP_PARAMS)
def test_ops_slug_matches(op, slug):
    """slug field must match the filename."""
    path = OPS_DIR / op / f"{slug}.json"
    data = json.loads(path.read_text())
    assert data[0]["slug"] == slug


@pytest.mark.parametrize("op,slug", OP_PARAMS)
def test_ops_timing_positive(op, slug):
    """Timing fields must be positive numbers."""
    path = OPS_DIR / op / f"{slug}.json"
    data = json.loads(path.read_text())
    entry = data[0]
    time_keys = [k for k in entry if k.endswith("_time_s")]
    assert len(time_keys) >= 1, f"No timing field found in {op}/{slug}"
    for k in time_keys:
        assert isinstance(entry[k], (int, float)), f"{k} must be numeric"
        assert entry[k] > 0, f"{k} must be positive, got {entry[k]}"


@pytest.mark.parametrize("op,slug", [p for p in OP_PARAMS if p[0] != "convert"])
def test_ops_method_is_nonempty_string(op, slug):
    """method field must be a non-empty string (convert uses 'format' instead)."""
    path = OPS_DIR / op / f"{slug}.json"
    data = json.loads(path.read_text())
    method = data[0]["method"]
    assert isinstance(method, str) and len(method) > 0


@pytest.mark.parametrize("op,slug", OP_PARAMS)
def test_ops_model_is_hf_id(op, slug):
    """model field must look like a HuggingFace model ID (contains /)."""
    path = OPS_DIR / op / f"{slug}.json"
    data = json.loads(path.read_text())
    model = data[0]["model"]
    assert "/" in model, f"model should be HF ID (org/name), got: {model}"


# ── Finetune-specific tests ─────────────────────────────────────


@pytest.mark.parametrize("slug", EXPECTED_SLUGS["finetune"])
def test_finetune_trainable_pct_reasonable(slug):
    """LoRA trainable params should be <5% of total."""
    path = OPS_DIR / "finetune" / f"{slug}.json"
    data = json.loads(path.read_text())
    entry = data[0]
    assert 0 < entry["trainable_pct"] < 5.0
    assert entry["trainable_params"] < entry["total_params"]


@pytest.mark.parametrize("slug", EXPECTED_SLUGS["finetune"])
def test_finetune_target_modules_are_strings(slug):
    """target_modules must be a list of strings."""
    path = OPS_DIR / "finetune" / f"{slug}.json"
    data = json.loads(path.read_text())
    modules = data[0]["target_modules"]
    assert isinstance(modules, list)
    assert all(isinstance(m, str) for m in modules)


# ── Prune-specific tests ────────────────────────────────────────


@pytest.mark.parametrize("slug", EXPECTED_SLUGS["prune"])
def test_prune_sparsity_close_to_target(slug):
    """actual_sparsity should be within 5% of target_sparsity."""
    path = OPS_DIR / "prune" / f"{slug}.json"
    data = json.loads(path.read_text())
    entry = data[0]
    assert abs(entry["actual_sparsity"] - entry["target_sparsity"]) < 0.05


# ── Convert-specific tests ──────────────────────────────────────


@pytest.mark.parametrize("slug", EXPECTED_SLUGS["convert"])
def test_convert_format_valid(slug):
    """format must be a known conversion target."""
    path = OPS_DIR / "convert" / f"{slug}.json"
    data = json.loads(path.read_text())
    assert data[0]["format"] in {"safetensors", "onnx", "gguf", "coreml"}


@pytest.mark.parametrize("slug", EXPECTED_SLUGS["convert"])
def test_convert_bytes_positive(slug):
    """total_bytes must be positive."""
    path = OPS_DIR / "convert" / f"{slug}.json"
    data = json.loads(path.read_text())
    assert data[0]["total_bytes"] > 0
