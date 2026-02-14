"""Oracle drift detection tests (PMAT-001).

Validates that oracle JSON files:
1. Exist for every model x prompt combination
2. Have the required schema (keys, types, token count)
3. Have stable SHA-256 hashes (detect silent HF/torch regressions)
4. Record provenance metadata (transformers_version, torch_version, etc.)
"""

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
ORACLE_DIR = ROOT / "oracle"

MODELS = ["smollm-135m", "qwen2-0.5b", "gpt2-124m"]
PROMPTS = ["arithmetic", "code", "completion", "greeting"]

REQUIRED_KEYS = {
    "model",
    "model_slug",
    "prompt",
    "prompt_file",
    "runtime",
    "format",
    "transformers_version",
    "torch_version",
    "tokens",
    "text",
    "token_count",
    "max_new_tokens",
    "do_sample",
}

# SHA-256 hashes of current oracle files. If gen_oracle.py output changes
# (e.g. transformers upgrade), these must be deliberately updated.
ORACLE_HASHES = {
    "smollm-135m/arithmetic": "4fc86efa24fc56a0",
    "smollm-135m/code": "c474ac828bc1b2b0",
    "smollm-135m/completion": "2856d78a5ebb0856",
    "smollm-135m/greeting": "1c317c01e73b3a44",
    "qwen2-0.5b/arithmetic": "9f38d4d8c6236a8c",
    "qwen2-0.5b/code": "991b1c210a7c6229",
    "qwen2-0.5b/completion": "dc9564792c11e5dc",
    "qwen2-0.5b/greeting": "b1ec6952c1dcfb39",
    "gpt2-124m/arithmetic": "067538c08a45ba78",
    "gpt2-124m/code": "cd2de5af8a634f22",
    "gpt2-124m/completion": "5a3b3001c14d429f",
    "gpt2-124m/greeting": "accf174793543dd7",
}


MODEL_PROMPT_PARAMS = [(m, p) for m in MODELS for p in PROMPTS]


@pytest.mark.parametrize("model,prompt", MODEL_PROMPT_PARAMS)
def test_oracle_file_exists(model, prompt):
    """Every model x prompt combination must have an oracle JSON file."""
    path = ORACLE_DIR / model / f"{prompt}.json"
    assert path.exists(), f"Missing oracle: {path}"


@pytest.mark.parametrize("model,prompt", MODEL_PROMPT_PARAMS)
def test_oracle_valid_json(model, prompt):
    """Oracle files must be valid JSON."""
    path = ORACLE_DIR / model / f"{prompt}.json"
    data = json.loads(path.read_text())
    assert isinstance(data, dict)


@pytest.mark.parametrize("model,prompt", MODEL_PROMPT_PARAMS)
def test_oracle_required_keys(model, prompt):
    """Oracle files must contain all required keys."""
    path = ORACLE_DIR / model / f"{prompt}.json"
    data = json.loads(path.read_text())
    missing = REQUIRED_KEYS - set(data.keys())
    assert not missing, f"Missing keys: {missing}"


@pytest.mark.parametrize("model,prompt", MODEL_PROMPT_PARAMS)
def test_oracle_tokens_type_and_length(model, prompt):
    """Tokens must be a list of exactly 32 integers."""
    path = ORACLE_DIR / model / f"{prompt}.json"
    data = json.loads(path.read_text())
    tokens = data["tokens"]
    assert isinstance(tokens, list), "tokens must be a list"
    assert len(tokens) == 32, f"Expected 32 tokens, got {len(tokens)}"
    assert all(isinstance(t, int) for t in tokens), "All tokens must be ints"


@pytest.mark.parametrize("model,prompt", MODEL_PROMPT_PARAMS)
def test_oracle_text_is_nonempty_string(model, prompt):
    """Generated text must be a non-empty string."""
    path = ORACLE_DIR / model / f"{prompt}.json"
    data = json.loads(path.read_text())
    assert isinstance(data["text"], str)
    assert len(data["text"]) > 0


@pytest.mark.parametrize("model,prompt", MODEL_PROMPT_PARAMS)
def test_oracle_provenance_metadata(model, prompt):
    """Oracle must record transformers version, torch version, and runtime settings."""
    path = ORACLE_DIR / model / f"{prompt}.json"
    data = json.loads(path.read_text())
    assert data["runtime"] == "transformers"
    assert data["format"] == "float32"
    assert data["do_sample"] is False, "Must use greedy decoding"
    assert data["max_new_tokens"] == 32
    assert "." in data["transformers_version"], "Must record transformers version"
    assert "." in data["torch_version"], "Must record torch version"


@pytest.mark.parametrize("model,prompt", MODEL_PROMPT_PARAMS)
def test_oracle_model_slug_matches_directory(model, prompt):
    """model_slug field must match the directory name."""
    path = ORACLE_DIR / model / f"{prompt}.json"
    data = json.loads(path.read_text())
    assert data["model_slug"] == model


@pytest.mark.parametrize("model,prompt", MODEL_PROMPT_PARAMS)
def test_oracle_hash_stability(model, prompt):
    """Oracle file SHA-256 must match pinned hash (detect silent regressions)."""
    path = ORACLE_DIR / model / f"{prompt}.json"
    content = path.read_bytes()
    actual = hashlib.sha256(content).hexdigest()[:16]
    key = f"{model}/{prompt}"
    expected = ORACLE_HASHES[key]
    assert actual == expected, (
        f"Oracle hash changed for {key}: {expected} -> {actual}. "
        f"If intentional (transformers upgrade), update ORACLE_HASHES in test_oracle_drift.py"
    )


@pytest.mark.parametrize("model,prompt", MODEL_PROMPT_PARAMS)
def test_oracle_token_count_consistent(model, prompt):
    """token_count field must equal len(tokens)."""
    path = ORACLE_DIR / model / f"{prompt}.json"
    data = json.loads(path.read_text())
    assert data["token_count"] == len(data["tokens"])
