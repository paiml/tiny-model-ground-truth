"""GPU oracle precision ladder tests (TMGT-GPU).

Validates GPU/precision oracle JSON files:
1. Schema: required keys, types, metadata
2. Precision drift bounds: BF16/FP16 vs float32
3. Precision ladder ordering: BF16 closer to float32 than FP16
4. GPU provenance metadata: CUDA version, GPU name, kernels info

All tests skip gracefully when oracle-gpu/ does not exist.
"""

import re

import pytest
from helpers import (
    ORACLE_GPU_DIR,
    PRECISIONS,
    PROMPTS,
    count_mismatches,
    load_gpu_oracle,
    load_oracle,
)

MODELS = ["smollm-135m", "qwen2-0.5b", "gpt2-124m"]

GPU_ORACLES_EXIST = ORACLE_GPU_DIR.exists()
skip_no_gpu_oracles = pytest.mark.skipif(
    not GPU_ORACLES_EXIST, reason="GPU oracles not generated (oracle-gpu/ missing)"
)

# Parametrization helpers
MODEL_PRECISION_PROMPT = [
    (m, p, pr) for m in MODELS for p in PROMPTS for pr in PRECISIONS
]
MODEL_PROMPT = [(m, p) for m in MODELS for p in PROMPTS]

# Required keys for GPU oracle JSON (superset of CPU oracle keys)
REQUIRED_KEYS = {
    "model",
    "model_slug",
    "prompt",
    "prompt_file",
    "runtime",
    "format",
    "transformers_version",
    "torch_version",
    "device",
    "precision",
    "tokens",
    "text",
    "token_count",
    "max_new_tokens",
    "do_sample",
}

# Precision drift tolerances (token mismatches out of 32)
BF16_MAX_DRIFT = 3  # BF16 vs float32: same tolerance as Int8
FP16_MAX_DRIFT = 5  # FP16 vs float32: same tolerance as Int4
BF16_FP16_MAX_DRIFT = 3  # BF16 vs FP16: higher precision ≤ lower + 1


# ── 4a. Schema validation ────────────────────────────────────────────


@skip_no_gpu_oracles
@pytest.mark.parametrize("model,prompt,precision", MODEL_PRECISION_PROMPT)
def test_gpu_oracle_file_exists(model, prompt, precision):
    """GPU oracle JSON file must exist for every model x prompt x precision."""
    path = ORACLE_GPU_DIR / model / precision / f"{prompt}.json"
    assert path.exists(), f"Missing GPU oracle: {path}"


@skip_no_gpu_oracles
@pytest.mark.parametrize("model,prompt,precision", MODEL_PRECISION_PROMPT)
def test_gpu_oracle_required_keys(model, prompt, precision):
    """GPU oracle files must contain all required keys."""
    data = load_gpu_oracle(model, precision, prompt)
    missing = REQUIRED_KEYS - set(data.keys())
    assert not missing, f"Missing keys: {missing}"


@skip_no_gpu_oracles
@pytest.mark.parametrize("model,prompt,precision", MODEL_PRECISION_PROMPT)
def test_gpu_oracle_tokens_type_and_length(model, prompt, precision):
    """Tokens must be a list of exactly 32 integers."""
    data = load_gpu_oracle(model, precision, prompt)
    tokens = data["tokens"]
    assert isinstance(tokens, list), "tokens must be a list"
    assert len(tokens) == 32, f"Expected 32 tokens, got {len(tokens)}"
    assert all(isinstance(t, int) for t in tokens), "All tokens must be ints"


@skip_no_gpu_oracles
@pytest.mark.parametrize("model,prompt,precision", MODEL_PRECISION_PROMPT)
def test_gpu_oracle_device_and_precision_match(model, prompt, precision):
    """device must be 'cuda' and precision must match directory."""
    data = load_gpu_oracle(model, precision, prompt)
    assert data["device"] == "cuda"
    assert data["precision"] == precision
    assert data["format"] == precision


@skip_no_gpu_oracles
@pytest.mark.parametrize("model,prompt,precision", MODEL_PRECISION_PROMPT)
def test_gpu_oracle_greedy_decoding(model, prompt, precision):
    """Must use greedy decoding (do_sample=False, max_new_tokens=32)."""
    data = load_gpu_oracle(model, precision, prompt)
    assert data["do_sample"] is False, "Must use greedy decoding"
    assert data["max_new_tokens"] == 32


@skip_no_gpu_oracles
@pytest.mark.parametrize("model,prompt,precision", MODEL_PRECISION_PROMPT)
def test_gpu_oracle_token_count_consistent(model, prompt, precision):
    """token_count field must equal len(tokens)."""
    data = load_gpu_oracle(model, precision, prompt)
    assert data["token_count"] == len(data["tokens"])


# ── 4b. Precision drift bounds ───────────────────────────────────────


@skip_no_gpu_oracles
@pytest.mark.parametrize("model,prompt", MODEL_PROMPT)
def test_bf16_drift_vs_float32(model, prompt):
    """BF16 vs float32: at most 3/32 token mismatches."""
    cpu = load_oracle(model, prompt)
    gpu = load_gpu_oracle(model, "bfloat16", prompt)
    drift = count_mismatches(cpu["tokens"], gpu["tokens"])
    assert drift <= BF16_MAX_DRIFT, (
        f"{model}/{prompt}: BF16 drift {drift}/32 exceeds threshold {BF16_MAX_DRIFT}/32"
    )


@skip_no_gpu_oracles
@pytest.mark.parametrize("model,prompt", MODEL_PROMPT)
def test_fp16_drift_vs_float32(model, prompt):
    """FP16 vs float32: at most 5/32 token mismatches."""
    cpu = load_oracle(model, prompt)
    gpu = load_gpu_oracle(model, "float16", prompt)
    drift = count_mismatches(cpu["tokens"], gpu["tokens"])
    assert drift <= FP16_MAX_DRIFT, (
        f"{model}/{prompt}: FP16 drift {drift}/32 exceeds threshold {FP16_MAX_DRIFT}/32"
    )


@skip_no_gpu_oracles
@pytest.mark.parametrize("model,prompt", MODEL_PROMPT)
def test_bf16_vs_fp16_drift(model, prompt):
    """BF16 vs FP16: at most 3/32 token mismatches."""
    bf16 = load_gpu_oracle(model, "bfloat16", prompt)
    fp16 = load_gpu_oracle(model, "float16", prompt)
    drift = count_mismatches(bf16["tokens"], fp16["tokens"])
    assert drift <= BF16_FP16_MAX_DRIFT, (
        f"{model}/{prompt}: BF16↔FP16 drift {drift}/32 exceeds threshold {BF16_FP16_MAX_DRIFT}/32"
    )


# ── 4c. Precision ladder ordering ────────────────────────────────────


@skip_no_gpu_oracles
@pytest.mark.parametrize("model,prompt", MODEL_PROMPT)
def test_precision_ladder_monotonic(model, prompt):
    """BF16 must be closer to float32 than FP16 (monotonic drift).

    mismatches(float32, bf16) <= mismatches(float32, fp16)
    """
    cpu = load_oracle(model, prompt)
    bf16 = load_gpu_oracle(model, "bfloat16", prompt)
    fp16 = load_gpu_oracle(model, "float16", prompt)
    bf16_drift = count_mismatches(cpu["tokens"], bf16["tokens"])
    fp16_drift = count_mismatches(cpu["tokens"], fp16["tokens"])
    assert bf16_drift <= fp16_drift, (
        f"{model}/{prompt}: BF16 drift ({bf16_drift}) > FP16 drift ({fp16_drift}) — "
        f"violates precision ladder monotonicity"
    )


# ── 4d. GPU provenance metadata ──────────────────────────────────────


@skip_no_gpu_oracles
@pytest.mark.parametrize("model,prompt,precision", MODEL_PRECISION_PROMPT)
def test_gpu_cuda_version_present(model, prompt, precision):
    """cuda_version must be present and have valid format (e.g. '12.8')."""
    data = load_gpu_oracle(model, precision, prompt)
    assert "cuda_version" in data, "Missing cuda_version"
    cv = data["cuda_version"]
    assert isinstance(cv, str) and (cv == "N/A" or re.match(r"\d+\.\d+", cv)), (
        f"Invalid cuda_version: {cv}"
    )


@skip_no_gpu_oracles
@pytest.mark.parametrize("model,prompt,precision", MODEL_PRECISION_PROMPT)
def test_gpu_name_present(model, prompt, precision):
    """gpu_name must be present and non-empty."""
    data = load_gpu_oracle(model, precision, prompt)
    assert "gpu_name" in data, "Missing gpu_name"
    assert isinstance(data["gpu_name"], str) and len(data["gpu_name"]) > 0


@skip_no_gpu_oracles
@pytest.mark.parametrize("model,prompt,precision", MODEL_PRECISION_PROMPT)
def test_gpu_torch_version_matches_cpu(model, prompt, precision):
    """torch_version in GPU oracle should match CPU oracle."""
    cpu = load_oracle(model, prompt)
    gpu = load_gpu_oracle(model, precision, prompt)
    assert gpu["torch_version"] == cpu["torch_version"], (
        f"torch_version mismatch: CPU={cpu['torch_version']} GPU={gpu['torch_version']}"
    )


@skip_no_gpu_oracles
@pytest.mark.parametrize("model,prompt,precision", MODEL_PRECISION_PROMPT)
def test_gpu_kernels_metadata(model, prompt, precision):
    """If kernels were used, kernels_version and kernels_enabled must be present."""
    data = load_gpu_oracle(model, precision, prompt)
    if "kernels_enabled" in data:
        assert "kernels_version" in data, "kernels_version missing when kernels_enabled present"
        assert isinstance(data["kernels_enabled"], bool)
        assert isinstance(data["kernels_version"], str)
