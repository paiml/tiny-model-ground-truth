"""Shared helpers for parity tests. Importable by all test_*.py files."""

import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
ORACLE_DIR = ROOT / "oracle"
ORACLE_GPU_DIR = ROOT / "oracle-gpu"
MODELS_DIR = ROOT / "models"

PRECISIONS = ["bfloat16", "float16"]

MODEL_METADATA = {
    "smollm-135m": {
        "architecture": "llama",
        "layers": 30,
        "heads": 9,
        "kv_heads": 3,
        "hidden_dim": 576,
        "vocab_size": 49152,
        "hf_id": "HuggingFaceTB/SmolLM-135M",
    },
    "qwen2-0.5b": {
        "architecture": "qwen2",
        "layers": 24,
        "heads": 14,
        "kv_heads": 2,
        "hidden_dim": 896,
        "vocab_size": 151936,
        "hf_id": "Qwen/Qwen2-0.5B",
    },
    "gpt2-124m": {
        "architecture": "gpt2",
        "layers": 12,
        "heads": 12,
        "kv_heads": 12,
        "hidden_dim": 768,
        "vocab_size": 50257,
        "hf_id": "openai-community/gpt2",
    },
}

MODELS = {
    "smollm-135m": {
        "int4": MODELS_DIR / "smollm-135m-int4.apr",
        "int8": MODELS_DIR / "smollm-135m-int8.apr",
        "gguf": MODELS_DIR / "smollm-135m-int4.gguf",
        "ppl_ceiling": 20.0,
    },
    "qwen2-0.5b": {
        "int4": MODELS_DIR / "qwen2-0.5b-int4.apr",
        "int8": MODELS_DIR / "qwen2-0.5b-int8.apr",
        "gguf": MODELS_DIR / "qwen2-0.5b-int4.gguf",
        "ppl_ceiling": 15.0,
    },
    "gpt2-124m": {
        "int4": MODELS_DIR / "gpt2-124m-int4.apr",
        "int8": MODELS_DIR / "gpt2-124m-int8.apr",
        "gguf": MODELS_DIR / "gpt2-124m-int4.gguf",
        "ppl_ceiling": 30.0,
    },
}

LLAMACPP_BIN = shutil.which("llama-completion") or ""

LLAMACPP_MODELS = {
    "smollm-135m": {
        "f16": MODELS_DIR / "smollm-135m-f16.gguf",
        "q4_0": MODELS_DIR / "smollm-135m-q4_0.gguf",
        "q8_0": MODELS_DIR / "smollm-135m-q8_0.gguf",
    },
    "qwen2-0.5b": {
        "f16": MODELS_DIR / "qwen2-0.5b-f16.gguf",
        "q4_0": MODELS_DIR / "qwen2-0.5b-q4_0.gguf",
        "q8_0": MODELS_DIR / "qwen2-0.5b-q8_0.gguf",
    },
    "gpt2-124m": {
        "f16": MODELS_DIR / "gpt2-124m-f16.gguf",
        "q4_0": MODELS_DIR / "gpt2-124m-q4_0.gguf",
        "q8_0": MODELS_DIR / "gpt2-124m-q8_0.gguf",
    },
}

PROMPTS = ["arithmetic", "code", "completion", "greeting"]

MODEL_PROMPT_PARAMS = [
    (slug, prompt) for slug in MODELS for prompt in PROMPTS
]

MODEL_PARAMS = list(MODELS.keys())


def load_oracle(slug: str, prompt_name: str) -> dict:
    path = ORACLE_DIR / slug / f"{prompt_name}.json"
    return json.loads(path.read_text())


def apr_run_json(model_path: str, prompt: str) -> tuple[dict | None, str | None]:
    try:
        proc = subprocess.run(
            ["apr", "run", str(model_path), "-p", prompt, "-n", "32", "--json"],
            capture_output=True, text=True, timeout=55,
        )
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT after 55s"
    except FileNotFoundError:
        return None, "apr not found in PATH"

    if proc.returncode != 0:
        return None, f"apr run failed (exit {proc.returncode}): {proc.stderr.strip()}"
    try:
        return json.loads(proc.stdout), None
    except json.JSONDecodeError:
        return None, f"invalid JSON: {proc.stdout[:200]}"


def apr_eval_json(model_path: str) -> tuple[dict | None, str | None]:
    try:
        proc = subprocess.run(
            ["apr", "eval", str(model_path), "--threshold", "50.0", "--json"],
            capture_output=True, text=True, timeout=55,
        )
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT after 55s"
    except FileNotFoundError:
        return None, "apr not found in PATH"

    if proc.returncode != 0:
        return None, f"apr eval failed (exit {proc.returncode}): {proc.stderr.strip()}"
    try:
        return json.loads(proc.stdout), None
    except json.JSONDecodeError:
        return None, f"invalid JSON: {proc.stdout[:200]}"


MODEL_QUANT_PARAMS = [
    (slug, quant) for slug in MODELS for quant in ["int4", "int8"]
]


def run_apr(args: list[str]) -> tuple[str, str, int]:
    """Run apr CLI and return (stdout, stderr, exit_code)."""
    try:
        proc = subprocess.run(
            ["apr", *args],
            capture_output=True,
            text=True,
            timeout=55,
        )
        return proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT after 55s", 1
    except FileNotFoundError:
        return "", "apr not found in PATH", 127


def apr_cmd_json(args: list[str]) -> tuple[dict | None, str | None]:
    """Run any apr subcommand with --json and return parsed JSON."""
    stdout, stderr, code = run_apr(args)
    if code != 0:
        return None, f"apr {args[0]} failed (exit {code}): {stderr.strip()}"
    try:
        return json.loads(stdout), None
    except json.JSONDecodeError:
        return None, f"apr {args[0]} returned invalid JSON: {stdout[:200]}"


def load_gpu_oracle(slug: str, precision: str, prompt_name: str) -> dict:
    """Load a GPU/precision oracle JSON file."""
    path = ORACLE_GPU_DIR / slug / precision / f"{prompt_name}.json"
    return json.loads(path.read_text())


def count_mismatches(a: list[int], b: list[int]) -> int:
    min_len = min(len(a), len(b))
    m = sum(1 for i in range(min_len) if a[i] != b[i])
    return m + abs(len(a) - len(b))


def llamacpp_run(gguf_path: str, prompt: str, n_tokens: int = 32) -> tuple[dict | None, str | None]:
    """Run llama-completion greedy inference, return {"text": ...} or None on failure."""
    cli = LLAMACPP_BIN
    if not cli:
        return None, "llama-completion not found in PATH"
    try:
        proc = subprocess.run(
            [
                cli, "-m", str(gguf_path),
                "-p", prompt,
                "-n", str(n_tokens),
                "--temp", "0", "--top-k", "1",
                "--no-display-prompt",
                "-s", "42",
            ],
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT after 120s"
    except FileNotFoundError:
        return None, "llama-completion not found in PATH"
    if proc.returncode != 0:
        return None, f"llama-completion failed (exit {proc.returncode}): {proc.stderr.strip()[:300]}"
    text = proc.stdout.strip()
    return {"text": text}, None


def count_char_mismatches(a: str, b: str) -> int:
    """Count character-level mismatches between two strings."""
    min_len = min(len(a), len(b))
    m = sum(1 for i in range(min_len) if a[i] != b[i])
    return m + abs(len(a) - len(b))
