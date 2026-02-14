"""Shared helpers for parity tests. Importable by all test_*.py files."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
ORACLE_DIR = ROOT / "oracle"
MODELS_DIR = ROOT / "models"

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


def count_mismatches(a: list[int], b: list[int]) -> int:
    min_len = min(len(a), len(b))
    m = sum(1 for i in range(min_len) if a[i] != b[i])
    return m + abs(len(a) - len(b))
