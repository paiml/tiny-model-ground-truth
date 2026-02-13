"""Shared helpers for parity tests. Importable by all test_*.py files."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
ORACLE_DIR = ROOT / "oracle"
MODELS_DIR = ROOT / "models"

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
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT after 120s"
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
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT after 120s"
    except FileNotFoundError:
        return None, "apr not found in PATH"

    if proc.returncode != 0:
        return None, f"apr eval failed (exit {proc.returncode}): {proc.stderr.strip()}"
    try:
        return json.loads(proc.stdout), None
    except json.JSONDecodeError:
        return None, f"invalid JSON: {proc.stdout[:200]}"


def count_mismatches(a: list[int], b: list[int]) -> int:
    min_len = min(len(a), len(b))
    m = sum(1 for i in range(min_len) if a[i] != b[i])
    return m + abs(len(a) - len(b))
