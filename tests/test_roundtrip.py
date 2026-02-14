"""Claim 5: Format Roundtrip Losslessness — APR -> GGUF -> APR.

APR export to GGUF and reimport must produce token-identical outputs.
Falsification: Any token mismatch after roundtrip. Threshold: 0 tolerance.
Sample size: n = 6 (3 models x 2 prompts).
"""

import subprocess
from pathlib import Path

import pytest
from helpers import (
    MODEL_PARAMS,
    MODELS,
    MODELS_DIR,
    apr_run_json,
    count_mismatches,
    load_oracle,
)

ROUNDTRIP_PROMPTS = ["completion", "arithmetic"]

skip_no_apr = pytest.importorskip("shutil").which("apr") is not None
pytestmark = [
    pytest.mark.roundtrip,
    pytest.mark.skipif(not skip_no_apr, reason="apr CLI not in PATH"),
]


@pytest.mark.parametrize("slug", MODEL_PARAMS)
@pytest.mark.parametrize("prompt_name", ROUNDTRIP_PROMPTS)
def test_apr_gguf_apr_roundtrip_tokens_identical(slug: str, prompt_name: str):
    """APR -> GGUF -> reimport APR produces identical tokens (Claim 5)."""
    reimported = MODELS_DIR / f"{slug}-roundtrip-tmp.apr"

    try:
        gguf_path = MODELS[slug]["gguf"]
        if not Path(gguf_path).exists():
            pytest.skip(f"GGUF file {gguf_path} not found")

        proc = subprocess.run(
            ["apr", "import", str(gguf_path), "-o", str(reimported)],
            capture_output=True, text=True, timeout=120,
        )
        assert proc.returncode == 0, (
            f"reimport failed (exit {proc.returncode}): {proc.stderr.strip()}"
        )

        oracle = load_oracle(slug, prompt_name)
        orig, err1 = apr_run_json(str(MODELS[slug]["int4"]), oracle["prompt"])
        assert err1 is None, f"original apr run failed: {err1}"

        rt, err2 = apr_run_json(str(reimported), oracle["prompt"])
        assert err2 is None, f"reimported apr run failed: {err2}"

        orig_tokens = orig.get("tokens", [])
        rt_tokens = rt.get("tokens", [])
        assert orig_tokens == rt_tokens, (
            f"Roundtrip mismatch for {slug}/{prompt_name}: "
            f"{count_mismatches(orig_tokens, rt_tokens)} diffs"
        )
    finally:
        Path(reimported).unlink(missing_ok=True)
