"""Claim 1: Int8 Canary Parity — exact text match vs oracle.

Falsification: Any text mismatch between Int8 APR output and float32
HuggingFace transformers oracle. Threshold: 0 tolerance.
Sample size: n = 12 (3 models x 4 prompts).
"""

import pytest
from helpers import MODEL_PROMPT_PARAMS, MODELS, apr_run_json, load_oracle

pytestmark = [
    pytest.mark.requires_apr,
    pytest.mark.canary,
]


@pytest.mark.parametrize("slug,prompt_name", MODEL_PROMPT_PARAMS)
def test_int8_text_matches_oracle(slug: str, prompt_name: str):
    """Int8 APR output text must exactly match float32 oracle (Claim 1)."""
    oracle = load_oracle(slug, prompt_name)
    model_path = MODELS[slug]["int8"]

    output, err = apr_run_json(str(model_path), oracle["prompt"])
    assert err is None, f"apr run failed: {err}"

    actual_text = output.get("text", "")
    expected_text = oracle["text"]
    assert actual_text == expected_text, (
        f"Int8 text mismatch for {slug}/{prompt_name}\n"
        f"  expected: {expected_text!r}\n"
        f"  got:      {actual_text!r}"
    )
