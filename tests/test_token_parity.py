"""Claims 2 & 3: Token parity bounds for Int4 and Int8.

Claim 2: Int4 produces at most 5/32 token mismatches vs oracle (n = 12).
Claim 3: Int8 produces at most 3/32 token mismatches vs oracle (n = 12).
Falsification: mismatches exceed threshold.
Sample size: n = 24 (3 models x 4 prompts x 2 quant levels).
"""

import pytest
from helpers import (
    MODEL_PROMPT_PARAMS,
    MODELS,
    apr_run_json,
    count_mismatches,
    load_oracle,
)

INT4_THRESHOLD = 5  # Claim 2: <=5/32 mismatches
INT8_THRESHOLD = 3  # Claim 3: <=3/32 mismatches

pytestmark = [
    pytest.mark.requires_apr,
    pytest.mark.token_parity,
]


@pytest.mark.parametrize("slug,prompt_name", MODEL_PROMPT_PARAMS)
def test_int4_token_mismatch_within_bound(slug: str, prompt_name: str):
    """Int4 ≤5/32 token mismatches vs oracle (Claim 2)."""
    oracle = load_oracle(slug, prompt_name)
    output, err = apr_run_json(str(MODELS[slug]["int4"]), oracle["prompt"])
    assert err is None, f"apr run failed: {err}"

    m = count_mismatches(output.get("tokens", []), oracle["tokens"])
    assert m <= INT4_THRESHOLD, (
        f"Int4 {slug}/{prompt_name}: {m} mismatches > threshold {INT4_THRESHOLD}"
    )


@pytest.mark.parametrize("slug,prompt_name", MODEL_PROMPT_PARAMS)
def test_int8_token_mismatch_within_bound(slug: str, prompt_name: str):
    """Int8 ≤3/32 token mismatches vs oracle (Claim 3)."""
    oracle = load_oracle(slug, prompt_name)
    output, err = apr_run_json(str(MODELS[slug]["int8"]), oracle["prompt"])
    assert err is None, f"apr run failed: {err}"

    m = count_mismatches(output.get("tokens", []), oracle["tokens"])
    assert m <= INT8_THRESHOLD, (
        f"Int8 {slug}/{prompt_name}: {m} mismatches > threshold {INT8_THRESHOLD}"
    )
