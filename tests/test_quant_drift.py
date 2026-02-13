"""Claim 4: Quantization Drift Ordering — Int8 <= Int4 + 1.

Higher precision (Int8) must produce fewer or equal token mismatches
compared to lower precision (Int4), with ±1 margin for error propagation.
Falsification: int8_mismatches > int4_mismatches + 1.
Sample size: n = 12 (3 models x 4 prompts).
"""

import pytest

from helpers import (
    MODELS, MODEL_PROMPT_PARAMS,
    apr_run_json, count_mismatches, load_oracle,
)

DRIFT_MARGIN = 1

skip_no_apr = pytest.importorskip("shutil").which("apr") is not None
pytestmark = [
    pytest.mark.quant_drift,
    pytest.mark.skipif(not skip_no_apr, reason="apr CLI not in PATH"),
]


@pytest.mark.parametrize("slug,prompt_name", MODEL_PROMPT_PARAMS)
def test_int8_drift_leq_int4(slug: str, prompt_name: str):
    """Int8 mismatches ≤ Int4 mismatches + 1 (Claim 4)."""
    oracle = load_oracle(slug, prompt_name)

    out_int4, err4 = apr_run_json(str(MODELS[slug]["int4"]), oracle["prompt"])
    assert err4 is None, f"Int4 apr run failed: {err4}"

    out_int8, err8 = apr_run_json(str(MODELS[slug]["int8"]), oracle["prompt"])
    assert err8 is None, f"Int8 apr run failed: {err8}"

    m4 = count_mismatches(out_int4.get("tokens", []), oracle["tokens"])
    m8 = count_mismatches(out_int8.get("tokens", []), oracle["tokens"])

    assert m8 <= m4 + DRIFT_MARGIN, (
        f"Drift violated for {slug}/{prompt_name}: "
        f"int8={m8} > int4={m4} + {DRIFT_MARGIN}"
    )
