"""Claims 7 & 8: Perplexity bounds and drift.

Claim 7: PPL < model-specific ceiling (SmolLM < 20.0, Qwen2 < 15.0, GPT-2 < 30.0).
Claim 8: |PPL_Int4 - PPL_Int8| < 0.5 for each model (σ = 0, deterministic).
Sample size: n = 6 (Claim 7), n = 3 (Claim 8).
"""

import pytest

from helpers import MODELS, MODEL_PARAMS, apr_eval_json

PPL_DRIFT_THRESHOLD = 0.5  # Claim 8

skip_no_apr = pytest.importorskip("shutil").which("apr") is not None
pytestmark = [
    pytest.mark.perplexity,
    pytest.mark.skipif(not skip_no_apr, reason="apr CLI not in PATH"),
]


@pytest.mark.parametrize("slug", MODEL_PARAMS)
@pytest.mark.parametrize("quant", ["int4", "int8"])
def test_perplexity_within_ceiling(slug: str, quant: str):
    """PPL < model-specific ceiling (Claim 7)."""
    data, err = apr_eval_json(str(MODELS[slug][quant]))
    assert err is None, f"apr eval failed: {err}"

    ppl = data.get("perplexity", float("inf"))
    ceiling = MODELS[slug]["ppl_ceiling"]
    assert ppl < ceiling, f"{slug}/{quant}: PPL={ppl:.2f} >= ceiling {ceiling}"


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_perplexity_drift_within_bound(slug: str):
    """|PPL_Int4 - PPL_Int8| < 0.5 (Claim 8). σ = 0 (deterministic)."""
    data4, err4 = apr_eval_json(str(MODELS[slug]["int4"]))
    assert err4 is None, f"Int4 apr eval failed: {err4}"

    data8, err8 = apr_eval_json(str(MODELS[slug]["int8"]))
    assert err8 is None, f"Int8 apr eval failed: {err8}"

    ppl4 = data4.get("perplexity", float("inf"))
    ppl8 = data8.get("perplexity", float("inf"))
    diff = abs(ppl4 - ppl8)

    assert diff < PPL_DRIFT_THRESHOLD, (
        f"{slug}: |{ppl4:.2f} - {ppl8:.2f}| = {diff:.3f} >= {PPL_DRIFT_THRESHOLD}"
    )
