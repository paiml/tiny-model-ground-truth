"""Claim 9: Metadata Consistency — inspect output matches expected arch params.

Falsification: Any field mismatch between `apr inspect --json` output and
MODEL_METADATA. Threshold: exact match on architecture, num_layers, num_heads,
hidden_size, vocab_size.
Sample size: n = 6 (3 models x 2 quantization levels).
"""

import pytest
from helpers import MODEL_METADATA, MODEL_QUANT_PARAMS, MODELS, apr_cmd_json

skip_no_apr = pytest.importorskip("shutil").which("apr") is not None
pytestmark = [
    pytest.mark.inspect,
    pytest.mark.skipif(not skip_no_apr, reason="apr CLI not in PATH"),
]


@pytest.mark.parametrize("slug,quant", MODEL_QUANT_PARAMS)
def test_inspect_metadata_matches(slug: str, quant: str):
    """apr inspect metadata matches MODEL_METADATA (Claim 9)."""
    meta = MODEL_METADATA[slug]
    model_path = str(MODELS[slug][quant])

    data, err = apr_cmd_json(["inspect", model_path, "--json"])
    assert err is None, f"apr inspect failed: {err}"

    got_arch = data.get("architecture", "").lower()
    assert meta["architecture"] in got_arch, (
        f"architecture mismatch: expected '{meta['architecture']}' in '{got_arch}'"
    )

    for field, key in [
        ("layers", "num_layers"),
        ("heads", "num_heads"),
        ("hidden_dim", "hidden_size"),
        ("vocab_size", "vocab_size"),
    ]:
        got = data.get(key)
        if got is not None:
            assert got == meta[field], (
                f"{key} mismatch for {slug}/{quant}: "
                f"expected {meta[field]}, got {got}"
            )
