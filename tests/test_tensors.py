"""Claim 11: Tensor Structure — correct tensor count and dtypes.

Falsification: Empty or missing tensor list from `apr tensors --json`.
Threshold: tensor_count > 0.
Sample size: n = 6 (3 models x 2 quantization levels).
"""

import pytest
from helpers import MODEL_QUANT_PARAMS, MODELS, apr_cmd_json

skip_no_apr = pytest.importorskip("shutil").which("apr") is not None
pytestmark = [
    pytest.mark.tensors,
    pytest.mark.skipif(not skip_no_apr, reason="apr CLI not in PATH"),
]


@pytest.mark.parametrize("slug,quant", MODEL_QUANT_PARAMS)
def test_tensors_non_empty(slug: str, quant: str):
    """apr tensors returns non-empty tensor list (Claim 11)."""
    model_path = str(MODELS[slug][quant])

    data, err = apr_cmd_json(["tensors", model_path, "--json"])
    assert err is None, f"apr tensors failed: {err}"

    tensors = data.get("tensors", data if isinstance(data, list) else [])
    assert len(tensors) > 0, (
        f"no tensors returned for {slug}/{quant}, "
        f"keys: {list(data.keys()) if isinstance(data, dict) else 'list'}"
    )
