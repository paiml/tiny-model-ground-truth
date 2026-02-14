"""Claim 10: Model Integrity — validate passes magic/header/version.

Falsification: Any status != PASS in `apr validate --json` output.
Threshold: all validation checks pass (0 tolerance).
Sample size: n = 6 (3 models x 2 quantization levels).
"""

import pytest
from helpers import MODEL_QUANT_PARAMS, MODELS, apr_cmd_json

skip_no_apr = pytest.importorskip("shutil").which("apr") is not None
pytestmark = [
    pytest.mark.validate,
    pytest.mark.skipif(not skip_no_apr, reason="apr CLI not in PATH"),
]


@pytest.mark.parametrize("slug,quant", MODEL_QUANT_PARAMS)
def test_validate_passes(slug: str, quant: str):
    """apr validate passes all integrity checks (Claim 10)."""
    model_path = str(MODELS[slug][quant])

    data, err = apr_cmd_json(["validate", model_path, "--json"])
    assert err is None, f"apr validate failed: {err}"

    checks = data.get("checks", data.get("results", []))
    if isinstance(checks, list):
        failures = [
            c for c in checks
            if c.get("status") not in ("PASS", "pass", "ok")
        ]
        assert not failures, (
            f"{len(failures)} validation failures for {slug}/{quant}: {failures}"
        )
    elif isinstance(checks, dict):
        failures = {
            k: v for k, v in checks.items()
            if v not in ("PASS", "pass", "ok")
        }
        assert not failures, (
            f"validation failures for {slug}/{quant}: {failures}"
        )
