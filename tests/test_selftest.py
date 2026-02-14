"""Claim 13: Pipeline Self-Test — >=7/10 check stages pass.

Falsification: Any model where passed_stages < 7 in `apr check --json`.
Threshold: >= 7/10 stages pass.
Sample size: n = 6 (3 models x 2 quantization levels).
"""

import pytest
from helpers import MODEL_QUANT_PARAMS, MODELS, apr_cmd_json

pytestmark = [
    pytest.mark.requires_apr,
    pytest.mark.selftest,
]


@pytest.mark.parametrize("slug,quant", MODEL_QUANT_PARAMS)
def test_self_test_passes(slug: str, quant: str):
    """apr check passes >=7/10 pipeline stages (Claim 13)."""
    model_path = str(MODELS[slug][quant])

    data, err = apr_cmd_json(["check", model_path, "--json"])
    assert err is None, f"apr check failed: {err}"

    stages = data.get("stages", data.get("checks", data.get("results", [])))
    if isinstance(stages, list):
        passed = sum(
            1 for s in stages
            if s.get("status") in ("PASS", "pass", "ok")
        )
        total = len(stages)
        failed_names = [
            s.get("name", "?") for s in stages
            if s.get("status") not in ("PASS", "pass", "ok")
        ]
        assert passed >= 7, (
            f"only {passed}/{total} stages passed for {slug}/{quant}, "
            f"failed: {failed_names}"
        )
    else:
        passed = data.get("passed", 0)
        total = data.get("total", 0)
        assert passed >= 7, (
            f"only {passed}/{total} stages passed for {slug}/{quant}"
        )
