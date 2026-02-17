"""Unit tests for scripts/parity_check.py — all check functions, Result, format_ticket, main.

All subprocess calls are mocked. No apr CLI or model files required.
"""

import json
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from scripts.parity_check import (
    Result,
    apr_cmd_json,
    apr_run_json,
    check_bench,
    check_canary,
    check_cross_runtime,
    check_debug,
    check_diff,
    check_hex_quality,
    check_inspect,
    check_lint,
    check_list_global,
    check_llamacpp_text,
    check_oracle_id,
    check_parity_gpu,
    check_perplexity,
    check_qa,
    check_quant_drift,
    check_rosetta_diff,
    check_roundtrip,
    check_self_test,
    check_tensors,
    check_token_parity,
    check_tree,
    check_validate,
    count_char_mismatches,
    count_mismatches,
    format_ticket,
    get_apr_version,
    llamacpp_run,
    load_oracle,
    main,
    run_apr,
)

# ── Helpers ──────────────────────────────────────────────────────

SLUG = "smollm-135m"
MODEL_INFO = {
    "int4": "models/smollm-135m-int4.apr",
    "int8": "models/smollm-135m-int8.apr",
    "gguf": "models/smollm-135m-int4.gguf",
    "ppl_ceiling": 20.0,
}

# Use 32 tokens so threshold checks are meaningful (int4=5/32, int8=3/32)
ORACLE_TOKENS = list(range(32))
ORACLE = {"prompt": "1+1=", "tokens": ORACLE_TOKENS, "text": " 2"}


def _mock_proc(stdout="", stderr="", returncode=0):
    p = MagicMock()
    p.stdout = stdout
    p.stderr = stderr
    p.returncode = returncode
    return p


def _patch_run(stdout="", stderr="", returncode=0):
    return patch(
        "scripts.parity_check.subprocess.run",
        return_value=_mock_proc(stdout, stderr, returncode),
    )


def _patch_run_side(side_effect):
    return patch("scripts.parity_check.subprocess.run", side_effect=side_effect)


def _patch_oracle(oracle=None):
    return patch("scripts.parity_check.load_oracle", return_value=oracle or ORACLE)


# ── Result ───────────────────────────────────────────────────────


class TestResult:
    def test_init(self):
        r = Result("test/name")
        assert r.name == "test/name"
        assert r.passed is False
        assert r.error == ""
        assert r.details == ""

    def test_pass(self):
        r = Result("t")
        r.pass_("all good")
        assert r.passed is True
        assert r.details == "all good"

    def test_fail(self):
        r = Result("t")
        r.fail("bad", "detail")
        assert r.passed is False
        assert r.error == "bad"
        assert r.details == "detail"


# ── Pure functions ───────────────────────────────────────────────


def test_count_mismatches_identical():
    assert count_mismatches([1, 2, 3], [1, 2, 3]) == 0


def test_count_mismatches_different():
    assert count_mismatches([1, 2, 3], [4, 5, 6]) == 3


def test_count_mismatches_length_diff():
    assert count_mismatches([1, 2], [1, 2, 3]) == 1


# ── run_apr / apr_run_json / apr_cmd_json ─────────────────────────


def test_run_apr_success():
    with _patch_run("output", "", 0):
        out, _err, code = run_apr(["inspect", "m.apr"])
    assert out == "output"
    assert code == 0


def test_run_apr_timeout():
    with _patch_run_side(subprocess.TimeoutExpired("apr", 120)):
        _out, err, code = run_apr(["run", "m.apr"])
    assert code == 1
    assert "TIMEOUT" in err


def test_run_apr_not_found():
    with _patch_run_side(FileNotFoundError):
        _out, _err, code = run_apr(["run", "m.apr"])
    assert code == 127


def test_apr_run_json_success():
    payload = {"tokens": [1], "text": "hi"}
    with _patch_run(json.dumps(payload), "", 0):
        data, err = apr_run_json("m.apr", "hello")
    assert data == payload
    assert err is None


def test_apr_run_json_failure():
    with _patch_run("", "crash", 1):
        data, err = apr_run_json("m.apr", "hello")
    assert data is None
    assert "exit 1" in err


def test_apr_run_json_bad_json():
    with _patch_run("{bad", "", 0):
        data, err = apr_run_json("m.apr", "hello")
    assert data is None
    assert "invalid JSON" in err


def test_apr_cmd_json_success():
    payload = {"key": "val"}
    with _patch_run(json.dumps(payload), "", 0):
        data, _err = apr_cmd_json(["inspect", "m.apr", "--json"])
    assert data == payload


def test_apr_cmd_json_failure():
    with _patch_run("", "err", 2):
        data, err = apr_cmd_json(["inspect", "m.apr"])
    assert data is None
    assert "inspect" in err


def test_apr_cmd_json_bad_json():
    with _patch_run("nope", "", 0):
        _data, err = apr_cmd_json(["validate", "m.apr"])
    assert "invalid JSON" in err


def test_load_oracle(tmp_path):
    slug_dir = tmp_path / "test-model"
    slug_dir.mkdir()
    (slug_dir / "prompt.json").write_text(json.dumps(ORACLE))
    with patch("scripts.parity_check.ORACLE_DIR", tmp_path):
        result = load_oracle("test-model", "prompt")
    assert result == ORACLE


# ── check_canary ─────────────────────────────────────────────────


def test_check_canary_pass():
    apr_output = {"text": " 2", "tokens": ORACLE_TOKENS}
    with _patch_oracle(), _patch_run(json.dumps(apr_output), "", 0):
        results = check_canary(SLUG, MODEL_INFO)
    assert all(r.passed for r in results)
    assert len(results) == 4


def test_check_canary_text_mismatch():
    apr_output = {"text": "wrong", "tokens": ORACLE_TOKENS}
    with _patch_oracle(), _patch_run(json.dumps(apr_output), "", 0):
        results = check_canary(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results)


def test_check_canary_apr_error():
    with _patch_oracle(), _patch_run("", "crash", 1):
        results = check_canary(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results)


# ── check_token_parity ───────────────────────────────────────────


def test_check_token_parity_pass():
    apr_output = {"tokens": ORACLE_TOKENS, "text": "x"}
    with _patch_oracle(), _patch_run(json.dumps(apr_output), "", 0):
        results = check_token_parity(SLUG, MODEL_INFO)
    assert len(results) == 8  # 2 quants x 4 prompts
    assert all(r.passed for r in results)


def test_check_token_parity_exceeds_threshold():
    # All 32 tokens differ → 32 mismatches, exceeds both int4(5) and int8(3) thresholds
    bad_tokens = [t + 1000 for t in ORACLE_TOKENS]
    apr_output = {"tokens": bad_tokens, "text": "x"}
    with _patch_oracle(), _patch_run(json.dumps(apr_output), "", 0):
        results = check_token_parity(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results)


def test_check_token_parity_apr_error():
    with _patch_oracle(), _patch_run("", "err", 1):
        results = check_token_parity(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results)


# ── check_quant_drift ────────────────────────────────────────────


def test_check_quant_drift_pass():
    apr_output = {"tokens": ORACLE_TOKENS, "text": "x"}
    with _patch_oracle(), _patch_run(json.dumps(apr_output), "", 0):
        results = check_quant_drift(SLUG, MODEL_INFO)
    assert all(r.passed for r in results)
    assert len(results) == 4


def test_check_quant_drift_violated():
    """int8 has more mismatches than int4 + 1."""
    int4_out = json.dumps({"tokens": ORACLE_TOKENS, "text": ""})  # 0 mismatches
    bad_tokens = [t + 1000 for t in ORACLE_TOKENS]
    int8_out = json.dumps({"tokens": bad_tokens, "text": ""})  # 32 mismatches

    call_count = [0]

    def side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] % 2 == 1:
            return _mock_proc(int4_out, "", 0)
        else:
            return _mock_proc(int8_out, "", 0)

    with _patch_oracle(), _patch_run_side(side_effect):
        results = check_quant_drift(SLUG, MODEL_INFO)
    assert any(not r.passed for r in results)


def test_check_quant_drift_int4_error():
    with _patch_oracle(), _patch_run("", "err", 1):
        results = check_quant_drift(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results)


def test_check_quant_drift_int8_error():
    """int4 succeeds, int8 fails."""
    call_count = [0]

    def side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] % 2 == 1:
            return _mock_proc(json.dumps({"tokens": ORACLE_TOKENS}), "", 0)
        else:
            return _mock_proc("", "int8 crash", 1)

    with _patch_oracle(), _patch_run_side(side_effect):
        results = check_quant_drift(SLUG, MODEL_INFO)
    assert any(not r.passed for r in results)


# ── check_roundtrip ──────────────────────────────────────────────


def test_check_roundtrip_pass():
    apr_output = {"tokens": ORACLE_TOKENS, "text": "x"}
    with _patch_oracle(), _patch_run(json.dumps(apr_output), "", 0):
        with patch("scripts.parity_check.Path.unlink"):
            results = check_roundtrip(SLUG, MODEL_INFO)
    assert len(results) == 2
    assert all(r.passed for r in results)


def test_check_roundtrip_import_fails():
    with _patch_run("", "import error", 1):
        with patch("scripts.parity_check.Path.unlink"):
            results = check_roundtrip(SLUG, MODEL_INFO)
    assert len(results) == 1
    assert not results[0].passed


def test_check_roundtrip_token_mismatch():
    call_count = [0]

    def side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return _mock_proc("", "", 0)  # import
        elif call_count[0] % 2 == 0:
            return _mock_proc(json.dumps({"tokens": [1, 2, 3]}), "", 0)  # original
        else:
            return _mock_proc(json.dumps({"tokens": [4, 5, 6]}), "", 0)  # roundtrip

    with _patch_oracle(), _patch_run_side(side_effect):
        with patch("scripts.parity_check.Path.unlink"):
            results = check_roundtrip(SLUG, MODEL_INFO)
    assert any(not r.passed for r in results)


def test_check_roundtrip_original_fails():
    call_count = [0]

    def side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return _mock_proc("", "", 0)  # import ok
        elif call_count[0] % 2 == 0:
            return _mock_proc("", "orig fail", 1)
        else:
            return _mock_proc(json.dumps({"tokens": [1]}), "", 0)

    with _patch_oracle(), _patch_run_side(side_effect):
        with patch("scripts.parity_check.Path.unlink"):
            results = check_roundtrip(SLUG, MODEL_INFO)
    assert any(not r.passed for r in results)


def test_check_roundtrip_reimported_fails():
    call_count = [0]

    def side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return _mock_proc("", "", 0)  # import ok
        elif call_count[0] % 2 == 0:
            return _mock_proc(json.dumps({"tokens": [1]}), "", 0)  # orig ok
        else:
            return _mock_proc("", "reimport fail", 1)

    with _patch_oracle(), _patch_run_side(side_effect):
        with patch("scripts.parity_check.Path.unlink"):
            results = check_roundtrip(SLUG, MODEL_INFO)
    assert any(not r.passed for r in results)


# ── check_perplexity ─────────────────────────────────────────────


def test_check_perplexity_pass():
    payload = json.dumps({"perplexity": 10.0})
    with _patch_run(payload, "", 0):
        results = check_perplexity(SLUG, MODEL_INFO)
    assert len(results) == 3  # 2 ceiling + 1 drift
    assert all(r.passed for r in results)


def test_check_perplexity_exceeds_ceiling():
    payload = json.dumps({"perplexity": 25.0})
    with _patch_run(payload, "", 0):
        results = check_perplexity(SLUG, MODEL_INFO)
    ceiling_results = [r for r in results if "ppl/" in r.name]
    assert all(not r.passed for r in ceiling_results)


def test_check_perplexity_drift_exceeded():
    call_count = [0]

    def side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return _mock_proc(json.dumps({"perplexity": 10.0}), "", 0)
        else:
            return _mock_proc(json.dumps({"perplexity": 11.0}), "", 0)

    with _patch_run_side(side_effect):
        results = check_perplexity(SLUG, MODEL_INFO)
    drift = [r for r in results if "drift" in r.name]
    assert len(drift) == 1
    assert not drift[0].passed


def test_check_perplexity_eval_fails():
    with _patch_run("", "eval err", 1):
        results = check_perplexity(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results if "ppl/" in r.name)


def test_check_perplexity_bad_json():
    with _patch_run("not-json", "", 0):
        results = check_perplexity(SLUG, MODEL_INFO)
    assert any(not r.passed for r in results)


# ── check_inspect ────────────────────────────────────────────────


def test_check_inspect_pass():
    payload = {
        "architecture": "Llama",
        "num_layers": 30,
        "num_heads": 9,
        "hidden_size": 576,
        "vocab_size": 49152,
    }
    with _patch_run(json.dumps(payload), "", 0):
        results = check_inspect(SLUG, MODEL_INFO)
    assert len(results) == 2
    assert all(r.passed for r in results)


def test_check_inspect_arch_mismatch():
    payload = {"architecture": "Gpt2", "num_layers": 30}
    with _patch_run(json.dumps(payload), "", 0):
        results = check_inspect(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results)


def test_check_inspect_field_mismatch():
    payload = {"architecture": "Llama", "num_layers": 999}
    with _patch_run(json.dumps(payload), "", 0):
        results = check_inspect(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results)


def test_check_inspect_error():
    with _patch_run("", "err", 1):
        results = check_inspect(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results)


# ── check_validate ───────────────────────────────────────────────


def test_check_validate_pass_list():
    payload = {"checks": [{"status": "PASS"}, {"status": "ok"}]}
    with _patch_run(json.dumps(payload), "", 0):
        results = check_validate(SLUG, MODEL_INFO)
    assert all(r.passed for r in results)


def test_check_validate_pass_dict():
    payload = {"checks": {"magic": "PASS", "header": "pass"}}
    with _patch_run(json.dumps(payload), "", 0):
        results = check_validate(SLUG, MODEL_INFO)
    assert all(r.passed for r in results)


def test_check_validate_fail_list():
    payload = {"checks": [{"status": "FAIL", "name": "magic"}]}
    with _patch_run(json.dumps(payload), "", 0):
        results = check_validate(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results)


def test_check_validate_fail_dict():
    payload = {"checks": {"magic": "FAIL"}}
    with _patch_run(json.dumps(payload), "", 0):
        results = check_validate(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results)


def test_check_validate_score_format():
    """checks key is not a list/dict — falls through to score lookup."""
    payload = {"checks": 42, "score": 100}
    with _patch_run(json.dumps(payload), "", 0):
        results = check_validate(SLUG, MODEL_INFO)
    assert all(r.passed for r in results)


def test_check_validate_unexpected_format():
    """checks is not list/dict and no score/total key."""
    payload = {"checks": 42}
    with _patch_run(json.dumps(payload), "", 0):
        results = check_validate(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results)


def test_check_validate_error():
    with _patch_run("", "err", 1):
        results = check_validate(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results)


# ── check_tensors ────────────────────────────────────────────────


def test_check_tensors_pass():
    payload = {"tensors": [{"name": "layer0", "dtype": "f16"}]}
    with _patch_run(json.dumps(payload), "", 0):
        results = check_tensors(SLUG, MODEL_INFO)
    assert all(r.passed for r in results)


def test_check_tensors_empty():
    payload = {"tensors": []}
    with _patch_run(json.dumps(payload), "", 0):
        results = check_tensors(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results)


def test_check_tensors_no_key():
    """No 'tensors' key and data is a dict → empty fallback."""
    payload = {"other": "stuff"}
    with _patch_run(json.dumps(payload), "", 0):
        results = check_tensors(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results)


def test_check_tensors_error():
    with _patch_run("", "err", 1):
        results = check_tensors(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results)


# ── check_lint ───────────────────────────────────────────────────


def test_check_lint_pass_no_critical():
    payload = {"violations": [{"severity": "warning", "msg": "x"}]}
    with _patch_run(json.dumps(payload), "", 0):
        results = check_lint(SLUG, MODEL_INFO)
    assert all(r.passed for r in results)


def test_check_lint_critical():
    payload = {"violations": [{"severity": "critical", "msg": "bad"}]}
    with _patch_run(json.dumps(payload), "", 0):
        results = check_lint(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results)


def test_check_lint_no_violations_key():
    payload = {"status": "ok"}
    with _patch_run(json.dumps(payload), "", 0):
        results = check_lint(SLUG, MODEL_INFO)
    assert all(r.passed for r in results)


def test_check_lint_error():
    with _patch_run("", "err", 1):
        results = check_lint(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results)


# ── check_self_test ──────────────────────────────────────────────


def test_check_self_test_pass_list():
    stages = [{"status": "PASS"}] * 8 + [{"status": "FAIL", "name": "x"}] * 2
    with _patch_run(json.dumps({"stages": stages}), "", 0):
        results = check_self_test(SLUG, MODEL_INFO)
    assert all(r.passed for r in results)


def test_check_self_test_fail_list():
    stages = [{"status": "PASS"}] * 3 + [{"status": "FAIL", "name": "x"}] * 7
    with _patch_run(json.dumps({"stages": stages}), "", 0):
        results = check_self_test(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results)


def test_check_self_test_non_list():
    """stages is an int, falls to the else branch using data.get('passed')."""
    with _patch_run(json.dumps({"stages": 10, "passed": 9, "total": 10}), "", 0):
        results = check_self_test(SLUG, MODEL_INFO)
    assert all(r.passed for r in results)


def test_check_self_test_non_list_fail():
    with _patch_run(json.dumps({"stages": 10, "passed": 3, "total": 10}), "", 0):
        results = check_self_test(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results)


def test_check_self_test_error():
    with _patch_run("", "err", 1):
        results = check_self_test(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results)


# ── check_diff ───────────────────────────────────────────────────


def test_check_diff_pass():
    payload = {"differences": [{"type": "dtype"}, {"type": "size"}]}
    with _patch_run(json.dumps(payload), "", 0):
        results = check_diff(SLUG, MODEL_INFO)
    assert all(r.passed for r in results)


def test_check_diff_structural():
    payload = {"differences": [{"type": "structural"}]}
    with _patch_run(json.dumps(payload), "", 0):
        results = check_diff(SLUG, MODEL_INFO)
    assert not results[0].passed


def test_check_diff_non_list():
    payload = {"summary": "same"}
    with _patch_run(json.dumps(payload), "", 0):
        results = check_diff(SLUG, MODEL_INFO)
    assert results[0].passed


def test_check_diff_error():
    with _patch_run("", "err", 1):
        results = check_diff(SLUG, MODEL_INFO)
    assert not results[0].passed


# ── check_tree ───────────────────────────────────────────────────


def test_check_tree_pass():
    payload = {"num_layers": 30, "total_tensors": 100}
    with _patch_run(json.dumps(payload), "", 0):
        results = check_tree(SLUG, MODEL_INFO)
    assert all(r.passed for r in results)


def test_check_tree_layer_mismatch():
    payload = {"num_layers": 5}
    with _patch_run(json.dumps(payload), "", 0):
        results = check_tree(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results)


def test_check_tree_no_layers_field():
    payload = {"other_key": "val"}
    with _patch_run(json.dumps(payload), "", 0):
        results = check_tree(SLUG, MODEL_INFO)
    assert all(r.passed for r in results)


def test_check_tree_error():
    with _patch_run("", "err", 1):
        results = check_tree(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results)


# ── check_oracle_id ──────────────────────────────────────────────


def test_check_oracle_id_pass():
    payload = {"architecture": "Llama", "parameters": "135M"}
    with _patch_run(json.dumps(payload), "", 0):
        results = check_oracle_id(SLUG, MODEL_INFO)
    assert all(r.passed for r in results)


def test_check_oracle_id_mismatch():
    payload = {"architecture": "gpt2"}
    with _patch_run(json.dumps(payload), "", 0):
        results = check_oracle_id(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results)


def test_check_oracle_id_error():
    with _patch_run("", "err", 1):
        results = check_oracle_id(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results)


# ── check_hex_quality ────────────────────────────────────────────


def test_check_hex_pass_with_std():
    payload = {"std": 0.05}
    with _patch_run(json.dumps(payload), "", 0):
        results = check_hex_quality(SLUG, MODEL_INFO)
    assert all(r.passed for r in results)


def test_check_hex_zero_std():
    payload = {"std": 0.0}
    with _patch_run(json.dumps(payload), "", 0):
        results = check_hex_quality(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results)


def test_check_hex_no_std():
    payload = {"data": "ok"}
    with _patch_run(json.dumps(payload), "", 0):
        results = check_hex_quality(SLUG, MODEL_INFO)
    assert all(r.passed for r in results)


def test_check_hex_nested_std():
    payload = {"statistics": {"std": 0.1}}
    with _patch_run(json.dumps(payload), "", 0):
        results = check_hex_quality(SLUG, MODEL_INFO)
    assert all(r.passed for r in results)


def test_check_hex_first_fails_fallback_succeeds():
    call_count = [0]

    def side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] % 2 == 1:
            return _mock_proc("", "tensor not found", 1)
        else:
            return _mock_proc(json.dumps({"data": "ok"}), "", 0)

    with _patch_run_side(side_effect):
        results = check_hex_quality(SLUG, MODEL_INFO)
    assert all(r.passed for r in results)


def test_check_hex_both_fail():
    with _patch_run("", "err", 1):
        results = check_hex_quality(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results)


# ── check_debug ──────────────────────────────────────────────────


def test_check_debug_pass():
    payload = {"health": "ok"}
    with _patch_run(json.dumps(payload), "", 0):
        results = check_debug(SLUG, MODEL_INFO)
    assert all(r.passed for r in results)


def test_check_debug_unhealthy():
    payload = {"health": "degraded"}
    with _patch_run(json.dumps(payload), "", 0):
        results = check_debug(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results)


def test_check_debug_error_field():
    payload = {"error": "something broke"}
    with _patch_run(json.dumps(payload), "", 0):
        results = check_debug(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results)


def test_check_debug_no_health_no_error():
    payload = {"status": ""}
    with _patch_run(json.dumps(payload), "", 0):
        results = check_debug(SLUG, MODEL_INFO)
    assert all(r.passed for r in results)


def test_check_debug_error():
    with _patch_run("", "err", 1):
        results = check_debug(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results)


# ── check_bench ──────────────────────────────────────────────────


def test_check_bench_pass():
    payload = {"tokens_per_second": 150.0}
    with _patch_run(json.dumps(payload), "", 0):
        results = check_bench(SLUG, MODEL_INFO)
    assert all(r.passed for r in results)


def test_check_bench_zero_throughput():
    payload = {"tokens_per_second": 0.0}
    with _patch_run(json.dumps(payload), "", 0):
        results = check_bench(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results)


def test_check_bench_no_throughput():
    payload = {"other": "data"}
    with _patch_run(json.dumps(payload), "", 0):
        results = check_bench(SLUG, MODEL_INFO)
    assert all(r.passed for r in results)


def test_check_bench_error():
    with _patch_run("", "err", 1):
        results = check_bench(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results)


# ── check_qa ─────────────────────────────────────────────────────


def test_check_qa_pass():
    gates = [{"status": "PASS"}] * 5
    with _patch_run(json.dumps({"gates": gates}), "", 0):
        results = check_qa(SLUG, MODEL_INFO)
    assert all(r.passed for r in results)


def test_check_qa_too_few_gates():
    gates = [{"status": "PASS"}] * 2
    with _patch_run(json.dumps({"gates": gates}), "", 0):
        results = check_qa(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results)


def test_check_qa_critical_failure():
    gates = [
        {"status": "PASS"},
        {"status": "PASS"},
        {"status": "PASS"},
        {"status": "CRITICAL", "severity": "critical"},
    ]
    with _patch_run(json.dumps({"gates": gates}), "", 0):
        results = check_qa(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results)


def test_check_qa_non_list():
    """gates is an int → falls to else branch using gates_executed."""
    with _patch_run(json.dumps({"gates": 10, "gates_executed": 5}), "", 0):
        results = check_qa(SLUG, MODEL_INFO)
    assert all(r.passed for r in results)


def test_check_qa_non_list_too_few():
    with _patch_run(json.dumps({"gates": 10, "gates_executed": 1}), "", 0):
        results = check_qa(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results)


def test_check_qa_error():
    with _patch_run("", "err", 1):
        results = check_qa(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results)


# ── check_list_global ────────────────────────────────────────────


def test_check_list_global_pass():
    with _patch_run(json.dumps({"models": []}), "", 0):
        results = check_list_global()
    assert len(results) == 1
    assert results[0].passed


def test_check_list_global_error():
    with _patch_run("", "err", 1):
        results = check_list_global()
    assert not results[0].passed


# ── check_rosetta_diff ───────────────────────────────────────────


def test_check_rosetta_diff_pass():
    payload = {"mismatches": []}
    with _patch_run(json.dumps(payload), "", 0):
        results = check_rosetta_diff(SLUG, MODEL_INFO)
    assert results[0].passed


def test_check_rosetta_diff_mismatches():
    payload = {"mismatches": [{"tensor": "layer.0", "issue": "layout"}]}
    with _patch_run(json.dumps(payload), "", 0):
        results = check_rosetta_diff(SLUG, MODEL_INFO)
    assert not results[0].passed


def test_check_rosetta_diff_error():
    with _patch_run("", "err", 1):
        results = check_rosetta_diff(SLUG, MODEL_INFO)
    assert not results[0].passed


# ── check_parity_gpu ─────────────────────────────────────────────


def test_check_parity_gpu_pass():
    payload = {"match": True}
    with _patch_run(json.dumps(payload), "", 0):
        with patch("scripts.parity_check.Path.exists", return_value=True):
            results = check_parity_gpu(SLUG, MODEL_INFO)
    assert results[0].passed


def test_check_parity_gpu_fail():
    payload = {"match": False}
    with _patch_run(json.dumps(payload), "", 0):
        with patch("scripts.parity_check.Path.exists", return_value=True):
            results = check_parity_gpu(SLUG, MODEL_INFO)
    assert not results[0].passed


def test_check_parity_gpu_no_gguf():
    info = {**MODEL_INFO, "gguf": None}
    results = check_parity_gpu(SLUG, info)
    assert not results[0].passed


def test_check_parity_gpu_cuda_error():
    """CUDA error → graceful skip (passes)."""
    with _patch_run("", "CUDA not available", 1):
        with patch("scripts.parity_check.Path.exists", return_value=True):
            results = check_parity_gpu(SLUG, MODEL_INFO)
    assert results[0].passed


def test_check_parity_gpu_no_match_key():
    payload = {"data": "ok"}
    with _patch_run(json.dumps(payload), "", 0):
        with patch("scripts.parity_check.Path.exists", return_value=True):
            results = check_parity_gpu(SLUG, MODEL_INFO)
    assert results[0].passed


# ── format_ticket ────────────────────────────────────────────────


def test_format_ticket_basic():
    r = Result("test/fail")
    r.fail("something broke", "detail info")
    with patch("scripts.parity_check.get_apr_version", return_value="apr 0.3.0"):
        ticket = format_ticket([r])
    assert "something broke" in ticket
    assert "detail info" in ticket
    assert "test/fail" in ticket
    assert "apr 0.3.0" in ticket


def test_format_ticket_no_details():
    r = Result("test/fail")
    r.fail("oops")
    with patch("scripts.parity_check.get_apr_version", return_value="unknown"):
        ticket = format_ticket([r])
    assert "oops" in ticket
    assert "1 failure" in ticket


# ── get_apr_version ──────────────────────────────────────────────


def test_get_apr_version_success():
    with patch(
        "scripts.parity_check.subprocess.run",
        return_value=_mock_proc("apr 0.3.0", "", 0),
    ):
        assert get_apr_version() == "apr 0.3.0"


def test_get_apr_version_failure():
    with patch(
        "scripts.parity_check.subprocess.run",
        return_value=_mock_proc("", "", 1),
    ):
        assert get_apr_version() == "unknown"


def test_get_apr_version_not_found():
    with patch(
        "scripts.parity_check.subprocess.run",
        side_effect=FileNotFoundError,
    ):
        assert get_apr_version() == "unknown"


def test_get_apr_version_timeout():
    with patch(
        "scripts.parity_check.subprocess.run",
        side_effect=subprocess.TimeoutExpired("apr", 5),
    ):
        assert get_apr_version() == "unknown"


# ── main ─────────────────────────────────────────────────────────


def test_main_check_list_pass():
    """--check list with mocked success."""
    with patch.object(sys, "argv", ["parity_check.py", "--check", "list"]):
        with _patch_run(json.dumps({"models": []}), "", 0):
            with patch("scripts.parity_check.Path.exists", return_value=False):
                with pytest.raises(SystemExit) as exc:
                    main()
    assert exc.value.code == 0


def test_main_check_list_fail():
    """--check list fails → exit 1."""
    with patch.object(sys, "argv", ["parity_check.py", "--check", "list"]):
        with _patch_run("", "err", 1):
            with patch("scripts.parity_check.Path.exists", return_value=False):
                with pytest.raises(SystemExit) as exc:
                    main()
    assert exc.value.code == 1


def test_main_ticket_flag():
    """--ticket with failures prints ticket markdown."""
    with patch.object(sys, "argv", ["parity_check.py", "--check", "list", "--ticket"]):
        with _patch_run("", "err", 1):
            with patch("scripts.parity_check.Path.exists", return_value=False):
                with patch("scripts.parity_check.get_apr_version", return_value="test"):
                    with pytest.raises(SystemExit) as exc:
                        main()
    assert exc.value.code == 1


def test_main_single_model():
    """--model flag with mocked success."""
    with patch.object(sys, "argv", ["parity_check.py", "--model", "smollm-135m", "--check", "inspect"]):
        payload = {"architecture": "Llama", "num_layers": 30}
        with _patch_run(json.dumps(payload), "", 0):
            with patch("scripts.parity_check.Path.exists", return_value=True):
                with pytest.raises(SystemExit) as exc:
                    main()
    assert exc.value.code == 0


def test_main_all_models_missing():
    """Default (all checks) with no model files → only global checks."""
    with patch.object(sys, "argv", ["parity_check.py"]):
        with _patch_run(json.dumps({"models": []}), "", 0):
            with patch("scripts.parity_check.Path.exists", return_value=False):
                with pytest.raises(SystemExit) as exc:
                    main()
    assert exc.value.code == 0


# ── llamacpp_run ────────────────────────────────────────────────


def test_llamacpp_run_success():
    with patch("scripts.parity_check.LLAMACPP_BIN", "/usr/bin/llama-completion"), _patch_run(
        "  output text  ", "", 0
    ):
        data, err = llamacpp_run("model.gguf", "hello")
    assert err is None
    assert data == {"text": "output text"}


def test_llamacpp_run_no_binary():
    with patch("scripts.parity_check.LLAMACPP_BIN", ""):
        data, err = llamacpp_run("model.gguf", "hello")
    assert data is None
    assert "not found" in err


def test_llamacpp_run_nonzero_exit():
    with patch("scripts.parity_check.LLAMACPP_BIN", "/usr/bin/llama-completion"), _patch_run(
        "", "load failed", 1
    ):
        data, err = llamacpp_run("model.gguf", "hello")
    assert data is None
    assert "exit 1" in err


def test_llamacpp_run_timeout():
    with patch(
        "scripts.parity_check.LLAMACPP_BIN", "/usr/bin/llama-completion"
    ), _patch_run_side(subprocess.TimeoutExpired(cmd="llama-completion", timeout=120)):
        data, err = llamacpp_run("model.gguf", "hello")
    assert data is None
    assert "TIMEOUT" in err


def test_llamacpp_run_file_not_found():
    with patch(
        "scripts.parity_check.LLAMACPP_BIN", "/usr/bin/llama-completion"
    ), _patch_run_side(FileNotFoundError):
        data, err = llamacpp_run("model.gguf", "hello")
    assert data is None
    assert "not found" in err


# ── count_char_mismatches ───────────────────────────────────────


def test_count_char_mismatches_identical():
    assert count_char_mismatches("hello", "hello") == 0


def test_count_char_mismatches_all_different():
    assert count_char_mismatches("abc", "xyz") == 3


def test_count_char_mismatches_length_diff():
    assert count_char_mismatches("hello", "he") == 3


def test_count_char_mismatches_partial():
    assert count_char_mismatches("abcd", "abxd") == 1


# ── check_llamacpp_text ────────────────────────────────────────


def _patch_llamacpp_run(text):
    return patch("scripts.parity_check.llamacpp_run", return_value=({"text": text}, None))


def _patch_llamacpp_run_err(err):
    return patch("scripts.parity_check.llamacpp_run", return_value=(None, err))


def test_check_llamacpp_text_pass():
    """Q8 text within threshold → pass."""
    with _patch_oracle(), _patch_llamacpp_run(" 2"), patch(
        "scripts.parity_check.Path.exists", return_value=True
    ):
        results = check_llamacpp_text(SLUG, MODEL_INFO)
    q8_results = [r for r in results if "q8_0" in r.name]
    assert all(r.passed for r in q8_results)


def test_check_llamacpp_text_fail():
    """Q8 text exceeds threshold → fail."""
    bad_text = "x" * 100  # totally different
    with _patch_oracle(), _patch_llamacpp_run(bad_text), patch(
        "scripts.parity_check.Path.exists", return_value=True
    ):
        results = check_llamacpp_text(SLUG, MODEL_INFO)
    assert any(not r.passed for r in results)


def test_check_llamacpp_text_error():
    """llama-completion returns error → fail."""
    with _patch_oracle(), _patch_llamacpp_run_err("model load failed"), patch(
        "scripts.parity_check.Path.exists", return_value=True
    ):
        results = check_llamacpp_text(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results)


def test_check_llamacpp_text_missing_gguf():
    """No GGUF file → fail with helpful message."""
    with patch("scripts.parity_check.Path.exists", return_value=False):
        results = check_llamacpp_text(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results)
    assert any("not found" in r.error for r in results)


# ── check_cross_runtime ────────────────────────────────────────


def test_check_cross_runtime_pass():
    """Both runtimes produce same text → pass."""
    apr_out = {"tokens": ORACLE_TOKENS, "text": "match text"}
    with _patch_oracle(), patch(
        "scripts.parity_check.apr_run_json", return_value=(apr_out, None)
    ), patch(
        "scripts.parity_check.llamacpp_run", return_value=({"text": "match text"}, None)
    ), patch("scripts.parity_check.Path.exists", return_value=True):
        results = check_cross_runtime(SLUG, MODEL_INFO)
    assert all(r.passed for r in results)


def test_check_cross_runtime_mismatch():
    """Different text → fail."""
    apr_out = {"tokens": ORACLE_TOKENS, "text": "apr text"}
    with _patch_oracle(), patch(
        "scripts.parity_check.apr_run_json", return_value=(apr_out, None)
    ), patch(
        "scripts.parity_check.llamacpp_run", return_value=({"text": "llama text"}, None)
    ), patch("scripts.parity_check.Path.exists", return_value=True):
        results = check_cross_runtime(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results)


def test_check_cross_runtime_apr_error():
    """apr fails → fail."""
    with _patch_oracle(), patch(
        "scripts.parity_check.apr_run_json", return_value=(None, "apr error")
    ), patch(
        "scripts.parity_check.llamacpp_run", return_value=({"text": "ok"}, None)
    ), patch("scripts.parity_check.Path.exists", return_value=True):
        results = check_cross_runtime(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results)


def test_check_cross_runtime_llamacpp_error():
    """llama-completion fails → fail."""
    apr_out = {"tokens": ORACLE_TOKENS, "text": "ok"}
    with _patch_oracle(), patch(
        "scripts.parity_check.apr_run_json", return_value=(apr_out, None)
    ), patch(
        "scripts.parity_check.llamacpp_run", return_value=(None, "llama error")
    ), patch("scripts.parity_check.Path.exists", return_value=True):
        results = check_cross_runtime(SLUG, MODEL_INFO)
    assert all(not r.passed for r in results)


def test_check_cross_runtime_missing_gguf():
    """No GGUF file → fail with helpful message."""
    with patch("scripts.parity_check.Path.exists", return_value=False):
        results = check_cross_runtime(SLUG, MODEL_INFO)
    assert len(results) == 1
    assert not results[0].passed
    assert "not found" in results[0].error
