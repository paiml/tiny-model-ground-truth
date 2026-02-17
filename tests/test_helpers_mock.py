"""Mock-based unit tests for tests/helpers.py subprocess wrappers.

Tests all apr_* functions and load_oracle by mocking subprocess.run
and filesystem I/O. No apr CLI or model files required.
"""

import json
import subprocess
from unittest.mock import MagicMock, patch

from helpers import (
    apr_cmd_json,
    apr_eval_json,
    apr_run_json,
    count_char_mismatches,
    llamacpp_run,
    load_oracle,
    run_apr,
)

# ── load_oracle ──────────────────────────────────────────────────


def test_load_oracle_reads_json(tmp_path):
    oracle_data = {"prompt": "hello", "tokens": [1, 2, 3], "text": "world"}
    slug_dir = tmp_path / "smollm-135m"
    slug_dir.mkdir()
    (slug_dir / "greeting.json").write_text(json.dumps(oracle_data))

    with patch("helpers.ORACLE_DIR", tmp_path):
        result = load_oracle("smollm-135m", "greeting")

    assert result == oracle_data


# ── run_apr ──────────────────────────────────────────────────────


def _mock_proc(stdout="", stderr="", returncode=0):
    proc = MagicMock()
    proc.stdout = stdout
    proc.stderr = stderr
    proc.returncode = returncode
    return proc


def test_run_apr_success():
    with patch("helpers.subprocess.run", return_value=_mock_proc("out", "err", 0)):
        stdout, stderr, code = run_apr(["inspect", "model.apr", "--json"])
    assert stdout == "out"
    assert stderr == "err"
    assert code == 0


def test_run_apr_nonzero_exit():
    with patch("helpers.subprocess.run", return_value=_mock_proc("", "bad", 1)):
        _stdout, stderr, code = run_apr(["run", "model.apr"])
    assert code == 1
    assert stderr == "bad"


def test_run_apr_timeout():
    with patch("helpers.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="apr", timeout=55)):
        _stdout, stderr, code = run_apr(["run", "model.apr"])
    assert code == 1
    assert "TIMEOUT" in stderr


def test_run_apr_not_found():
    with patch("helpers.subprocess.run", side_effect=FileNotFoundError):
        _stdout, stderr, code = run_apr(["run", "model.apr"])
    assert code == 127
    assert "not found" in stderr


# ── apr_run_json ─────────────────────────────────────────────────


def test_apr_run_json_success():
    payload = {"tokens": [1, 2, 3], "text": "hello"}
    with patch("helpers.subprocess.run", return_value=_mock_proc(json.dumps(payload), "", 0)):
        data, err = apr_run_json("model.apr", "hello")
    assert err is None
    assert data == payload


def test_apr_run_json_nonzero_exit():
    with patch("helpers.subprocess.run", return_value=_mock_proc("", "segfault", 139)):
        data, err = apr_run_json("model.apr", "hello")
    assert data is None
    assert "exit 139" in err


def test_apr_run_json_timeout():
    with patch("helpers.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="apr", timeout=55)):
        data, err = apr_run_json("model.apr", "hello")
    assert data is None
    assert "TIMEOUT" in err


def test_apr_run_json_not_found():
    with patch("helpers.subprocess.run", side_effect=FileNotFoundError):
        data, err = apr_run_json("model.apr", "hello")
    assert data is None
    assert "not found" in err


def test_apr_run_json_invalid_json():
    with patch("helpers.subprocess.run", return_value=_mock_proc("not json{", "", 0)):
        data, err = apr_run_json("model.apr", "hello")
    assert data is None
    assert "invalid JSON" in err


# ── apr_eval_json ────────────────────────────────────────────────


def test_apr_eval_json_success():
    payload = {"perplexity": 12.5}
    with patch("helpers.subprocess.run", return_value=_mock_proc(json.dumps(payload), "", 0)):
        data, err = apr_eval_json("model.apr")
    assert err is None
    assert data["perplexity"] == 12.5


def test_apr_eval_json_nonzero_exit():
    with patch("helpers.subprocess.run", return_value=_mock_proc("", "error msg", 1)):
        data, err = apr_eval_json("model.apr")
    assert data is None
    assert "exit 1" in err


def test_apr_eval_json_timeout():
    with patch("helpers.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="apr", timeout=55)):
        data, err = apr_eval_json("model.apr")
    assert data is None
    assert "TIMEOUT" in err


def test_apr_eval_json_not_found():
    with patch("helpers.subprocess.run", side_effect=FileNotFoundError):
        data, err = apr_eval_json("model.apr")
    assert data is None
    assert "not found" in err


def test_apr_eval_json_invalid_json():
    with patch("helpers.subprocess.run", return_value=_mock_proc("broken", "", 0)):
        data, err = apr_eval_json("model.apr")
    assert data is None
    assert "invalid JSON" in err


# ── apr_cmd_json ─────────────────────────────────────────────────


def test_apr_cmd_json_success():
    payload = {"checks": [{"status": "PASS"}]}
    with patch("helpers.subprocess.run", return_value=_mock_proc(json.dumps(payload), "", 0)):
        data, err = apr_cmd_json(["validate", "model.apr", "--json"])
    assert err is None
    assert data == payload


def test_apr_cmd_json_nonzero_exit():
    with patch("helpers.subprocess.run", return_value=_mock_proc("", "fail", 5)):
        data, err = apr_cmd_json(["check", "model.apr", "--json"])
    assert data is None
    assert "exit 5" in err
    assert "check" in err


def test_apr_cmd_json_invalid_json():
    with patch("helpers.subprocess.run", return_value=_mock_proc("not-json", "", 0)):
        data, err = apr_cmd_json(["inspect", "model.apr", "--json"])
    assert data is None
    assert "invalid JSON" in err
    assert "inspect" in err


# ── llamacpp_run ────────────────────────────────────────────────


def test_llamacpp_run_success():
    with patch("helpers.LLAMACPP_BIN", "/usr/bin/llama-completion"), patch(
        "helpers.subprocess.run", return_value=_mock_proc("  hello world  ", "", 0)
    ):
        data, err = llamacpp_run("model.gguf", "hi")
    assert err is None
    assert data == {"text": "hello world"}


def test_llamacpp_run_no_binary():
    with patch("helpers.LLAMACPP_BIN", ""):
        data, err = llamacpp_run("model.gguf", "hi")
    assert data is None
    assert "not found" in err


def test_llamacpp_run_nonzero_exit():
    with patch("helpers.LLAMACPP_BIN", "/usr/bin/llama-completion"), patch(
        "helpers.subprocess.run", return_value=_mock_proc("", "model load failed", 1)
    ):
        data, err = llamacpp_run("model.gguf", "hi")
    assert data is None
    assert "exit 1" in err


def test_llamacpp_run_timeout():
    with patch("helpers.LLAMACPP_BIN", "/usr/bin/llama-completion"), patch(
        "helpers.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="llama-completion", timeout=120),
    ):
        data, err = llamacpp_run("model.gguf", "hi")
    assert data is None
    assert "TIMEOUT" in err


def test_llamacpp_run_file_not_found():
    with patch("helpers.LLAMACPP_BIN", "/usr/bin/llama-completion"), patch(
        "helpers.subprocess.run", side_effect=FileNotFoundError
    ):
        data, err = llamacpp_run("model.gguf", "hi")
    assert data is None
    assert "not found" in err


# ── count_char_mismatches ───────────────────────────────────────


def test_count_char_mismatches_identical():
    assert count_char_mismatches("hello", "hello") == 0


def test_count_char_mismatches_all_different():
    assert count_char_mismatches("abc", "xyz") == 3


def test_count_char_mismatches_different_lengths():
    assert count_char_mismatches("hello", "he") == 3  # 0 char diffs + 3 length diff


def test_count_char_mismatches_partial():
    assert count_char_mismatches("abcd", "abxd") == 1
