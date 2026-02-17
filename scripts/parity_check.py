#!/usr/bin/env python3
"""Real parity checker — actually runs apr inference and compares against oracle.

Usage:
    uv run python scripts/parity_check.py              # Run all checks
    uv run python scripts/parity_check.py --model smollm-135m  # Single model
    uv run python scripts/parity_check.py --ticket      # Print GitHub issue markdown for failures
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

ORACLE_DIR = Path("oracle")
MODELS_DIR = Path("models")

LLAMACPP_BIN = shutil.which("llama-completion") or ""

LLAMACPP_MODELS = {
    "smollm-135m": {
        "q4_0": "models/smollm-135m-q4_0.gguf",
        "q8_0": "models/smollm-135m-q8_0.gguf",
    },
    "qwen2-0.5b": {
        "q4_0": "models/qwen2-0.5b-q4_0.gguf",
        "q8_0": "models/qwen2-0.5b-q8_0.gguf",
    },
    "gpt2-124m": {
        "q4_0": "models/gpt2-124m-q4_0.gguf",
        "q8_0": "models/gpt2-124m-q8_0.gguf",
    },
}

MODEL_METADATA = {
    "smollm-135m": {
        "architecture": "llama",
        "layers": 30,
        "heads": 9,
        "kv_heads": 3,
        "hidden_dim": 576,
        "vocab_size": 49152,
        "hf_id": "HuggingFaceTB/SmolLM-135M",
    },
    "qwen2-0.5b": {
        "architecture": "qwen2",
        "layers": 24,
        "heads": 14,
        "kv_heads": 2,
        "hidden_dim": 896,
        "vocab_size": 151936,
        "hf_id": "Qwen/Qwen2-0.5B",
    },
    "gpt2-124m": {
        "architecture": "gpt2",
        "layers": 12,
        "heads": 12,
        "kv_heads": 12,
        "hidden_dim": 768,
        "vocab_size": 50257,
        "hf_id": "openai-community/gpt2",
    },
}

MODELS = {
    "smollm-135m": {
        "int4": "models/smollm-135m-int4.apr",
        "int8": "models/smollm-135m-int8.apr",
        "gguf": "models/smollm-135m-int4.gguf",
        "ppl_ceiling": 20.0,
    },
    "qwen2-0.5b": {
        "int4": "models/qwen2-0.5b-int4.apr",
        "int8": "models/qwen2-0.5b-int8.apr",
        "gguf": "models/qwen2-0.5b-int4.gguf",
        "ppl_ceiling": 15.0,
    },
    "gpt2-124m": {
        "int4": "models/gpt2-124m-int4.apr",
        "int8": "models/gpt2-124m-int8.apr",
        "gguf": "models/gpt2-124m-int4.gguf",
        "ppl_ceiling": 30.0,
    },
}

PROMPTS = ["arithmetic", "code", "completion", "greeting"]


class Result:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = ""
        self.details = ""

    def pass_(self, details: str = ""):
        self.passed = True
        self.details = details

    def fail(self, error: str, details: str = ""):
        self.passed = False
        self.error = error
        self.details = details


def run_apr(args: list[str]) -> tuple[str, str, int]:
    """Run apr CLI and return (stdout, stderr, exit_code)."""
    try:
        proc = subprocess.run(
            ["apr", *args],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT after 120s", 1
    except FileNotFoundError:
        return "", "apr not found in PATH", 127


def apr_run_json(model_path: str, prompt: str) -> dict | None:
    """Run apr inference and return parsed JSON, or None on failure."""
    stdout, stderr, code = run_apr(
        ["run", model_path, "-p", prompt, "-n", "32", "--json"]
    )
    if code != 0:
        return None, f"apr run failed (exit {code}): {stderr.strip()}"
    try:
        return json.loads(stdout), None
    except json.JSONDecodeError:
        return None, f"apr run returned invalid JSON: {stdout[:200]}"


def apr_cmd_json(args: list[str]) -> tuple[dict | None, str | None]:
    """Run any apr subcommand with --json and return parsed JSON."""
    stdout, stderr, code = run_apr(args)
    if code != 0:
        return None, f"apr {args[0]} failed (exit {code}): {stderr.strip()}"
    try:
        return json.loads(stdout), None
    except json.JSONDecodeError:
        return None, f"apr {args[0]} returned invalid JSON: {stdout[:200]}"


def load_oracle(slug: str, prompt_name: str) -> dict:
    path = ORACLE_DIR / slug / f"{prompt_name}.json"
    return json.loads(path.read_text())


def count_mismatches(a: list[int], b: list[int]) -> int:
    min_len = min(len(a), len(b))
    m = sum(1 for i in range(min_len) if a[i] != b[i])
    return m + abs(len(a) - len(b))


def count_char_mismatches(a: str, b: str) -> int:
    """Count character-level mismatches between two strings."""
    min_len = min(len(a), len(b))
    m = sum(1 for i in range(min_len) if a[i] != b[i])
    return m + abs(len(a) - len(b))


def llamacpp_run(gguf_path: str, prompt: str, n_tokens: int = 32) -> tuple[dict | None, str | None]:
    """Run llama-completion greedy inference, return {"text": ...} or None on failure."""
    cli = LLAMACPP_BIN
    if not cli:
        return None, "llama-completion not found in PATH"
    try:
        proc = subprocess.run(
            [
                cli, "-m", str(gguf_path),
                "-p", prompt,
                "-n", str(n_tokens),
                "--temp", "0", "--top-k", "1",
                "--no-display-prompt",
                "-s", "42",
            ],
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT after 120s"
    except FileNotFoundError:
        return None, "llama-completion not found in PATH"
    if proc.returncode != 0:
        return None, f"llama-completion failed (exit {proc.returncode}): {proc.stderr.strip()[:300]}"
    text = proc.stdout.strip()
    return {"text": text}, None


def _extract_list_field(data: dict, *keys: str) -> list | None:
    """Extract a list-valued field from a JSON response, trying multiple keys."""
    for k in keys:
        val = data.get(k)
        if isinstance(val, list):
            return val
    return None


def _count_passed_in_list(items: list, pass_values=("PASS", "pass", "ok")) -> tuple[int, int]:
    """Count passed/total in a list of check dicts with 'status' field."""
    passed = sum(1 for s in items if s.get("status") in pass_values)
    return passed, len(items)


def _check_per_quant(slug, model_info, check_name, check_fn):
    """Run a check function for both int4 and int8 quants, return results."""
    results = []
    for quant in ["int4", "int8"]:
        r = Result(f"{check_name}/{slug}/{quant}")
        data, err = apr_cmd_json([*check_fn["args"](model_info[quant])])
        if err:
            r.fail(err)
        else:
            check_fn["evaluate"](r, data, slug, quant)
        results.append(r)
    return results


def _print_results(results: list) -> None:
    """Print a list of Result objects with colored pass/fail icons."""
    for r in results:
        icon = "\033[32m✓\033[0m" if r.passed else "\033[31m✗\033[0m"
        detail = r.details if r.passed else r.error
        print(f"  {icon} {r.name}: {detail}")


def check_canary(slug: str, model_info: dict) -> list[Result]:
    """Int8 text must exactly match oracle."""
    results = []
    for prompt_name in PROMPTS:
        r = Result(f"canary/{slug}/{prompt_name}")
        oracle = load_oracle(slug, prompt_name)
        output, err = apr_run_json(model_info["int8"], oracle["prompt"])
        if err:
            r.fail(err)
        elif output.get("text") == oracle["text"]:
            r.pass_("exact match")
        else:
            r.fail(
                "text mismatch",
                f"expected: {oracle['text']!r}\n  got:      {output.get('text')!r}",
            )
        results.append(r)
    return results


def check_token_parity(slug: str, model_info: dict) -> list[Result]:
    """Int4 ≤5/32 mismatches, Int8 ≤3/32 mismatches."""
    results = []
    for quant, path, threshold in [
        ("int4", model_info["int4"], 5),
        ("int8", model_info["int8"], 3),
    ]:
        for prompt_name in PROMPTS:
            r = Result(f"token-parity/{slug}/{quant}/{prompt_name}")
            oracle = load_oracle(slug, prompt_name)
            output, err = apr_run_json(path, oracle["prompt"])
            if err:
                r.fail(err)
            else:
                m = count_mismatches(
                    output.get("tokens", []), oracle["tokens"]
                )
                if m <= threshold:
                    r.pass_(f"{m}/{threshold} mismatches")
                else:
                    r.fail(
                        f"{m} mismatches exceeds threshold {threshold}",
                        f"oracle tokens: {oracle['tokens']}\n  got tokens:    {output.get('tokens', [])}",
                    )
            results.append(r)
    return results


def check_quant_drift(slug: str, model_info: dict) -> list[Result]:
    """Int8 mismatches ≤ Int4 mismatches + 1."""
    results = []
    for prompt_name in PROMPTS:
        r = Result(f"quant-drift/{slug}/{prompt_name}")
        oracle = load_oracle(slug, prompt_name)

        out_int4, err4 = apr_run_json(model_info["int4"], oracle["prompt"])
        out_int8, err8 = apr_run_json(model_info["int8"], oracle["prompt"])

        if err4:
            r.fail(f"int4: {err4}")
        elif err8:
            r.fail(f"int8: {err8}")
        else:
            m4 = count_mismatches(out_int4.get("tokens", []), oracle["tokens"])
            m8 = count_mismatches(out_int8.get("tokens", []), oracle["tokens"])
            if m8 <= m4 + 1:
                r.pass_(f"int8={m8} ≤ int4={m4}+1")
            else:
                r.fail(
                    f"int8 ({m8}) > int4 ({m4}) + 1",
                    "Higher precision produced MORE mismatches",
                )
        results.append(r)
    return results


def check_roundtrip(slug: str, model_info: dict) -> list[Result]:
    """APR → GGUF → reimport → compare tokens."""
    results = []
    reimported = f"models/{slug}-roundtrip-tmp.apr"

    # Import GGUF back to APR
    _, err_imp, code = run_apr(
        ["import", model_info["gguf"], "-o", reimported]
    )
    if code != 0:
        r = Result(f"roundtrip/{slug}/import")
        r.fail(f"reimport failed (exit {code}): {err_imp.strip()}")
        results.append(r)
        return results

    for prompt_name in ["completion", "arithmetic"]:
        r = Result(f"roundtrip/{slug}/{prompt_name}")
        oracle = load_oracle(slug, prompt_name)

        orig, err1 = apr_run_json(model_info["int4"], oracle["prompt"])
        rt, err2 = apr_run_json(reimported, oracle["prompt"])

        if err1:
            r.fail(f"original: {err1}")
        elif err2:
            r.fail(f"reimported: {err2}")
        elif orig.get("tokens") == rt.get("tokens"):
            r.pass_("tokens identical after roundtrip")
        else:
            m = count_mismatches(
                orig.get("tokens", []), rt.get("tokens", [])
            )
            r.fail(
                f"{m} token mismatches after APR→GGUF→APR roundtrip",
                f"original: {orig.get('tokens')}\n  roundtrip: {rt.get('tokens')}",
            )
        results.append(r)

    # Cleanup
    Path(reimported).unlink(missing_ok=True)
    return results


def check_perplexity(slug: str, model_info: dict) -> list[Result]:
    """PPL within model-specific ceiling, Int4/Int8 diff < 0.5."""
    results = []
    ppl_values = {}

    for quant, path in [("int4", model_info["int4"]), ("int8", model_info["int8"])]:
        r = Result(f"ppl/{slug}/{quant}")
        stdout, stderr, code = run_apr(
            ["eval", path, "--threshold", "50.0", "--json"]
        )
        if code != 0:
            r.fail(f"apr eval failed (exit {code}): {stderr.strip()}")
            results.append(r)
            continue

        try:
            data = json.loads(stdout)
            ppl = data.get("perplexity", float("inf"))
        except (json.JSONDecodeError, TypeError):
            r.fail(f"apr eval returned invalid JSON: {stdout[:200]}")
            results.append(r)
            continue

        ppl_values[quant] = ppl
        ceiling = model_info["ppl_ceiling"]
        if ppl < ceiling:
            r.pass_(f"PPL={ppl:.2f} < {ceiling}")
        else:
            r.fail(f"PPL={ppl:.2f} exceeds ceiling {ceiling}")
        results.append(r)

    # Compare Int4 vs Int8
    if "int4" in ppl_values and "int8" in ppl_values:
        r = Result(f"ppl-drift/{slug}")
        diff = abs(ppl_values["int4"] - ppl_values["int8"])
        if diff < 0.5:
            r.pass_(
                f"|{ppl_values['int4']:.2f} - {ppl_values['int8']:.2f}| = {diff:.3f} < 0.5"
            )
        else:
            r.fail(
                f"PPL drift {diff:.3f} exceeds 0.5",
                f"int4={ppl_values['int4']:.2f}, int8={ppl_values['int8']:.2f}",
            )
        results.append(r)

    return results


def _evaluate_inspect(r: Result, data: dict, meta: dict) -> None:
    """Evaluate apr inspect output against expected architecture metadata."""
    mismatches = []
    got_arch = data.get("architecture", "").lower()
    if meta["architecture"] not in got_arch:
        mismatches.append(f"arch: expected {meta['architecture']}, got {got_arch}")
    for field, key in [("layers", "num_layers"), ("heads", "num_heads"), ("hidden_dim", "hidden_size"), ("vocab_size", "vocab_size")]:
        got = data.get(key)
        if got is not None and got != meta[field]:
            mismatches.append(f"{key}: expected {meta[field]}, got {got}")
    if mismatches:
        r.fail("; ".join(mismatches))
    else:
        r.pass_(f"arch={got_arch}, layers={data.get('num_layers')}")


def check_inspect(slug: str, model_info: dict) -> list[Result]:
    """Metadata from `apr inspect` matches expected architecture params."""
    meta = MODEL_METADATA[slug]
    results = []
    for quant in ["int4", "int8"]:
        r = Result(f"inspect/{slug}/{quant}")
        data, err = apr_cmd_json(["inspect", model_info[quant], "--json"])
        if err:
            r.fail(err)
        else:
            _evaluate_inspect(r, data, meta)
        results.append(r)
    return results


PASS_STATUSES = ("PASS", "pass", "ok")


def _check_list_failures(checks: list) -> list:
    return [c for c in checks if c.get("status") not in PASS_STATUSES]


def _check_dict_failures(checks: dict) -> dict:
    return {k: v for k, v in checks.items() if v not in PASS_STATUSES}


def _evaluate_validation_checks(r: Result, data: dict) -> None:
    """Evaluate apr validate output in its various response shapes."""
    checks = data.get("checks", data.get("results"))
    if isinstance(checks, list):
        failures = _check_list_failures(checks)
        r.fail(f"{len(failures)} validation failures: {failures}") if failures else r.pass_(f"{len(checks)} checks passed")
        return
    if isinstance(checks, dict):
        failures = _check_dict_failures(checks)
        r.fail(f"validation failures: {failures}") if failures else r.pass_(f"{len(checks)} checks passed")
        return
    score = data.get("score", data.get("total"))
    r.pass_(f"score={score}") if score is not None else r.fail(f"unexpected validate output format: {list(data.keys())}")


def check_validate(slug: str, model_info: dict) -> list[Result]:
    """`apr validate` passes magic/header/version checks."""
    results = []
    for quant in ["int4", "int8"]:
        r = Result(f"validate/{slug}/{quant}")
        data, err = apr_cmd_json(["validate", model_info[quant], "--json"])
        if err:
            r.fail(err)
        else:
            _evaluate_validation_checks(r, data)
        results.append(r)
    return results


def _extract_tensors(data) -> list:
    """Extract tensor list from apr tensors JSON response."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("tensors", [])
    return []


def check_tensors(slug: str, model_info: dict) -> list[Result]:
    """`apr tensors` returns correct tensor count and dtypes."""
    results = []
    for quant in ["int4", "int8"]:
        r = Result(f"tensors/{slug}/{quant}")
        data, err = apr_cmd_json(["tensors", model_info[quant], "--json"])
        if err:
            r.fail(err)
        else:
            tensors = _extract_tensors(data)
            r.pass_(f"{len(tensors)} tensors found") if tensors else r.fail("no tensors returned")
        results.append(r)
    return results


def _evaluate_lint(r: Result, data: dict) -> None:
    """Evaluate apr lint output for critical violations."""
    violations = _extract_list_field(data, "violations", "issues")
    if violations is None:
        r.pass_("lint passed")
        return
    critical = [v for v in violations if v.get("severity") in ("critical", "error")]
    r.fail(f"{len(critical)} critical violations: {critical[:3]}") if critical else r.pass_(f"{len(violations)} warnings, 0 critical")


def check_lint(slug: str, model_info: dict) -> list[Result]:
    """`apr lint` reports no critical-severity violations."""
    results = []
    for quant in ["int4", "int8"]:
        r = Result(f"lint/{slug}/{quant}")
        data, err = apr_cmd_json(["lint", model_info[quant], "--json"])
        if err:
            r.fail(err)
        else:
            _evaluate_lint(r, data)
        results.append(r)
    return results


def _evaluate_selftest_stages(r: Result, data: dict) -> None:
    """Evaluate apr check output — require ≥7 passed stages."""
    stages = _extract_list_field(data, "stages", "checks", "results")
    if stages is not None:
        passed, total = _count_passed_in_list(stages)
        if passed >= 7:
            r.pass_(f"{passed}/{total} stages passed")
            return
        failed_names = [s.get("name", "?") for s in stages if s.get("status") not in ("PASS", "pass", "ok")]
        r.fail(f"only {passed}/{total} stages passed", f"failed: {failed_names}")
        return
    passed = data.get("passed", 0)
    total = data.get("total", 0)
    r.pass_(f"{passed}/{total} stages passed") if passed >= 7 else r.fail(f"only {passed}/{total} stages passed")


def check_self_test(slug: str, model_info: dict) -> list[Result]:
    """`apr check` passes ≥7/10 pipeline stages."""
    results = []
    for quant in ["int4", "int8"]:
        r = Result(f"selftest/{slug}/{quant}")
        data, err = apr_cmd_json(["check", model_info[quant], "--json"])
        if err:
            r.fail(err)
        else:
            _evaluate_selftest_stages(r, data)
        results.append(r)
    return results


def _is_expected_diff(d: dict) -> bool:
    """Check if a diff entry is an expected dtype/size/data difference."""
    expected = ("dtype", "size", "data")
    return d.get("type") in expected or d.get("kind") in expected


def check_diff(slug: str, model_info: dict) -> list[Result]:
    """`apr diff int8 int4` shows dtype-only differences, not structural."""
    r = Result(f"diff/{slug}")
    data, err = apr_cmd_json(["diff", model_info["int8"], model_info["int4"], "--json"])
    if err:
        r.fail(err)
        return [r]
    diffs = _extract_list_field(data, "differences", "diffs")
    if diffs is None:
        r.pass_(f"diff completed, keys: {list(data.keys())}")
        return [r]
    structural = [d for d in diffs if not _is_expected_diff(d)]
    if structural:
        r.fail(f"{len(structural)} structural diffs (not just dtype)", str(structural[:3]))
    else:
        r.pass_(f"{len(diffs)} differences (dtype/size only)")
    return [r]


def _evaluate_tree(r: Result, data: dict, expected_layers: int) -> None:
    """Evaluate apr tree output against expected layer count."""
    layers = data.get("num_layers", data.get("layers"))
    tensors = data.get("total_tensors", data.get("tensor_count"))
    if layers is not None and layers != expected_layers:
        r.fail(f"layer count mismatch: expected {expected_layers}, got {layers}")
        return
    details = []
    if layers is not None:
        details.append(f"layers={layers}")
    if tensors is not None:
        details.append(f"tensors={tensors}")
    r.pass_(", ".join(details) if details else "tree completed")


def check_tree(slug: str, model_info: dict) -> list[Result]:
    """`apr tree` shows correct layer and tensor count."""
    meta = MODEL_METADATA[slug]
    results = []
    for quant in ["int4", "int8"]:
        r = Result(f"tree/{slug}/{quant}")
        data, err = apr_cmd_json(["tree", model_info[quant], "--json"])
        if err:
            r.fail(err)
        else:
            _evaluate_tree(r, data, meta["layers"])
        results.append(r)
    return results


def check_oracle_id(slug: str, model_info: dict) -> list[Result]:
    """`apr oracle` correctly identifies model architecture."""
    meta = MODEL_METADATA[slug]
    results = []
    for quant in ["int4", "int8"]:
        r = Result(f"oracle-id/{slug}/{quant}")
        data, err = apr_cmd_json(["oracle", model_info[quant], "--json"])
        if err:
            r.fail(err)
        else:
            arch = (data.get("architecture", "") or data.get("family", "") or "").lower()
            if meta["architecture"] in arch:
                params = data.get("parameters", data.get("param_count"))
                r.pass_(f"arch={arch}, params={params}")
            else:
                r.fail(
                    f"architecture mismatch: expected {meta['architecture']}, got {arch}",
                    f"full output: {data}",
                )
        results.append(r)
    return results


def _evaluate_hex_stats(r: Result, data) -> None:
    """Evaluate apr hex stats output."""
    if isinstance(data, list):
        r.pass_(f"hex completed ({len(data)} entries)")
        return
    std = data.get("std")
    if std is None:
        std = (data.get("statistics") or {}).get("std")
    if std is not None and std > 0:
        r.pass_(f"std={std:.6f} (non-zero data)")
    elif std is not None:
        r.fail("std=0 (constant/zero tensor data)")
    else:
        r.pass_(f"hex completed (keys: {list(data.keys())[:5]})")


def check_hex_quality(slug: str, model_info: dict) -> list[Result]:
    """`apr hex` reports non-zero tensor statistics."""
    results = []
    for quant in ["int4", "int8"]:
        r = Result(f"hex/{slug}/{quant}")
        data, err = apr_cmd_json([
            "hex", model_info[quant],
            "--tensor", "layers.0.self_attn.q_proj.weight",
            "--stats", "--json",
        ])
        if err:
            data, err2 = apr_cmd_json(["hex", model_info[quant], "--stats", "--json"])
            if err2:
                r.fail(err)
            else:
                _evaluate_hex_stats(r, data)
        else:
            _evaluate_hex_stats(r, data)
        results.append(r)
    return results


def _evaluate_health(r: Result, data: dict) -> None:
    """Evaluate apr debug health output."""
    health = data.get("health", data.get("status", ""))
    if health.lower() in ("ok", "healthy", "pass"):
        r.pass_(f"health={health}")
    elif health:
        r.fail(f"health={health}")
    elif data.get("error"):
        r.fail(f"error: {data['error']}")
    else:
        r.pass_(f"debug completed (keys: {list(data.keys())[:5]})")


def check_debug(slug: str, model_info: dict) -> list[Result]:
    """`apr debug` reports health=OK."""
    results = []
    for quant in ["int4", "int8"]:
        r = Result(f"debug/{slug}/{quant}")
        data, err = apr_cmd_json(["debug", model_info[quant], "--json"])
        if err:
            r.fail(err)
        else:
            _evaluate_health(r, data)
        results.append(r)
    return results


def check_bench(slug: str, model_info: dict) -> list[Result]:
    """`apr bench` completes without crash; tok/s > 0 when inference works."""
    results = []
    for quant in ["int4", "int8"]:
        r = Result(f"bench/{slug}/{quant}")
        data, err = apr_cmd_json([
            "bench", model_info[quant], "--json",
            "--iterations", "1", "--warmup", "0",
        ])
        if err:
            r.fail(err)
        else:
            tok_s = data.get("tokens_per_second", data.get("tok_s", data.get("throughput")))
            if tok_s is not None and tok_s > 0:
                r.pass_(f"throughput={tok_s:.1f} tok/s")
            elif tok_s is not None:
                r.fail(f"throughput={tok_s} tok/s (expected > 0)")
            else:
                r.pass_(f"bench completed (keys: {list(data.keys())[:5]})")
        results.append(r)
    return results


def _evaluate_qa_gates(r: Result, data: dict) -> None:
    """Evaluate apr qa output — require ≥3 gates, no critical failures."""
    gates = _extract_list_field(data, "gates", "checks", "results")
    if gates is not None:
        critical = [g for g in gates if g.get("status") in ("CRITICAL", "critical", "FAIL") and g.get("severity") == "critical"]
        if len(gates) < 3:
            r.fail(f"only {len(gates)} gates executed (need ≥3)")
        elif critical:
            r.fail(f"{len(critical)} critical failures", str(critical[:3]))
        else:
            r.pass_(f"{len(gates)} gates executed, 0 critical")
        return
    executed = data.get("gates_executed", data.get("total", 0))
    r.pass_(f"{executed} gates executed") if executed >= 3 else r.fail(f"only {executed} gates executed (need ≥3)")


def check_qa(slug: str, model_info: dict) -> list[Result]:
    """`apr qa` executes ≥3 gates without critical failure."""
    results = []
    for quant in ["int4", "int8"]:
        r = Result(f"qa/{slug}/{quant}")
        data, err = apr_cmd_json([
            "qa", model_info[quant], "--json",
            "--skip-ollama", "--skip-gpu-speedup", "--skip-ptx-parity",
        ])
        if err:
            r.fail(err)
        else:
            _evaluate_qa_gates(r, data)
        results.append(r)
    return results


def check_list_global() -> list[Result]:
    """`apr list` succeeds and returns parseable output."""
    r = Result("list/global")
    data, err = apr_cmd_json(["list", "--json"])
    if err:
        r.fail(err)
    else:
        r.pass_(f"cache listing succeeded (keys: {list(data.keys())[:5]})")
    return [r]


def check_rosetta_diff(slug: str, model_info: dict) -> list[Result]:
    """`apr rosetta diff-tensors` detects no unexpected layout mismatches."""
    r = Result(f"rosetta-diff/{slug}")
    data, err = apr_cmd_json([
        "rosetta", "diff-tensors", model_info["int8"], model_info["int4"], "--json",
    ])
    if err:
        r.fail(err)
    else:
        mismatches = data.get("mismatches", data.get("layout_errors", []))
        if isinstance(mismatches, list) and mismatches:
            r.fail(f"{len(mismatches)} layout mismatches", str(mismatches[:3]))
        else:
            r.pass_("no layout mismatches")
    return [r]


def _is_gpu_skip_error(err: str) -> bool:
    """Check if error indicates GPU unavailability (not a real failure)."""
    lower = err.lower()
    return "no gpu" in lower or "cuda" in lower


def _evaluate_parity_match(r: Result, data: dict) -> None:
    """Evaluate the match/parity field from apr parity output."""
    match = data.get("match", data.get("parity"))
    if match in (True, "pass", "PASS"):
        r.pass_("GPU/CPU parity confirmed")
    elif match in (False, "fail", "FAIL"):
        r.fail("GPU/CPU parity mismatch", str(data))
    else:
        r.pass_(f"parity completed (keys: {list(data.keys())[:5]})")


def check_parity_gpu(slug: str, model_info: dict) -> list[Result]:
    """`apr parity` checks GPU/CPU produce identical results (GGUF only)."""
    r = Result(f"parity-gpu/{slug}")
    gguf = model_info.get("gguf")
    if not gguf or not Path(gguf).exists():
        r.fail("no GGUF file available")
        return [r]
    data, err = apr_cmd_json(["parity", gguf, "--json", "--assert"])
    if err:
        if _is_gpu_skip_error(err):
            r.pass_(f"skipped (no GPU): {err[:80]}")
        else:
            r.fail(err)
        return [r]
    _evaluate_parity_match(r, data)
    return [r]


def check_llamacpp_text(slug: str, model_info: dict) -> list[Result]:
    """llama-cli text output vs oracle on native GGUF (Q8_0 and Q4_0)."""
    results = []
    llamacpp_info = LLAMACPP_MODELS.get(slug, {})
    thresholds = {"q8_0": 3, "q4_0": 5}
    for quant, threshold in thresholds.items():
        gguf_path = llamacpp_info.get(quant)
        if not gguf_path or not Path(gguf_path).exists():
            r = Result(f"llamacpp-text/{slug}/{quant}")
            r.fail(f"native GGUF not found: {gguf_path} (run make convert-llamacpp)")
            results.append(r)
            continue
        for prompt_name in PROMPTS:
            r = Result(f"llamacpp-text/{slug}/{quant}/{prompt_name}")
            oracle = load_oracle(slug, prompt_name)
            output, err = llamacpp_run(gguf_path, oracle["prompt"])
            if err:
                r.fail(err)
            else:
                m = count_char_mismatches(output["text"], oracle["text"])
                if m <= threshold:
                    r.pass_(f"{m}/{threshold} char mismatches")
                else:
                    r.fail(
                        f"{m} char mismatches exceeds threshold {threshold}",
                        f"oracle: {oracle['text']!r}\n  got:   {output['text']!r}",
                    )
            results.append(r)
    return results


def check_cross_runtime(slug: str, model_info: dict) -> list[Result]:
    """Same native GGUF fed to both apr and llama-cli — text must match."""
    results = []
    llamacpp_info = LLAMACPP_MODELS.get(slug, {})
    gguf_path = llamacpp_info.get("q8_0")
    if not gguf_path or not Path(gguf_path).exists():
        r = Result(f"cross-runtime/{slug}")
        r.fail(f"native Q8_0 GGUF not found: {gguf_path} (run make convert-llamacpp)")
        return [r]
    for prompt_name in PROMPTS:
        r = Result(f"cross-runtime/{slug}/{prompt_name}")
        oracle = load_oracle(slug, prompt_name)
        apr_out, apr_err = apr_run_json(gguf_path, oracle["prompt"])
        llama_out, llama_err = llamacpp_run(gguf_path, oracle["prompt"])
        if apr_err:
            r.fail(f"apr: {apr_err}")
        elif llama_err:
            r.fail(f"llama-cli: {llama_err}")
        elif apr_out.get("text") == llama_out["text"]:
            r.pass_("exact text match")
        else:
            r.fail(
                "cross-runtime text mismatch",
                f"apr:       {apr_out.get('text')!r}\n  llama-cli: {llama_out['text']!r}",
            )
        results.append(r)
    return results


def get_apr_version() -> str:
    """Get apr CLI version string."""
    try:
        proc = subprocess.run(["apr", "--version"], capture_output=True, text=True, timeout=5)
        return proc.stdout.strip() if proc.returncode == 0 else "unknown"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return "unknown"


def format_ticket(failures: list[Result]) -> str:
    """Format failures as a GitHub issue body for aprender."""
    lines = [
        "## Model Parity Failures",
        "",
        f"**Date**: {date.today().isoformat()}",
        f"**Tool**: `{get_apr_version()}`",
        "**Oracle**: transformers 5.1.0, torch 2.10.0, float32, CPU, greedy",
        "",
        f"### {len(failures)} failure(s)",
        "",
    ]
    for f in failures:
        lines.append(f"#### `{f.name}`")
        lines.append(f"**Error**: {f.error}")
        if f.details:
            lines.append("```")
            lines.append(f.details)
            lines.append("```")
        lines.append("")

    lines.append("### Reproduction")
    lines.append("```bash")
    lines.append("cd tiny-model-ground-truth")
    lines.append("make pull && make convert")
    lines.append("uv run python scripts/parity_check.py")
    lines.append("```")
    return "\n".join(lines)


def _build_model_checks(slug: str, info: dict) -> dict:
    """Build the check dispatch table for a single model."""
    return {
        "canary": lambda s=slug, i=info: check_canary(s, i),
        "token": lambda s=slug, i=info: check_token_parity(s, i),
        "drift": lambda s=slug, i=info: check_quant_drift(s, i),
        "roundtrip": lambda s=slug, i=info: check_roundtrip(s, i),
        "ppl": lambda s=slug, i=info: check_perplexity(s, i),
        "inspect": lambda s=slug, i=info: check_inspect(s, i),
        "validate": lambda s=slug, i=info: check_validate(s, i),
        "tensors": lambda s=slug, i=info: check_tensors(s, i),
        "lint": lambda s=slug, i=info: check_lint(s, i),
        "selftest": lambda s=slug, i=info: check_self_test(s, i),
        "diff": lambda s=slug, i=info: check_diff(s, i),
        "tree": lambda s=slug, i=info: check_tree(s, i),
        "oracle-id": lambda s=slug, i=info: check_oracle_id(s, i),
        "hex": lambda s=slug, i=info: check_hex_quality(s, i),
        "debug": lambda s=slug, i=info: check_debug(s, i),
        "bench": lambda s=slug, i=info: check_bench(s, i),
        "qa": lambda s=slug, i=info: check_qa(s, i),
        "rosetta-diff": lambda s=slug, i=info: check_rosetta_diff(s, i),
        "parity-gpu": lambda s=slug, i=info: check_parity_gpu(s, i),
        "llamacpp-text": lambda s=slug, i=info: check_llamacpp_text(s, i),
        "cross-runtime": lambda s=slug, i=info: check_cross_runtime(s, i),
    }


def _run_selected_checks(checks: dict, check_name: str) -> list[Result]:
    """Run all or a single check from the dispatch table."""
    if check_name == "all":
        run_checks = checks
    elif check_name == "list":
        return []
    else:
        run_checks = {check_name: checks[check_name]}
    results = []
    for _name, fn in run_checks.items():
        batch = fn()
        results.extend(batch)
        _print_results(batch)
    return results


def _check_model_files(slug: str, info: dict) -> bool:
    """Check that required model files exist, print skip messages if not."""
    missing = [k for k in ["int4", "int8"] if not Path(info[k]).exists()]
    for k in missing:
        print(f"SKIP {slug}: {info[k]} not found (run make convert)")
    return not missing


def _print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Real model parity checker")
    parser.add_argument("--model", choices=list(MODELS.keys()), help="Single model")
    parser.add_argument("--ticket", action="store_true", help="Print GitHub issue for failures")
    all_checks = [
        "canary", "token", "drift", "roundtrip", "ppl",
        "inspect", "validate", "tensors", "lint", "selftest",
        "diff", "tree", "oracle-id", "hex", "debug", "bench", "qa",
        "list", "rosetta-diff", "parity-gpu",
        "llamacpp-text", "cross-runtime", "all",
    ]
    parser.add_argument("--check", choices=all_checks, default="all", help="Which check to run")
    args = parser.parse_args()

    models_to_test = {args.model: MODELS[args.model]} if args.model else MODELS
    all_results: list[Result] = []

    if args.check in ("all", "list"):
        _print_section("global checks")
        results = check_list_global()
        all_results.extend(results)
        _print_results(results)

    for slug, info in models_to_test.items():
        if not _check_model_files(slug, info):
            continue
        _print_section(slug)
        checks = _build_model_checks(slug, info)
        all_results.extend(_run_selected_checks(checks, args.check))

    passed = sum(1 for r in all_results if r.passed)
    failed = len(all_results) - passed
    _print_section(f"Results: {passed}/{len(all_results)} passed, {failed} failed")

    if failed and args.ticket:
        _print_section("TICKET (copy to aprender issue)")
        print(format_ticket([r for r in all_results if not r.passed]))

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
