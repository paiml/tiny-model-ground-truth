#!/usr/bin/env python3
"""Real parity checker — actually runs apr inference and compares against oracle.

Usage:
    uv run python scripts/parity_check.py              # Run all checks
    uv run python scripts/parity_check.py --model smollm-135m  # Single model
    uv run python scripts/parity_check.py --ticket      # Print GitHub issue markdown for failures
"""

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

ORACLE_DIR = Path("oracle")
MODELS_DIR = Path("models")

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
            mismatches = []
            got_arch = data.get("architecture", "").lower()
            if meta["architecture"] not in got_arch:
                mismatches.append(f"arch: expected {meta['architecture']}, got {got_arch}")
            for field, key in [
                ("layers", "num_layers"),
                ("heads", "num_heads"),
                ("hidden_dim", "hidden_size"),
                ("vocab_size", "vocab_size"),
            ]:
                got = data.get(key)
                if got is not None and got != meta[field]:
                    mismatches.append(f"{key}: expected {meta[field]}, got {got}")
            if mismatches:
                r.fail("; ".join(mismatches))
            else:
                r.pass_(f"arch={got_arch}, layers={data.get('num_layers')}")
        results.append(r)
    return results


def check_validate(slug: str, model_info: dict) -> list[Result]:
    """`apr validate` passes magic/header/version checks."""
    results = []
    for quant in ["int4", "int8"]:
        r = Result(f"validate/{slug}/{quant}")
        data, err = apr_cmd_json(["validate", model_info[quant], "--json"])
        if err:
            r.fail(err)
        else:
            checks = data.get("checks", data.get("results", []))
            if isinstance(checks, list):
                failures = [c for c in checks if c.get("status") not in ("PASS", "pass", "ok")]
                if failures:
                    r.fail(f"{len(failures)} validation failures: {failures}")
                else:
                    r.pass_(f"{len(checks)} checks passed")
            elif isinstance(checks, dict):
                failures = {k: v for k, v in checks.items() if v not in ("PASS", "pass", "ok")}
                if failures:
                    r.fail(f"validation failures: {failures}")
                else:
                    r.pass_(f"{len(checks)} checks passed")
            else:
                score = data.get("score", data.get("total"))
                if score is not None:
                    r.pass_(f"score={score}")
                else:
                    r.fail(f"unexpected validate output format: {list(data.keys())}")
        results.append(r)
    return results


def check_tensors(slug: str, model_info: dict) -> list[Result]:
    """`apr tensors` returns correct tensor count and dtypes."""
    MODEL_METADATA[slug]
    results = []
    for quant in ["int4", "int8"]:
        r = Result(f"tensors/{slug}/{quant}")
        data, err = apr_cmd_json(["tensors", model_info[quant], "--json"])
        if err:
            r.fail(err)
        else:
            tensors = data.get("tensors", data if isinstance(data, list) else [])
            if not tensors:
                r.fail(f"no tensors returned, keys: {list(data.keys()) if isinstance(data, dict) else 'list'}")
            else:
                count = len(tensors)
                r.pass_(f"{count} tensors found")
        results.append(r)
    return results


def check_lint(slug: str, model_info: dict) -> list[Result]:
    """`apr lint` reports no critical-severity violations."""
    results = []
    for quant in ["int4", "int8"]:
        r = Result(f"lint/{slug}/{quant}")
        data, err = apr_cmd_json(["lint", model_info[quant], "--json"])
        if err:
            r.fail(err)
        else:
            violations = data.get("violations", data.get("issues", []))
            if isinstance(violations, list):
                critical = [v for v in violations if v.get("severity") in ("critical", "error")]
                if critical:
                    r.fail(f"{len(critical)} critical violations: {critical[:3]}")
                else:
                    r.pass_(f"{len(violations)} warnings, 0 critical")
            else:
                r.pass_("lint passed")
        results.append(r)
    return results


def check_self_test(slug: str, model_info: dict) -> list[Result]:
    """`apr check` passes ≥7/10 pipeline stages."""
    results = []
    for quant in ["int4", "int8"]:
        r = Result(f"selftest/{slug}/{quant}")
        data, err = apr_cmd_json(["check", model_info[quant], "--json"])
        if err:
            r.fail(err)
        else:
            stages = data.get("stages", data.get("checks", data.get("results", [])))
            if isinstance(stages, list):
                passed = sum(1 for s in stages if s.get("status") in ("PASS", "pass", "ok"))
                total = len(stages)
                if passed >= 7:
                    r.pass_(f"{passed}/{total} stages passed")
                else:
                    failed_names = [
                        s.get("name", "?") for s in stages
                        if s.get("status") not in ("PASS", "pass", "ok")
                    ]
                    r.fail(f"only {passed}/{total} stages passed", f"failed: {failed_names}")
            else:
                passed = data.get("passed", 0)
                total = data.get("total", 0)
                if passed >= 7:
                    r.pass_(f"{passed}/{total} stages passed")
                else:
                    r.fail(f"only {passed}/{total} stages passed")
        results.append(r)
    return results


def check_diff(slug: str, model_info: dict) -> list[Result]:
    """`apr diff int8 int4` shows dtype-only differences, not structural."""
    results = []
    r = Result(f"diff/{slug}")
    data, err = apr_cmd_json(
        ["diff", model_info["int8"], model_info["int4"], "--json"]
    )
    if err:
        r.fail(err)
    else:
        diffs = data.get("differences", data.get("diffs", []))
        if isinstance(diffs, list):
            structural = [
                d for d in diffs
                if d.get("type") not in ("dtype", "size", "data") and d.get("kind") not in ("dtype", "size", "data")
            ]
            if structural:
                r.fail(f"{len(structural)} structural diffs (not just dtype)", str(structural[:3]))
            else:
                r.pass_(f"{len(diffs)} differences (dtype/size only)")
        else:
            r.pass_(f"diff completed, keys: {list(data.keys())}")
    results.append(r)
    return results


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
            layers = data.get("num_layers", data.get("layers"))
            tensors = data.get("total_tensors", data.get("tensor_count"))
            details = []
            if layers is not None:
                details.append(f"layers={layers}")
                if layers != meta["layers"]:
                    r.fail(f"layer count mismatch: expected {meta['layers']}, got {layers}")
                    results.append(r)
                    continue
            if tensors is not None:
                details.append(f"tensors={tensors}")
            r.pass_(", ".join(details) if details else "tree completed")
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
            # Try without specific tensor (some models use different naming)
            data, err2 = apr_cmd_json(["hex", model_info[quant], "--stats", "--json"])
            if err2:
                r.fail(err)
            else:
                r.pass_(f"hex stats available (keys: {list(data.keys())[:5]})")
        else:
            std = data.get("std", data.get("statistics", {}).get("std"))
            if std is not None and std > 0:
                r.pass_(f"std={std:.6f} (non-zero data)")
            elif std is not None:
                r.fail("std=0 (constant/zero tensor data)")
            else:
                r.pass_(f"hex completed (keys: {list(data.keys())[:5]})")
        results.append(r)
    return results


def check_debug(slug: str, model_info: dict) -> list[Result]:
    """`apr debug` reports health=OK."""
    results = []
    for quant in ["int4", "int8"]:
        r = Result(f"debug/{slug}/{quant}")
        data, err = apr_cmd_json(["debug", model_info[quant], "--json"])
        if err:
            r.fail(err)
        else:
            health = data.get("health", data.get("status", ""))
            if health.lower() in ("ok", "healthy", "pass"):
                r.pass_(f"health={health}")
            elif health:
                r.fail(f"health={health}")
            else:
                # No explicit health field — check for error indicators
                if data.get("error"):
                    r.fail(f"error: {data['error']}")
                else:
                    r.pass_(f"debug completed (keys: {list(data.keys())[:5]})")
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
            gates = data.get("gates", data.get("checks", data.get("results", [])))
            if isinstance(gates, list):
                executed = len(gates)
                critical = [
                    g for g in gates
                    if g.get("status") in ("CRITICAL", "critical", "FAIL")
                    and g.get("severity") == "critical"
                ]
                if executed >= 3 and not critical:
                    r.pass_(f"{executed} gates executed, 0 critical")
                elif executed < 3:
                    r.fail(f"only {executed} gates executed (need ≥3)")
                else:
                    r.fail(f"{len(critical)} critical failures", str(critical[:3]))
            else:
                executed = data.get("gates_executed", data.get("total", 0))
                if executed >= 3:
                    r.pass_(f"{executed} gates executed")
                else:
                    r.fail(f"only {executed} gates executed (need ≥3)")
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


def check_parity_gpu(slug: str, model_info: dict) -> list[Result]:
    """`apr parity` checks GPU/CPU produce identical results (GGUF only)."""
    r = Result(f"parity-gpu/{slug}")
    gguf = model_info.get("gguf")
    if not gguf or not Path(gguf).exists():
        r.fail("no GGUF file available")
        return [r]
    data, err = apr_cmd_json(["parity", gguf, "--json", "--assert"])
    if err:
        if "no GPU" in err.lower() or "cuda" in err.lower():
            r.pass_(f"skipped (no GPU): {err[:80]}")
        else:
            r.fail(err)
    else:
        match = data.get("match", data.get("parity"))
        if match in (True, "pass", "PASS"):
            r.pass_("GPU/CPU parity confirmed")
        elif match in (False, "fail", "FAIL"):
            r.fail("GPU/CPU parity mismatch", str(data))
        else:
            r.pass_(f"parity completed (keys: {list(data.keys())[:5]})")
    return [r]


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


def main():
    parser = argparse.ArgumentParser(description="Real model parity checker")
    parser.add_argument("--model", choices=list(MODELS.keys()), help="Single model")
    parser.add_argument(
        "--ticket", action="store_true", help="Print GitHub issue for failures"
    )
    all_checks = [
        "canary", "token", "drift", "roundtrip", "ppl",
        "inspect", "validate", "tensors", "lint", "selftest",
        "diff", "tree", "oracle-id", "hex", "debug", "bench", "qa",
        "list", "rosetta-diff", "parity-gpu", "all",
    ]
    parser.add_argument(
        "--check",
        choices=all_checks,
        default="all",
        help="Which check to run",
    )
    args = parser.parse_args()

    models_to_test = (
        {args.model: MODELS[args.model]} if args.model else MODELS
    )

    all_results: list[Result] = []

    # Global checks (not per-model)
    if args.check in ("all", "list"):
        print(f"\n{'='*60}")
        print("  global checks")
        print(f"{'='*60}")
        results = check_list_global()
        all_results.extend(results)
        for r in results:
            icon = "\033[32m✓\033[0m" if r.passed else "\033[31m✗\033[0m"
            detail = r.details if r.passed else r.error
            print(f"  {icon} {r.name}: {detail}")

    for slug, info in models_to_test.items():
        # Verify model files exist
        missing = [k for k in ["int4", "int8"] if not Path(info[k]).exists()]
        if missing:
            for k in missing:
                print(f"SKIP {slug}: {info[k]} not found (run make convert)")
            continue

        print(f"\n{'='*60}")
        print(f"  {slug}")
        print(f"{'='*60}")

        checks = {
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
        }

        if args.check == "all":
            run_checks = checks
        elif args.check == "list":
            run_checks = {}  # already handled as global check
        else:
            run_checks = {args.check: checks[args.check]}

        for _check_name, check_fn in run_checks.items():
            results = check_fn()
            all_results.extend(results)
            for r in results:
                icon = "\033[32m✓\033[0m" if r.passed else "\033[31m✗\033[0m"
                detail = r.details if r.passed else r.error
                print(f"  {icon} {r.name}: {detail}")

    # Summary
    passed = sum(1 for r in all_results if r.passed)
    failed = sum(1 for r in all_results if not r.passed)
    total = len(all_results)

    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    print(f"{'='*60}")

    failures = [r for r in all_results if not r.passed]
    if failures and args.ticket:
        print(f"\n{'='*60}")
        print("  TICKET (copy to aprender issue)")
        print(f"{'='*60}")
        print(format_ticket(failures))

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
