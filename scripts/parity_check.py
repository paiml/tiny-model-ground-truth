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
            ["apr"] + args,
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
            r.pass_(f"exact match")
        else:
            r.fail(
                f"text mismatch",
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
                    f"Higher precision produced MORE mismatches",
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
        f"**Oracle**: transformers 5.1.0, torch 2.10.0, float32, CPU, greedy",
        "",
        f"### {len(failures)} failure(s)",
        "",
    ]
    for f in failures:
        lines.append(f"#### `{f.name}`")
        lines.append(f"**Error**: {f.error}")
        if f.details:
            lines.append(f"```")
            lines.append(f.details)
            lines.append(f"```")
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
    parser.add_argument(
        "--check",
        choices=["canary", "token", "drift", "roundtrip", "ppl", "all"],
        default="all",
        help="Which check to run",
    )
    args = parser.parse_args()

    models_to_test = (
        {args.model: MODELS[args.model]} if args.model else MODELS
    )

    all_results: list[Result] = []

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
            "canary": lambda: check_canary(slug, info),
            "token": lambda: check_token_parity(slug, info),
            "drift": lambda: check_quant_drift(slug, info),
            "roundtrip": lambda: check_roundtrip(slug, info),
            "ppl": lambda: check_perplexity(slug, info),
        }

        if args.check == "all":
            run_checks = checks
        else:
            run_checks = {args.check: checks[args.check]}

        for check_name, check_fn in run_checks.items():
            results = check_fn()
            all_results.extend(results)
            for r in results:
                status = "PASS" if r.passed else "FAIL"
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
        print(f"  TICKET (copy to aprender issue)")
        print(f"{'='*60}")
        print(format_ticket(failures))

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
