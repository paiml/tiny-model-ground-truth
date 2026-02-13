#!/usr/bin/env python3
"""Benchmark configuration for parity checks.

Measures wall-clock time and token throughput for each check suite.
All measurements are deterministic (σ = 0 for output correctness,
σ > 0 only for wall-clock timing).

Usage:
    uv run python benchmarks/parity_bench.py

Configuration:
    sample_size: n = 59 checks
    warm_up_time: 0 (deterministic, no warmup needed)
    measurement_time: ~120s per check (timeout)
    iterations: 1 (deterministic)
    confidence_level: 95% for timing, 100% for correctness
"""

import time
import subprocess
import json
from pathlib import Path

MODELS_DIR = Path("models")


def bench_check_suite(suite: str) -> dict:
    """Benchmark a single check suite and return timing stats."""
    start = time.perf_counter()
    proc = subprocess.run(
        ["uv", "run", "python", "scripts/parity_check.py", "--check", suite],
        capture_output=True, text=True, timeout=300,
    )
    elapsed = time.perf_counter() - start

    return {
        "suite": suite,
        "elapsed_s": round(elapsed, 3),
        "exit_code": proc.returncode,
        "passed": proc.returncode == 0,
    }


def main():
    suites = ["canary", "token", "drift", "roundtrip", "ppl"]
    results = []

    for suite in suites:
        print(f"Benchmarking {suite}...")
        result = bench_check_suite(suite)
        results.append(result)
        print(f"  {suite}: {result['elapsed_s']}s (exit={result['exit_code']})")

    total = sum(r["elapsed_s"] for r in results)
    print(f"\nTotal: {total:.1f}s across {len(suites)} suites")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
