# Design Document (Pre-Registration)

**Date**: 2026-02-13
**Status**: Pre-registered before implementation

## Pre-Registration

This design document was authored and committed before the test suite was executed. It constitutes a pre-registration of the experimental design, hypotheses, and analysis plan per Popperian falsification methodology.

### Hypotheses (Pre-Registered)

1. **Canary Parity**: Int8 APR inference matches float32 oracle text exactly
2. **Token Bound**: Int4 ≤5/32 mismatches, Int8 ≤3/32 mismatches
3. **Drift Ordering**: Int8 mismatches ≤ Int4 mismatches + 1
4. **Roundtrip**: APR→GGUF→APR produces identical tokens
5. **Runtime Parity**: `apr` matches `llama-cli` on same GGUF
6. **PPL Bound**: Per-model ceiling, |Int4-Int8| < 0.5

### Analysis Plan

- No statistical analysis needed (deterministic greedy decoding)
- No data exclusion criteria
- All results reported
- Deviations logged in CHANGELOG.md

### Sample Size Justification

| Dimension | N | Justification |
|-----------|---|---------------|
| Models | 3 | Exhaustive over roster (LLaMA, Qwen/GQA, GPT-2) |
| Prompts | 4 | Categorical coverage (math, NLP, code, social) |
| Quant levels | 2 | Int4 (lower) and Int8 (higher) precision |
| Test cases | 87 | Exhaustive cross-product + property tests |

### Confidence Intervals

All tests are deterministic (greedy decoding, temperature=0).
- **CI**: 100% — outputs are bit-for-bit identical across runs
- **Variance**: 0 — no stochastic component
- **Replications**: 1 required (deterministic)
- **p-value**: Not applicable (no null distribution)

### Effect Sizes

| Metric | Threshold | Cohen's d equivalent |
|--------|-----------|---------------------|
| Int4 mismatch rate | 15.6% (5/32) | Large (above chance) |
| Int8 mismatch rate | 9.4% (3/32) | Medium |
| PPL drift | 0.5 points | Small (negligible practical significance) |
| Canary/roundtrip/runtime | 0% divergence | N/A (binary pass/fail) |

### Benchmark Configuration

- **CPU**: Any x86_64/ARM64 (no GPU needed)
- **RAM**: ≥4GB
- **Duration**: <1 sec for test suite, ~4 min for full CI
- **Reproducibility**: Docker, Nix flake, uv.lock, .tool-versions

### Archival

- **Version control**: git with signed commits
- **Lock files**: uv.lock (Python), .tool-versions (Rust/Python versions)
- **Oracle provenance**: transformers 5.1.0, torch 2.10.0 (recorded in each JSON)
- **DOI**: To be assigned upon first public release
- **Archive**: Repository will be archived on Zenodo for long-term availability
