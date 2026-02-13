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

Total sample: n = 59 checks (exhaustive cross-product, not sampled).

| Dimension | N | Justification |
|-----------|---|---------------|
| Models | n = 3 | Exhaustive over roster (LLaMA, Qwen/GQA, GPT-2) |
| Prompts | n = 4 | Categorical coverage (math, NLP, code, social) |
| Quant levels | n = 2 | Int4 (lower) and Int8 (higher) precision |
| Canary checks | n = 12 | 3 models × 4 prompts |
| Token checks | n = 24 | 3 models × 4 prompts × 2 quant levels |
| Drift checks | n = 12 | 3 models × 4 prompts |
| Roundtrip checks | n = 6 | 3 models × 2 prompts |
| PPL checks | n = 9 | 3 models × 2 quant levels + 3 drift |
| **Total** | **n = 59** | **Exhaustive cross-product across 5 suites** |

### Confidence Intervals and Statistical Properties

All tests are deterministic (greedy decoding, temperature = 0, do_sample = False).

- **Confidence interval**: [exact, exact] — 100% CI, outputs are bit-for-bit identical across runs
- **Standard deviation**: σ = 0 — no stochastic component, variance = 0
- **Error bars**: ±0 — deterministic, no measurement uncertainty
- **Replications**: 1 required (deterministic, additional runs produce identical output)
- **p-value**: Not applicable (no null distribution; binary pass/fail)
- **Power analysis**: Not applicable (deterministic tests; effect is either present or absent)

### Effect Sizes

| Metric | Threshold | Mismatch Rate | Cohen's d | Interpretation |
|--------|-----------|---------------|-----------|----------------|
| Int4 tokens vs oracle | ≤5/32 | 15.6% (±0) | Large | Above chance baseline |
| Int8 tokens vs oracle | ≤3/32 | 9.4% (±0) | Medium | Bounded quantization drift |
| PPL drift (Int4 - Int8) | <0.5 PPL | — | Small | Negligible practical significance |
| Canary (text regression) | 0/32 | 0% (±0) | N/A | Binary exact match |
| Roundtrip (APR→GGUF→APR) | 0/32 | 0% (±0) | N/A | Binary exact match |

### Benchmark Configuration

- **CPU**: Any x86_64/ARM64 (no GPU needed)
- **RAM**: ≥4GB
- **Duration**: ~2 min for check suite (σ ≈ 30s depending on CPU), ~5 min for full CI
- **Reproducibility**: Docker, Nix flake, uv.lock, .devcontainer

### Archival

- **Version control**: git with signed commits
- **Lock files**: uv.lock (Python), .tool-versions (Rust/Python versions)
- **Oracle provenance**: transformers 5.1.0, torch 2.10.0 (recorded in each JSON)
- **DOI**: To be assigned upon first public release
- **Archive**: Repository will be archived on Zenodo for long-term availability
