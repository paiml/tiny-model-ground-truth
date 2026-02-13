# Pre-Registration Document

**Date**: 2026-02-13
**Project**: tiny-model-ground-truth
**Investigators**: PAIML ML Infrastructure Team

## Pre-Registered Hypotheses

This document was written and committed **before** executing the test suite against the Sovereign AI Stack's `apr` CLI. It constitutes a pre-registration of our falsifiable predictions.

### Primary Hypothesis
**H1**: All format conversions (SafeTensors → APR → GGUF → APR) and runtime engines (`apr`, `llama-cli`) in the Sovereign AI Stack produce token-identical greedy outputs for sub-1B parameter models, within bounded quantization drift.

### Null Hypothesis
**H0**: At least one format conversion or runtime engine produces outputs that diverge beyond the specified tolerance bounds, indicating a bug.

### Registered Claims (Pre-Test)

1. Int8 APR produces text identical to float32 oracle (exact match, n=12)
2. Int4 APR produces ≤5/32 token mismatches vs oracle (n=12)
3. Int8 APR produces ≤3/32 token mismatches vs oracle (n=12)
4. Int8 mismatches ≤ Int4 mismatches + 1 for same prompt (n=12)
5. APR → GGUF → APR roundtrip produces identical tokens (n=6)
6. `apr` and `llama-cli` produce identical text on same GGUF (n=12, conditional on llama-cli availability)
7. PPL < model-specific ceiling for each quantization level (n=6)
8. |PPL_Int4 - PPL_Int8| < 0.5 for each model (n=3)

### Analysis Plan

- All tests are deterministic (greedy decoding). No statistical analysis needed.
- A single test failure constitutes falsification of the corresponding claim.
- No data exclusion criteria: all test results are reported.
- No stopping rules: all 75 tests run to completion.

### Deviations from Pre-Registration

Any deviation from the above claims (e.g., relaxing tolerances, excluding models) will be documented in CHANGELOG.md with rationale.
