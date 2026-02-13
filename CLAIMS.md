# Falsifiable Claims

Pre-registered on 2026-02-13, before test execution.

## Hypothesis

**H0 (Null)**: Format conversions and runtime engines in the Sovereign AI Stack produce outputs that diverge beyond bounded quantization drift from the HuggingFace transformers oracle.

**H1 (Alternative)**: All format conversions and runtime engines produce token-identical greedy outputs (or bounded quantization drift within specified tolerances).

A single test failure constitutes evidence for H0 (a bug exists).

## Falsifiable Claims

### Claim 1: Int8 Canary Parity
- **Statement**: Int8-quantized APR models produce text identical to the float32 HuggingFace transformers oracle for all 4 prompts across all 3 models.
- **Falsification**: Any `assert_eq(result.text, oracle.text)` failure in `parity_check.py (canary suite)`.
- **Threshold**: Exact text match (0 tolerance).
- **Sample size**: 12 test cases (3 models x 4 prompts).

### Claim 2: Int4 Token Parity Bound
- **Statement**: Int4-quantized models produce at most 5 token mismatches per 32 generated tokens vs the oracle.
- **Falsification**: Any test in `parity_check.py (token suite)` where `mismatches > 5`.
- **Threshold**: ≤5 mismatches per 32 tokens.
- **Effect size**: 15.6% mismatch rate (5/32) is the maximum acceptable quantization drift for Int4.
- **Sample size**: 12 test cases (3 models x 4 prompts).

### Claim 3: Int8 Token Parity Bound
- **Statement**: Int8-quantized models produce at most 3 token mismatches per 32 generated tokens vs the oracle.
- **Falsification**: Any test in `parity_check.py (token suite)` where `mismatches > 3`.
- **Threshold**: ≤3 mismatches per 32 tokens.
- **Effect size**: 9.4% mismatch rate (3/32) is the maximum acceptable quantization drift for Int8.
- **Sample size**: 12 test cases (3 models x 4 prompts).

### Claim 4: Quantization Drift Ordering
- **Statement**: Int8 (higher precision) produces fewer or equal token mismatches vs oracle compared to Int4 (lower precision), with at most 1 extra mismatch allowed.
- **Falsification**: Any test in `parity_check.py (drift suite)` where `int8_mismatches > int4_mismatches + 1`.
- **Threshold**: Int8 ≤ Int4 + 1.
- **Confidence interval**: The +1 margin accounts for autoregressive error propagation where a single early mismatch cascades differently across precision levels.
- **Sample size**: 12 test cases (3 models x 4 prompts).

### Claim 5: Format Roundtrip Losslessness
- **Statement**: APR → GGUF → reimport to APR produces token-identical outputs for the same prompt.
- **Falsification**: Any `assert_eq(roundtripped.tokens, original.tokens)` failure in `parity_check.py (roundtrip suite)`.
- **Threshold**: Exact token match (0 tolerance).
- **Sample size**: 6 test cases (3 models x 2 prompts).

### Claim 6: Cross-Runtime Parity
- **Statement**: `apr` and `llama.cpp` produce identical text output when running the same GGUF model with deterministic greedy decoding (temperature=0).
- **Falsification**: Any `assert_eq` failure in `parity_check.py (runtime suite)`.
- **Threshold**: Exact text match.
- **Sample size**: 12 test cases (3 models x 4 prompts). Skipped gracefully if llama-cli unavailable.

### Claim 7: Perplexity Bounds
- **Statement**: Model perplexity stays within architecture-specific ceilings: SmolLM <20.0, Qwen2 <15.0, GPT-2 <30.0.
- **Falsification**: Any `assert(result.perplexity < ceiling)` failure in `parity_check.py (ppl suite)`.
- **Threshold**: Model-specific ceilings based on known architecture capacity.
- **Sample size**: 6 test cases (3 models x 2 quantization levels).

### Claim 8: Perplexity Drift Bound
- **Statement**: The perplexity difference between Int4 and Int8 quantization of the same model is less than 0.5.
- **Falsification**: Any `assert(diff < 0.5)` failure in `parity_check.py (ppl suite)`.
- **Threshold**: |PPL_Int4 - PPL_Int8| < 0.5.
- **Confidence interval**: 0.5 PPL points represents the maximum expected quantization-induced perplexity shift for sub-1B parameter models.
- **Sample size**: 3 test cases (1 per model).

## Design Decisions (ADR)

### ADR-001: Greedy Decoding Only
- **Decision**: All tests use greedy decoding (temperature=0, do_sample=False).
- **Rationale**: Greedy decoding is deterministic, enabling exact reproducibility without seed management. Sampling-based decoding would require statistical testing over many runs, increasing test time from seconds to hours.
- **Consequences**: Tests cannot detect sampling-related bugs. Acceptable because inference parity is the primary hypothesis.

### ADR-002: Fixed Prompt Set
- **Decision**: 4 fixed prompts covering arithmetic, completion, code, and greeting patterns.
- **Rationale**: Prompts are designed to exercise different tokenization paths (numbers, natural language, code syntax, names). Sample size of 4 prompts x 3 models = 12 data points per claim balances coverage against execution time.
- **Consequences**: Cannot detect prompt-specific edge cases. Mitigated by choosing diverse prompt categories.

### ADR-003: Oracle Source
- **Decision**: HuggingFace `transformers` with float32, CPU, greedy decode as ground truth.
- **Rationale**: `transformers` is the reference implementation for all 3 model architectures. Float32 CPU mode avoids GPU non-determinism. The oracle is generated once and committed to version control.
- **Consequences**: Bugs in `transformers` become our ground truth. Acceptable because `transformers` is the industry standard.

### ADR-004: Int4/Int8 Instead of Q4K/Q6K
- **Decision**: Use `int4` and `int8` quantization instead of `q4k`/`q6k`.
- **Rationale**: `apr import --quantize` supports `int4, int8, fp16`. Q4K is only available via `--preserve-q4k` for pre-quantized GGUF imports. Int4/Int8 provide the same precision gradient for drift testing.
- **Consequences**: Tolerance thresholds may differ from Q4K/Q6K. Thresholds are pre-registered; empirical validation pending (see Filed Issues in README).
