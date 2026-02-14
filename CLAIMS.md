# Falsifiable Claims

Pre-registered on 2026-02-13, before test execution.

## Hypothesis

**H0 (Null)**: Format conversions and runtime engines in the Sovereign AI Stack produce
outputs that diverge beyond bounded quantization drift from the HuggingFace transformers
oracle.

**H1 (Alternative)**: All format conversions and runtime engines produce token-identical
greedy outputs (or bounded quantization drift within specified tolerances).

A single test failure constitutes evidence for H0 (a bug exists).

## Falsifiable Claims

### Claim 1: Int8 Canary Parity
- **Statement**: Int8-quantized APR models produce text identical to the float32 HuggingFace
  transformers oracle for all 4 prompts across all 3 models.
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
- **Statement**: Int8 (higher precision) produces fewer or equal token mismatches vs oracle
  compared to Int4 (lower precision), with at most 1 extra mismatch allowed.
- **Falsification**: Any test in `parity_check.py (drift suite)` where `int8_mismatches > int4_mismatches + 1`.
- **Threshold**: Int8 ≤ Int4 + 1.
- **Confidence interval**: The +1 margin accounts for autoregressive error propagation where
  a single early mismatch cascades differently across precision levels.
- **Sample size**: 12 test cases (3 models x 4 prompts).

### Claim 5: Format Roundtrip Losslessness
- **Statement**: APR → GGUF → reimport to APR produces token-identical outputs for the same prompt.
- **Falsification**: Any `assert_eq(roundtripped.tokens, original.tokens)` failure in
  `parity_check.py (roundtrip suite)`.
- **Threshold**: Exact token match (0 tolerance).
- **Sample size**: 6 test cases (3 models x 2 prompts).

### Claim 6: Cross-Runtime Parity (Deferred)
- **Statement**: `apr` and `llama.cpp` produce identical text output when running the same
  GGUF model with deterministic greedy decoding (temperature=0).
- **Status**: Deferred. Not implemented in parity checker. Requires `apr run` to produce
  correct output first (blocked by aprender#239).
- **Threshold**: Exact text match.
- **Sample size**: 12 test cases (3 models x 4 prompts), when implemented.

### Claim 7: Perplexity Bounds
- **Statement**: Model perplexity stays within architecture-specific ceilings: SmolLM <20.0, Qwen2 <15.0, GPT-2 <30.0.
- **Falsification**: Any `assert(result.perplexity < ceiling)` failure in `parity_check.py (ppl suite)`.
- **Threshold**: Model-specific ceilings based on known architecture capacity.
- **Sample size**: 6 test cases (3 models x 2 quantization levels).

### Claim 8: Perplexity Drift Bound
- **Statement**: The perplexity difference between Int4 and Int8 quantization of the same model is less than 0.5.
- **Falsification**: Any `assert(diff < 0.5)` failure in `parity_check.py (ppl suite)`.
- **Threshold**: |PPL_Int4 - PPL_Int8| < 0.5.
- **Confidence interval**: 0.5 PPL points represents the maximum expected
  quantization-induced perplexity shift for sub-1B parameter models.
- **Sample size**: 3 test cases (1 per model).

### Claim 9: Metadata Consistency
- **Statement**: `apr inspect` output matches expected architecture parameters (architecture, num_layers, num_heads, hidden_size, vocab_size) for all models at both quantization levels.
- **Falsification**: Any field mismatch between `apr inspect --json` output and MODEL_METADATA in `parity_check.py`.
- **Threshold**: Exact match on all fields.
- **Sample size**: 6 test cases (3 models x 2 quantization levels).

### Claim 10: Model Integrity
- **Statement**: `apr validate` passes magic, header, and version checks for all APR model files.
- **Falsification**: Any `status != PASS` in `apr validate --json` output.
- **Threshold**: All validation checks pass (0 tolerance).
- **Sample size**: 6 test cases (3 models x 2 quantization levels).

### Claim 11: Tensor Structure
- **Statement**: `apr tensors` returns a non-empty tensor list for all APR model files, confirming correct internal structure.
- **Falsification**: Empty or missing tensor list from `apr tensors --json`.
- **Threshold**: tensor_count > 0.
- **Sample size**: 6 test cases (3 models x 2 quantization levels).

### Claim 12: Lint Compliance
- **Statement**: `apr lint` reports no critical-severity violations for any APR model file.
- **Falsification**: Any violation with `severity == "critical"` in `apr lint --json` output.
- **Threshold**: 0 critical violations (warnings acceptable).
- **Sample size**: 6 test cases (3 models x 2 quantization levels).

### Claim 13: Pipeline Self-Test
- **Statement**: `apr check` passes at least 7 of 10 pipeline stages (Embedding, RoPE, QKV, Attention, MLP, LayerNorm, LM Head, etc.) for all models.
- **Falsification**: Any model where `passed_stages < 7` in `apr check --json` output.
- **Threshold**: >= 7/10 stages pass.
- **Sample size**: 6 test cases (3 models x 2 quantization levels).

### Claim 14: Quantization Diff Consistency
- **Statement**: `apr diff` between Int8 and Int4 variants of the same model shows only dtype/size differences, not structural tensor mismatches.
- **Falsification**: Any difference classified as structural (missing tensors, shape mismatches) rather than dtype/size.
- **Threshold**: 0 structural differences.
- **Sample size**: 3 test cases (1 per model).

### Claim 15: Architecture Tree
- **Statement**: `apr tree` reports layer counts matching expected architecture parameters for all models.
- **Falsification**: Any `num_layers` mismatch between `apr tree --json` output and MODEL_METADATA.
- **Threshold**: Exact match on layer count.
- **Sample size**: 6 test cases (3 models x 2 quantization levels).

### Claim 16: Model Identification
- **Statement**: `apr oracle` correctly identifies the architecture family (llama, qwen2, gpt2) for all models at both quantization levels.
- **Falsification**: Any `architecture` field not containing the expected family string.
- **Threshold**: Architecture substring match.
- **Sample size**: 6 test cases (3 models x 2 quantization levels).

### Claim 17: Tensor Data Quality
- **Statement**: Quantized tensors contain non-zero data as verified by `apr hex --stats`, confirming successful quantization write.
- **Falsification**: Any tensor with `std == 0` (constant or all-zero data).
- **Threshold**: std > 0 for sampled tensors.
- **Sample size**: 6 test cases (3 models x 2 quantization levels).

### Claim 18: Model Health
- **Statement**: `apr debug` reports healthy status for all APR model files, confirming file integrity and loadability.
- **Falsification**: Any `health != OK` in `apr debug --json` output, or presence of error fields.
- **Threshold**: health == OK (or no error indicators).
- **Sample size**: 6 test cases (3 models x 2 quantization levels).

### Claim 19: Throughput Minimum
- **Statement**: `apr bench` produces throughput > 0 tok/s for all models, confirming the inference pipeline executes end-to-end.
- **Falsification**: Any `tokens_per_second == 0` or bench crash.
- **Threshold**: tok/s > 0.
- **Status**: Currently expected to fail (0 tok/s bug) — documents the defect.
- **Sample size**: 6 test cases (3 models x 2 quantization levels).

### Claim 20: QA Gate
- **Statement**: `apr qa` executes at least 3 quality gates without critical failure for all models.
- **Falsification**: Fewer than 3 gates executed, or any gate with `severity == "critical"` and `status == "FAIL"`.
- **Threshold**: >= 3 gates executed, 0 critical failures.
- **Sample size**: 6 test cases (3 models x 2 quantization levels).

## Design Decisions (ADR)

### ADR-001: Greedy Decoding Only
- **Decision**: All tests use greedy decoding
  (temperature=0, do_sample=False).
- **Rationale**: Greedy decoding is deterministic, enabling
  exact reproducibility without seed management. Sampling-based
  decoding would require statistical testing over many runs,
  increasing test time from seconds to hours.
- **Consequences**: Tests cannot detect sampling-related bugs.
  Acceptable because inference parity is the primary hypothesis.

### ADR-002: Fixed Prompt Set
- **Decision**: 4 fixed prompts covering arithmetic, completion,
  code, and greeting patterns.
- **Rationale**: Prompts exercise different tokenization paths
  (numbers, natural language, code syntax, names). Sample size
  of 4 prompts x 3 models = 12 data points per claim balances
  coverage against execution time.
- **Consequences**: Cannot detect prompt-specific edge cases.
  Mitigated by choosing diverse prompt categories.

### ADR-003: Oracle Source
- **Decision**: HuggingFace `transformers` with float32, CPU,
  greedy decode as ground truth.
- **Rationale**: `transformers` is the reference implementation
  for all 3 model architectures. Float32 CPU mode avoids GPU
  non-determinism. The oracle is generated once and committed
  to version control.
- **Consequences**: Bugs in `transformers` become our ground
  truth. Acceptable because `transformers` is the industry
  standard.

### ADR-004: Int4/Int8 Instead of Q4K/Q6K
- **Decision**: Use `int4` and `int8` quantization instead of
  `q4k`/`q6k`.
- **Rationale**: `apr import --quantize` supports
  `int4, int8, fp16`. Q4K is only available via
  `--preserve-q4k` for pre-quantized GGUF imports. Int4/Int8
  provide the same precision gradient for drift testing.
- **Consequences**: Tolerance thresholds may differ from
  Q4K/Q6K. Thresholds are pre-registered; empirical validation
  pending (see Filed Issues in README).
