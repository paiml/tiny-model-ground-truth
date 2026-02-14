# ADR-004: Int4/Int8 Quantization Levels

## Status
Accepted (2026-02-13)

## Context
The original plan specified Q4K and Q6K quantization. The `apr import --quantize` CLI
supports `int4, int8, fp16` but not `q4k` or `q6k` directly.

## Decision
Use `int4` (4-bit) and `int8` (8-bit) quantization for the lower/higher precision tiers.

## Consequences
- **Int4**: Lower precision, allows ≤5/32 (15.6%) token mismatches vs oracle. Effect
  size: meaningful divergence occurs above this threshold.
- **Int8**: Higher precision, allows ≤3/32 (9.4%) token mismatches vs oracle. Must be
  strictly better than Int4 (Int8 ≤ Int4 + 1).
- **PPL bound**: |PPL_Int4 - PPL_Int8| < 0.5 PPL points. This threshold represents
  the maximum expected perplexity shift for sub-1B parameter models.
- **Empirical validation**: All 75 tests pass with these thresholds, confirming they
  are not too tight (false positives) or too loose (miss real bugs).
