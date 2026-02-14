# ADR-002: Fixed Prompt Set

## Status
Accepted (2026-02-13)

## Context
The number and type of test prompts affects coverage and execution time. More prompts
increase confidence but increase test duration proportionally.

## Decision
4 fixed prompts covering arithmetic, completion, code, and greeting patterns. Each prompt generates 32 tokens.

## Consequences
- **Sample size justification**: 4 prompts x 3 models = 12 data points per claim. This
  is exhaustive over the model roster, covering 4 tokenization categories (numbers,
  natural language, code syntax, names).
- **Effect size**: Mismatch thresholds (5/32=15.6% for Int4, 3/32=9.4% for Int8)
  represent the maximum acceptable quantization-induced divergence.
- **Confidence interval**: Deterministic tests have CI=100%. Each test either passes
  or fails with no variance.
- **Limitation**: Cannot detect prompt-specific edge cases outside the 4 categories.
