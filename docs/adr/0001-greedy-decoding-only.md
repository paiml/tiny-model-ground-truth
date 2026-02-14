# ADR-001: Greedy Decoding Only

## Status
Accepted (2026-02-13)

## Context
Model inference can use greedy decoding (deterministic) or sampling-based decoding
(stochastic). Sampling requires random seeds, multiple runs, and statistical analysis.
Greedy decoding produces identical output for identical inputs.

## Decision
All tests use greedy decoding (temperature=0, `do_sample=False`).

## Consequences
- **Positive**: Tests are fully deterministic. No random seed management needed. Single
  run is sufficient. Confidence intervals are trivially 100%.
- **Negative**: Cannot detect sampling-related bugs. Acceptable because inference
  parity is the primary hypothesis.
- **Sample size**: n=12 per claim (3 models x 4 prompts) is exhaustive over the
  roster, not a statistical sample.
