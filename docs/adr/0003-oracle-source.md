# ADR-003: Oracle Source

## Status
Accepted (2026-02-13)

## Context
A ground truth oracle is needed to compare quantized model
outputs against. Options include HuggingFace transformers,
llama.cpp reference, or ONNX Runtime.

## Decision
HuggingFace `transformers` with float32 precision, CPU-only,
greedy decode serves as the ground truth oracle.

## Consequences
- **Reproducibility**: Oracle JSON includes
  `transformers_version` (5.1.0) and `torch_version` (2.10.0)
  for exact reproducibility.
- **Versioning**: Oracle files are committed to version control.
  Regeneration: `uv run python scripts/gen_oracle.py --all`.
- **Baseline**: Float32 CPU is the highest-precision reference.
  All quantization comparisons measure drift from this baseline.
- **Risk**: Bugs in `transformers` become our ground truth.
  Mitigated by `transformers` being the industry standard.
