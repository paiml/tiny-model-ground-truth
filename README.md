# tiny-model-ground-truth

[![Methodology](https://img.shields.io/badge/methodology-Popperian%20falsification-red)](https://en.wikipedia.org/wiki/Falsifiability) [![Models](https://img.shields.io/badge/models-3-blue)]() [![Tests](https://img.shields.io/badge/tests-6%20suites-green)]()

**Thesis**: Given a tiny model from HuggingFace, every format conversion and runtime engine in the Sovereign AI Stack must produce token-identical greedy outputs (or bounded quantization drift). A single failure proves a bug.

## Quick Start

```bash
make pull      # Download 3 tiny models (~1.5GB)
make convert   # Import to APR (Q4K/Q6K) + export GGUF
make test      # Run all parity tests
```

## Parity Matrix

| Model | APR Q4K | APR Q6K | GGUF Q4K | llama.cpp | PPL |
|-------|---------|---------|----------|-----------|-----|
| SmolLM-135M | - | - | - | - | - |
| Qwen2-0.5B | - | - | - | - | - |
| GPT-2 124M | - | - | - | - | - |

## Test Suites

| Suite | What it tests |
|-------|--------------|
| `test_canary` | Golden output regression — catches inference regressions |
| `test_token_parity` | Q4K/Q6K token mismatch bounds vs oracle |
| `test_quant_drift` | Q6K strictly better than Q4K ordering |
| `test_format_roundtrip` | APR → GGUF → reimport produces identical tokens |
| `test_runtime_parity` | `apr` vs `llama.cpp` on same GGUF — exact match |
| `test_perplexity` | PPL within model-specific bounds, Q4K/Q6K diff < 0.5 |

## Methodology

This repo uses **Popperian falsification**: we attempt to *disprove* parity rather than *prove* it. Each test encodes a specific falsifiable prediction. A single failure constitutes evidence of a bug in the format conversion or runtime engine.

- **Oracle**: HuggingFace `transformers` float32 CPU greedy decode (ground truth)
- **Prompts**: 4 fixed deterministic prompts, 32 max new tokens
- **Tolerance**: Quantization drift bounded by precision tier (Q4K ≤ 5, Q6K ≤ 3 mismatches per 32 tokens)
- **Cross-runtime**: Deterministic greedy on same GGUF must produce exact text match

## Architecture

```
Python (uv)                    Ruchy Tests
───────────                    ───────────
gen_oracle.py ──► oracle/*.json ◄── test_canary.ruchy
  (rare, manual)                    test_token_parity.ruchy
                                    test_quant_drift.ruchy
                apr CLI             test_format_roundtrip.ruchy
                ───────             test_runtime_parity.ruchy
                pull, import,       test_perplexity.ruchy
                export, run,
                eval, diff
```
