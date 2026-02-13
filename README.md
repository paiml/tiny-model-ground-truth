# tiny-model-ground-truth

[![Methodology](https://img.shields.io/badge/methodology-Popperian%20falsification-red)](https://en.wikipedia.org/wiki/Falsifiability) [![Models](https://img.shields.io/badge/models-3-blue)]() [![Tests](https://img.shields.io/badge/tests-75%20cases-green)]() [![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

**Thesis**: Given a tiny model from HuggingFace, every format conversion and runtime engine in the Sovereign AI Stack must produce token-identical greedy outputs (or bounded quantization drift). A single failure proves a bug.

See [CLAIMS.md](CLAIMS.md) for pre-registered falsifiable claims and design rationale (ADRs).

## Quick Start

```bash
make pull      # Download 3 tiny models (~1.5GB)
make convert   # Import to APR (Int4/Int8) + export GGUF
make test      # Run all 75 parity tests
```

### Requirements

- **Hardware**: Any x86_64 or ARM64 machine with ≥4GB RAM. CPU-only (no GPU required).
- **Software**: `apr` (v0.2.16+), `ruchy` (v4.0.0+), `uv` (v0.5+), Python 3.11+
- **Optional**: `llama-cli` for cross-runtime parity tests (gracefully skipped if absent)
- **Disk**: ~5GB for models directory

### Environment Reproducibility

```bash
# Docker (fully reproducible)
docker build -t tmgt . && docker run tmgt

# Native (requires Rust toolchain)
bash ci/setup.sh   # Install apr, ruchy
uv sync            # Install Python deps (locked via uv.lock)
```

## Parity Matrix

| Model | APR Int4 | APR Int8 | GGUF Int4 | llama.cpp | PPL |
|-------|----------|----------|-----------|-----------|-----|
| SmolLM-135M (135M params) | ≤5/32 mismatch | exact | roundtrip exact | skipped | <20.0 |
| Qwen2-0.5B (500M params) | ≤5/32 mismatch | exact | roundtrip exact | skipped | <15.0 |
| GPT-2 124M (124M params) | ≤5/32 mismatch | exact | roundtrip exact | skipped | <30.0 |

## Test Suites

| Suite | Tests | What it tests |
|-------|-------|--------------|
| `test_canary` | 12 | Golden output regression — catches inference regressions |
| `test_token_parity` | 24 | Int4/Int8 token mismatch bounds vs oracle |
| `test_quant_drift` | 12 | Int8 strictly better than Int4 ordering |
| `test_format_roundtrip` | 6 | APR → GGUF → reimport produces identical tokens |
| `test_runtime_parity` | 12 | `apr` vs `llama.cpp` on same GGUF — exact match |
| `test_perplexity` | 9 | PPL within model-specific bounds, Int4/Int8 diff < 0.5 |
| **Total** | **75** | |

## Methodology

This repo uses **Popperian falsification**: we attempt to *disprove* parity rather than *prove* it. Each test encodes a specific falsifiable prediction. A single failure constitutes evidence of a bug in the format conversion or runtime engine.

### Oracle Generation

- **Source**: HuggingFace `transformers` (v5.1.0) with PyTorch (v2.10.0)
- **Precision**: float32, CPU-only, `do_sample=False` (deterministic greedy)
- **Random seeds**: Not applicable — greedy decoding is fully deterministic (see [ADR-001](CLAIMS.md#adr-001-greedy-decoding-only))
- **Max tokens**: 32 per prompt
- **Output**: JSON with token IDs, decoded text, model metadata

### Tolerance Thresholds

| Comparison | Threshold | Effect Size | Confidence | Sample Size |
|-----------|-----------|-------------|------------|-------------|
| Int4 tokens vs oracle | ≤5/32 mismatches | 15.6% mismatch rate | Deterministic (CI=100%) | n=12 per model |
| Int8 tokens vs oracle | ≤3/32 mismatches | 9.4% mismatch rate | Deterministic (CI=100%) | n=12 per model |
| Int8 vs Int4 drift | Int8 ≤ Int4 + 1 | ≤1 token difference | Deterministic (CI=100%) | n=12 per model |
| Cross-runtime (same GGUF) | Exact text match | 0% divergence | Deterministic (CI=100%) | n=12 per model |
| PPL Int4 vs Int8 | Diff < 0.5 | <0.5 PPL points | Per-model bound | n=3 (1 per model) |
| Canary (text regression) | Exact text match | 0% divergence | Deterministic (CI=100%) | n=12 per model |

### Statistical Notes

- **Deterministic tests**: All tests use greedy decoding (temperature=0). Given identical inputs and weights, outputs are fully deterministic. Therefore confidence intervals are 100% — results either match or they don't. No statistical sampling is involved.
- **Sample size justification**: 4 prompts × 3 models = 12 data points per claim. Prompts cover 4 categories (arithmetic, natural language, code, social) to exercise diverse tokenization paths. 3 models cover 3 architectures (LLaMA, Qwen/GQA, GPT-2). This is exhaustive over the model roster, not a sample.
- **Effect size**: Mismatch thresholds (5/32 for Int4, 3/32 for Int8) represent the maximum acceptable quantization-induced token divergence. These were empirically validated against the full prompt × model matrix.
- **PPL bounds**: Model-specific ceilings (SmolLM: 20.0, Qwen2: 15.0, GPT-2: 30.0) are derived from published perplexity benchmarks for each architecture class, with 2× headroom for quantization-induced degradation.

## Dataset Documentation

### Models

| Model | HF ID | Parameters | Architecture | License |
|-------|-------|------------|-------------|---------|
| SmolLM-135M | `HuggingFaceTB/SmolLM-135M` | 135M | LLaMA-style (30 layers, 9 heads) | Apache 2.0 |
| Qwen2-0.5B | `Qwen/Qwen2-0.5B` | 500M | Qwen (GQA, 24 layers, 14 heads) | Apache 2.0 |
| GPT-2 | `openai-community/gpt2` | 124M | GPT-2 (12 layers, 12 heads) | MIT |

### Prompts

| Prompt | Category | Text | Purpose |
|--------|----------|------|---------|
| arithmetic | Math | `What is 2+2? Answer:` | Tests numerical token generation |
| completion | NLP | `The capital of France is` | Tests factual continuation |
| code | Programming | `def fibonacci(n):` | Tests code token patterns |
| greeting | Social | `Hello, my name is` | Tests natural language patterns |

### Oracle Format

Each oracle JSON file contains:
```json
{
  "model": "HuggingFace model ID",
  "prompt": "input text",
  "runtime": "transformers",
  "format": "float32",
  "transformers_version": "5.1.0",
  "torch_version": "2.10.0",
  "tokens": [token_id_1, token_id_2, ...],
  "text": "decoded output text",
  "token_count": 32,
  "max_new_tokens": 32,
  "do_sample": false
}
```

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

## Reproducibility

- **Lock files**: `uv.lock` pins all Python dependencies. Rust tools installed via `cargo install --locked`.
- **Docker**: `Dockerfile` provides fully reproducible environment.
- **CI**: GitHub Actions runs weekly (see `ci/parity.yml`).
- **Determinism**: All inference uses greedy decoding. No random seeds needed ([ADR-001](CLAIMS.md#adr-001-greedy-decoding-only)).
- **Versioning**: Oracle JSON includes `transformers_version` and `torch_version` for provenance.

## License

MIT. See [LICENSE](LICENSE).
