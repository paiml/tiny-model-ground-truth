# Benchmark Configuration

## Test Execution Environment

| Parameter | Value |
|-----------|-------|
| CPU | Any x86_64 or ARM64 |
| RAM | ≥4GB |
| GPU | Not required |
| OS | Linux (Ubuntu 22.04+ recommended) |
| Rust | 1.84+ |
| Python | 3.11+ |
| apr | 0.2.16+ |
| ruchy | 4.0.0+ |
| transformers | 5.1.0 |
| torch | 2.10.0 |

## Benchmark Parameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| max_new_tokens | 32 | Sufficient to detect divergence while keeping test fast |
| temperature | 0.0 | Deterministic greedy decoding |
| do_sample | false | No stochastic sampling |
| dtype (oracle) | float32 | Maximum precision baseline |
| device | cpu | Avoids GPU non-determinism |
| num_prompts | 4 | Covers arithmetic, NLP, code, social categories |
| num_models | 3 | Covers LLaMA, Qwen/GQA, GPT-2 architectures |
| num_runs | 1 | Deterministic tests require only 1 run |

## Confidence Intervals

All tests are deterministic (greedy decoding with temperature=0). Given identical inputs and model weights, outputs are bit-for-bit identical across runs. Therefore:

- **Confidence level**: 100% (deterministic)
- **Variance**: 0 (no stochastic component)
- **Required replications**: 1 (single run is sufficient)
- **Statistical test**: Not applicable (deterministic comparison)

## Effect Sizes

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| Int4 token mismatch | ≤5/32 (15.6%) | Maximum acceptable 4-bit quantization drift |
| Int8 token mismatch | ≤3/32 (9.4%) | Maximum acceptable 8-bit quantization drift |
| PPL difference | <0.5 | Negligible perplexity degradation |
| Cross-runtime divergence | 0% | Same GGUF + greedy = identical output |
| Format roundtrip loss | 0% | Lossless conversion expected |

## Sample Size Justification

- **Models** (n=3): Exhaustive over the tiny model roster. Covers 3 distinct architectures (LLaMA-style, GQA, GPT-2), representing the primary architecture families supported by apr.
- **Prompts** (n=4): Categorical coverage of tokenization patterns. Arithmetic (numbers/operators), NLP (factual completion), Code (Python syntax), Social (name generation).
- **Quantization levels** (n=2): Int4 and Int8 span the precision range of interest. FP16 excluded as it's lossless in practice.
- **Total test cases**: 75 across 6 test suites.

## Timing

| Phase | Expected Duration |
|-------|------------------|
| `make pull` | ~2 min (network dependent, ~1.5GB) |
| `make convert` | ~1 min (9 model files) |
| `make test` | <1 sec (75 ruchy tests) |
| `make oracle` | ~5 min (3 model loads + inference) |
| Full CI (`make ci`) | ~4 min |
