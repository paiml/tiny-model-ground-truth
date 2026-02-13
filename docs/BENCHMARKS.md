# Benchmark Configuration

## Test Execution Environment

| Parameter | Value |
|-----------|-------|
| CPU | Any x86_64 or ARM64 |
| RAM | >=4GB |
| GPU | Not required |
| OS | Linux (Ubuntu 22.04+ recommended) |
| Rust | 1.84+ |
| Python | 3.11+ |
| apr | 0.2.16+ |
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

## Sample Size Justification

- **Models** (n=3): Exhaustive over the tiny model roster. Covers 3 distinct architectures (LLaMA-style, GQA, GPT-2).
- **Prompts** (n=4): Categorical coverage of tokenization patterns. Arithmetic, NLP, Code, Social.
- **Quantization levels** (n=2): Int4 and Int8 span the precision range.
- **Total checks**: 59 across 5 check suites.

## Timing

| Phase | Expected Duration |
|-------|------------------|
| `make pull` | ~2 min (network dependent, ~1.5GB) |
| `make convert` | ~1 min (9 model files) |
| `make check` | ~2 min (59 checks, each invokes apr inference) |
| `make oracle` | ~5 min (3 model loads + inference) |
| Full CI (`make ci`) | ~5 min |
