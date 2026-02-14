# Benchmark Configuration

## Test Execution Environment

| Parameter | Value |
|-----------|-------|
| CPU | Any x86_64 or ARM64 |
| RAM | >=4GB |
| GPU | Not required (CUDA optional for GPU oracle) |
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
| dtype (GPU oracle) | bfloat16, float16 | Precision ladder variants |
| device | cpu (default), cuda (GPU oracle) | CPU avoids non-determinism; GPU tests precision drift |
| num_prompts | 4 | Covers arithmetic, NLP, code, social categories |
| num_models | 3 | Covers LLaMA, Qwen/GQA, GPT-2 architectures |
| num_runs | 1 | Deterministic tests require only 1 run |

## Sample Size Justification

- **Models**: n = 3 (exhaustive over the tiny model roster). Covers 3 distinct architectures (LLaMA-style, GQA, GPT-2).
- **Prompts**: n = 4 (categorical coverage of tokenization patterns). Arithmetic, NLP, Code, Social.
- **Quantization levels**: n = 2 (Int4 and Int8 span the precision range).
- **Total parity checks**: n = 59 across 5 check suites.
- **Total pytest tests**: n = 69 (59 parity + 6 property-based + 4 unit).
- **Hypothesis iterations**: n = 100 per property-based test (configurable).

## Statistical Properties

All parity checks are deterministic (greedy decoding, temperature = 0):

- **Standard deviation**: σ = 0 (no stochastic component)
- **Measurement uncertainty**: ±0 (bit-for-bit identical across runs)
- **Confidence interval**: [exact, exact] — trivially 100%
- **p-value**: Not applicable (binary pass/fail, no null distribution)
- **Required replications**: 1 (deterministic)

## Timing

| Phase | Expected Duration (±σ) |
|-------|------------------------|
| `make pull` | ~2 min ±1 min (network dependent, ~1.5GB) |
| `make convert` | ~1 min ±30s (9 model files) |
| `make check` | ~2 min ±30s (59 checks, each invokes apr inference) |
| `make test` | ~3s ±1s (property-based tests only, no apr needed) |
| `make oracle` | ~5 min ±2 min (3 model loads + inference) |
| `make oracle-gpu` | ~3 min ±1 min (GPU inference, 2 precisions × 3 models) |
| Full CI (`make ci`) | ~5 min ±2 min |
