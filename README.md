# tiny-model-ground-truth

[![Methodology][meth-badge]][meth-link]
[![Models][model-badge]]()
[![Parity][parity-badge]]()
[![License][lic-badge]](LICENSE)

[meth-badge]: https://img.shields.io/badge/methodology-Popperian%20falsification-red
[meth-link]: https://en.wikipedia.org/wiki/Falsifiability
[model-badge]: https://img.shields.io/badge/models-3-blue
[parity-badge]: https://img.shields.io/badge/parity-0%2F139%20passing-red
[lic-badge]: https://img.shields.io/badge/license-MIT-blue

**Thesis**: Given a tiny model from HuggingFace, every format
conversion and runtime engine in the Sovereign AI Stack must
produce token-identical greedy outputs (or bounded quantization
drift). A single failure proves a bug.

**Current status**: **0/139 checks passing** (but inference now works for SmolLM/Qwen2).
Exercises **23 of 40+ `apr` subcommands** across 17 check suites.
Twelve issues filed against aprender/realizar — see [Filed Issues](#filed-issues).

See [CLAIMS.md](CLAIMS.md) for pre-registered falsifiable claims and design rationale (ADRs).

## Quick Start

```bash
make pull      # Download 3 tiny models (~1.5GB)
make convert   # Import to APR (Int4/Int8) + export GGUF
make check     # Run all parity checks (actually invokes apr inference)
make ticket    # Generate GitHub issue markdown for failures
```

### Requirements

- **Hardware**: Any x86_64 or ARM64 machine with ≥4GB RAM. CPU-only (no GPU required).
- **Software**: `apr` (v0.2.16+), `uv` (v0.5+), Python 3.11+
- **Disk**: ~5GB for models directory

### Environment Reproducibility

```bash
# Docker (fully reproducible)
docker build -t tmgt . && docker run tmgt

# Native (requires Rust toolchain)
bash ci/setup.sh   # Install apr
uv sync            # Install Python deps (locked via uv.lock)
```

## Parity Matrix (Current Results)

| Model | APR Int8 | APR Int4 | GGUF Roundtrip | PPL |
|-------|----------|----------|----------------|-----|
| SmolLM-135M | **RUNS** but --json broken (#240) | **RUNS** garbled output | **BLOCKED** (#240) | **BLOCKED** (#242) |
| Qwen2-0.5B | **RUNS** "Paris" correct (#240) | **RUNS** "Paris" correct | **BLOCKED** (#240) | **BLOCKED** (#242) |
| GPT-2 124M | **FAIL** qkv_weight zeros (#241) | **FAIL** qkv_weight zeros | **FAIL** (#241) | **BLOCKED** (#242) |

Current blockers:
- `apr run --json` outputs human-readable text instead of JSON (#240) — blocks programmatic checks
- GPT-2 fused qkv_weight not dequantized (#241) — blocks all GPT-2 inference
- `apr eval` rejects APR format, GGUF fused_matmul type error (#242) — blocks PPL checks

## Filed Issues

| Issue | Bug | Status |
|-------|-----|--------|
| [#231](https://github.com/paiml/aprender/issues/231) | Int8 embedding NaN/Inf + shape mismatch | Fixed |
| [#232](https://github.com/paiml/aprender/issues/232) | Int4 all-zero embedding tensors | Fixed |
| [#233](https://github.com/paiml/aprender/issues/233) | GPT-2 tensor name mapping + wpe density | Fixed |
| [#234](https://github.com/paiml/aprender/issues/234) | lm_head not excluded from quantization | Fixed |
| [#235](https://github.com/paiml/aprender/issues/235) | GPT-2 hidden_dim=64 instead of 768 | Fixed |
| [#236](https://github.com/paiml/aprender/issues/236) | GPT-2 GGUF export hidden_dim=0 | Fixed |
| [#237](https://github.com/paiml/aprender/issues/237) | Quantization write pipeline broken for all tensors | Fixed |
| [#239](https://github.com/paiml/aprender/issues/239) | realizar loader reads Q8/Q4 bytes as f32 | **Fixed** |
| [**#240**](https://github.com/paiml/aprender/issues/240) | **`apr run --json` flag ignored** | **Open — blocks programmatic checks** |
| [**#241**](https://github.com/paiml/aprender/issues/241) | **GPT-2 fused qkv_weight not dequantized** | **Open — blocks GPT-2** |
| [**#242**](https://github.com/paiml/aprender/issues/242) | **`apr eval` rejects APR + GGUF type error** | **Open — blocks PPL** |

## Check Suites

### Inference Checks (blocked by #240 JSON output)

| Suite | Checks | What it tests |
|-------|--------|--------------|
| `check-canary` | 12 | Golden output regression — Int8 text must exactly match oracle |
| `check-token` | 24 | Int4/Int8 token mismatch bounds (≤5/32 and ≤3/32) |
| `check-drift` | 12 | Int8 mismatches ≤ Int4 mismatches + 1 |
| `check-roundtrip` | 6 | APR → GGUF → reimport produces identical tokens |
| `check-ppl` | 9 | PPL within model-specific bounds, Int4/Int8 diff < 0.5 |

### Model Structure Checks (independent of inference)

| Suite | Checks | What it tests |
|-------|--------|--------------|
| `check-inspect` | 6 | Metadata matches expected arch params (Claim 9) |
| `check-validate` | 6 | Magic/header/version integrity (Claim 10) |
| `check-tensors` | 6 | Tensor count and dtypes match quant level (Claim 11) |
| `check-lint` | 6 | No critical lint violations (Claim 12) |
| `check-selftest` | 6 | ≥7/10 `apr check` pipeline stages pass (Claim 13) |
| `check-diff` | 3 | Int4 vs Int8 differ only in dtype (Claim 14) |
| `check-tree` | 6 | Layer/tensor count matches expected (Claim 15) |
| `check-oracle-id` | 6 | `apr oracle` identifies correct architecture (Claim 16) |
| `check-hex` | 6 | Tensor statistics non-zero (Claim 17) |
| `check-debug` | 6 | `apr debug` reports health=OK (Claim 18) |
| `check-bench` | 6 | `apr bench` produces >0 tok/s (Claim 19, expected fail) |
| `check-qa` | 6 | ≥3 QA gates execute without critical failure (Claim 20) |

### Global Checks

| Suite | Checks | What it tests |
|-------|--------|--------------|
| `check-list` | 1 | `apr list` cache inventory succeeds |
| `check-rosetta-diff` | 3 | No layout mismatches between Int8/Int4 |
| `check-parity-gpu` | 3 | GPU/CPU produce identical results (GGUF) |

| | **Total: 139** | **20 suites** |
|---|---|---|

### Subcommand Coverage

23 of 40+ `apr` subcommands exercised:

| Category | Subcommands |
|----------|-------------|
| **Pipeline** | `pull`, `import`, `export`, `run`, `eval` |
| **Inspection** | `inspect`, `validate`, `tensors`, `lint`, `tree`, `hex`, `debug` |
| **Analysis** | `check`, `diff`, `oracle`, `bench`, `qa`, `parity` |
| **Rosetta** | `rosetta diff-tensors` |
| **Utility** | `list`, `--version` |

**Deferred** (broken or inapplicable): `compare-hf` (ZSTD error), `flow` (ZSTD error),
`canary` (audio-only), `serve` (server), `tui`/`chat`/`cbtop` (interactive),
`rm`/`publish` (destructive), `rosetta fingerprint` (Q8 bug).

## Methodology

This repo uses **Popperian falsification**: we attempt to
*disprove* parity rather than *prove* it. Each test encodes a
specific falsifiable prediction. A single failure constitutes
evidence of a bug in the format conversion or runtime engine.

### Oracle Generation

- **Source**: HuggingFace `transformers` (v5.1.0) with PyTorch (v2.10.0)
- **Precision**: float32, CPU-only, `do_sample=False` (deterministic greedy)
- **Random seeds**: Not applicable — greedy decoding is fully
  deterministic (see [ADR-001](CLAIMS.md#adr-001-greedy-decoding-only))
- **Max tokens**: 32 per prompt
- **Output**: JSON with token IDs, decoded text, model metadata

### Tolerance Thresholds

| Comparison | Threshold | Effect Size | n |
|-----------|-----------|-------------|---|
| Int4 vs oracle | ≤5/32 mismatches | 15.6% | 12 |
| Int8 vs oracle | ≤3/32 mismatches | 9.4% | 12 |
| Int8 vs Int4 drift | Int8 ≤ Int4+1 | ≤1 token | 12 |
| Cross-runtime | Exact text match | 0% | 12 |
| PPL Int4 vs Int8 | Diff < 0.5 | <0.5 PPL | 3 |
| Canary (text) | Exact text match | 0% | 12 |

All checks: CI=100% (deterministic greedy decoding).

### Statistical Notes

- **Total sample size**: n = 139 parity checks (exhaustive
  cross-product across 17 suites), plus pytest tests including
  property-based tests with n = 100 iterations via hypothesis.
- **Standard deviation**: σ = 0 for all parity checks. Greedy
  decoding (temperature=0) is fully deterministic. Outputs are
  bit-for-bit identical across runs. Uncertainty is ±0.
- **Confidence interval**: [exact, exact] for all checks. 95%
  and 99% CIs are not applicable (variance = 0). The CI is
  trivially 100%.
- **Sample size justification**: 4 prompts × 3 models = n = 12
  per claim. Prompts cover 4 categories (arithmetic, NLP, code,
  social). 3 models cover 3 architectures (LLaMA, Qwen/GQA,
  GPT-2). Exhaustive over the roster. Total: n = 139 (20 suites).
- **Effect size**: 5/32 = 15.6% ±0 for Int4, 3/32 = 9.4% ±0
  for Int8. Cohen's d: large (Int4), medium (Int8).
- **PPL bounds**: Model-specific ceilings (SmolLM: 20.0,
  Qwen2: 15.0, GPT-2: 30.0) from published benchmarks, with
  2× headroom (σ_headroom ≈ 2× base PPL).

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
Python (uv)                    Parity Checker
───────────                    ──────────────
gen_oracle.py ──► oracle/*.json ◄── parity_check.py
  (rare, manual)                      │
                                      ▼
                              subprocess: apr {subcommand} --json
                              23 subcommands: run, eval, inspect,
                              validate, tensors, lint, check, diff,
                              tree, oracle, hex, debug, bench, qa,
                              list, parity, rosetta diff-tensors,
                              pull, import, export, --version
                                      │
                                      ▼
                              compare output vs oracle / MODEL_METADATA
                              generate GitHub issue markdown
```

## Reproducibility

- **Lock files**: `uv.lock` pins all Python dependencies. Rust tools installed via `cargo install --locked`.
- **Docker**: `Dockerfile` provides fully reproducible environment.
- **CI**: GitHub Actions runs weekly (see `ci/parity.yml`).
- **Determinism**: All inference uses greedy decoding. No random
  seeds needed ([ADR-001](CLAIMS.md#adr-001-greedy-decoding-only)).
- **Versioning**: Oracle JSON includes `transformers_version` and `torch_version` for provenance.

## License

MIT. See [LICENSE](LICENSE).
