# CLAUDE.md

## Project Overview

**tiny-model-ground-truth** is a Popperian falsification repo
for model format conversions and runtime parity across the
Sovereign AI Stack.

Central hypothesis: given a tiny model from HuggingFace, every
format conversion and runtime engine must produce
token-identical greedy outputs (or bounded quantization drift).

### Architecture

- **Python** (via `uv`): Oracle generation (golden JSON from
  `transformers`) AND parity checking (shells out to `apr`,
  compares against oracle)
- **`apr` CLI**: All heavy lifting — `pull`, `import`, `export`,
  `run --json`, `eval --json`

## Model Roster

| Model | HF ID | Params | Architecture |
|-------|-------|--------|-------------|
| SmolLM-135M | `HuggingFaceTB/SmolLM-135M` | 135M | LLaMA-style |
| Qwen2-0.5B | `Qwen/Qwen2-0.5B` | 500M | Qwen (GQA) |
| GPT-2 | `openai-community/gpt2` | 124M | GPT-2 |

## Commands

| Command | Description |
|---------|-------------|
| `make oracle` | Generate golden JSON from HuggingFace transformers |
| `make pull` | Download models from HuggingFace |
| `make convert` | Import/export to APR and GGUF formats |
| `make test` | Unit + property tests (no apr/model deps, 60s timeout) |
| `make test-parity` | Integration parity tests (requires apr + models) |
| `make coverage` | Coverage report for unit tests (term + HTML) |
| `make check` | Run all parity checks (shells out to `apr run`) |
| `make check-canary` | Golden output regression (Int8 exact match) |
| `make check-token` | Token mismatch bounds (Int4 ≤5/32, Int8 ≤3/32) |
| `make check-drift` | Quantization drift ordering (Int8 ≤ Int4 + 1) |
| `make check-roundtrip` | APR → GGUF → reimport token identity |
| `make check-ppl` | Perplexity bounds and Int4/Int8 drift |
| `make ticket` | Generate GitHub issue markdown for failures |
| `make ci` | Full pipeline (pull + convert + check) |
| `make recheck` | After apr upgrades (clean + convert + check) |
| `make oracle-gpu` | Generate BF16/FP16 GPU oracles (requires CUDA) |
| `make oracle-gpu-kernels` | GPU oracles with HF custom kernels (requires CUDA + kernels) |
| `make clean` | Remove generated model files |

## Tolerance Standards

| Comparison | Tolerance | Rationale |
|-----------|-----------|-----------|
| Int4 tokens vs oracle | ≤5/32 mismatches | Quantization drift |
| Int8 tokens vs oracle | ≤3/32 mismatches | Higher precision |
| Int8 vs Int4 drift | Int8 ≤ Int4 + 1 | Int8 strictly better |
| PPL Int4 vs Int8 | Diff < 0.5 | Statistical bound |
| BF16 tokens vs float32 | ≤3/32 mismatches | GPU precision drift |
| FP16 tokens vs float32 | ≤5/32 mismatches | GPU precision drift |
| BF16 vs FP16 drift | ≤3/32 mismatches | Inter-precision bound |
| Precision ladder | BF16 ≤ FP16 drift | Monotonic ordering |
| Canary (text regression) | Exact text match | No inference regression |

## Parity Status (2026-02-14, apr 0.2.16)

| Check | Result | Notes |
|-------|--------|-------|
| list | 1/1 PASS | Cache listing works |
| quant-drift | 12/12 PASS | Int8 ≤ Int4+1 holds |
| roundtrip | 4/5 PASS | GPT-2 GGUF roundtrip broken |
| tensors | 6/6 PASS | Correct tensor counts |
| hex | 6/6 PASS | Non-zero tensor data |
| rosetta-diff | 3/3 PASS | No layout mismatches |
| tree | 6/6 PASS | Tensor counts correct |
| debug | 4/4 PASS | Health=OK all models |
| canary | 0/12 FAIL | Inference produces wrong tokens |
| token-parity | 0/24 FAIL | 32/32 mismatches (all wrong) |
| ppl | 0/6 FAIL | --json output includes ANSI escapes |
| inspect | 0/6 FAIL | arch field empty in JSON |
| validate | 0/6 FAIL | 22-23 items not implemented |
| lint | 0/6 FAIL | --json flag ignored |
| selftest | 0/6 FAIL | GH-187 embedding issue |
| bench | 0/6 FAIL | 0.0 tok/s throughput |
| diff | 0/2 FAIL | Structural diffs in qwen2/gpt2 |
| oracle-id | 0/6 FAIL | arch field empty |
| qa | 1/6 FAIL | GPT-2 + others fail |
| parity-gpu | 2 skip, 1 FAIL | GPT-2 tensor not found |
| **Total** | **42/135** | **93 failures** |

## Quality Standards

- All checks must pass before merge
- Commit format: `feat|fix|test|docs: msg (Refs TMGT-XXX)`
- `bashrs comply check` must score A+ (100/100)
- `make test` must pass (330 tests, <25s)
- `make coverage` must meet 95% fail_under per file
  - `tests/helpers.py`: 100%
  - `scripts/parity_check.py`: 98%

## Code Search

**NEVER use grep or rg for code discovery. ALWAYS use pmat query.**

| Task | Command |
|------|---------|
| Find functions | `pmat query "oracle" --limit 10` |
| Find high-quality | `pmat query "check" --min-grade A` |
| Find with faults | `pmat query "mismatch" --faults` |
| Include source | `pmat query "parity" --include-source` |
| Regex search | `pmat query --regex "def\s+check_"` |
| Literal search | `pmat query --literal "count_mismatches"` |
| Exclude tests | `pmat query "oracle" --exclude-tests` |

```bash
pmat query "oracle generation" --include-source
pmat query "parity check" --faults --limit 10
pmat query "count_mismatches" --include-source --limit 5
```
