# CLAUDE.md

## Project Overview

**tiny-model-ground-truth** is a Popperian falsification repository for model format conversions and runtime parity across the Sovereign AI Stack. Central hypothesis: *"Given a tiny model from HuggingFace, every format conversion and runtime engine must produce token-identical greedy outputs (or bounded quantization drift)."*

### Architecture

- **Python** (via `uv`): ONLY for generating oracle golden JSON files from HuggingFace `transformers`. Rare, manual step.
- **Ruchy**: ALL parity tests. Shells out to `apr`, `llama-cli`, etc. via `process::execute()`, parses JSON with `parse_json()`, asserts with `@test` + `assert()`/`assert_eq()`.
- **`apr` CLI**: All heavy lifting — `pull`, `import`, `export`, `run --json`, `eval --json`, `diff --json`, `validate`.

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
| `make test` | Run all ruchy parity tests |
| `make test-canary` | Golden output regression tests |
| `make test-token` | Token match bound tests |
| `make test-quant` | Quantization drift ordering tests |
| `make test-roundtrip` | APR/GGUF format roundtrip tests |
| `make test-runtime` | Cross-runtime parity tests |
| `make test-ppl` | Perplexity bound tests |
| `make ci` | Full CI pipeline (pull + convert + test) |
| `make clean` | Remove generated model files |

## Tolerance Standards

| Comparison | Tolerance | Rationale |
|-----------|-----------|-----------|
| Q4K tokens vs oracle | ≤5/32 mismatches | Quantization drift |
| Q6K tokens vs oracle | ≤3/32 mismatches | Higher precision |
| Q6K vs Q4K drift | Q6K ≤ Q4K + 1 | Q6K strictly better |
| Cross-runtime (same GGUF) | Exact text match | Deterministic greedy |
| PPL Q4K vs Q6K | Diff < 0.5 | Statistical bound |
| Canary (text regression) | Exact text match | No inference regression |

## Quality Standards

- All tests must pass before merge
- Commit format: `feat|fix|test: msg (Refs TMGT-XXX)`
- Python only for oracle generation, ruchy for all tests
- Each ruchy test file is self-contained (no file imports)

## Code Search

Use `pmat query` for code search (per org convention):

```bash
pmat query "oracle generation" --include-source
pmat query "parity test" --limit 10
```
