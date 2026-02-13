# API Documentation

## Oracle Generator (`scripts/gen_oracle.py`)

### Usage

```bash
uv run python scripts/gen_oracle.py --all          # All 3 models
uv run python scripts/gen_oracle.py --model smollm-135m  # Single model
```

### Models Registry

| Slug | HuggingFace ID |
|------|---------------|
| `smollm-135m` | `HuggingFaceTB/SmolLM-135M` |
| `qwen2-0.5b` | `Qwen/Qwen2-0.5B` |
| `gpt2-124m` | `openai-community/gpt2` |

### Output Format

Each oracle JSON file (`oracle/{slug}/{prompt_name}.json`) contains:

| Field | Type | Description |
|-------|------|-------------|
| `model` | string | HuggingFace model ID |
| `model_slug` | string | Short model identifier |
| `prompt` | string | Input prompt text |
| `prompt_file` | string | Source prompt filename |
| `runtime` | string | Always `"transformers"` |
| `format` | string | Always `"float32"` |
| `transformers_version` | string | HuggingFace transformers version |
| `torch_version` | string | PyTorch version |
| `tokens` | int[] | Generated token IDs (length = max_new_tokens) |
| `text` | string | Decoded text output |
| `token_count` | int | Number of generated tokens |
| `max_new_tokens` | int | Always 32 |
| `do_sample` | bool | Always false (greedy) |

## Parity Checker (`scripts/parity_check.py`)

### Usage

```bash
uv run python scripts/parity_check.py                     # All checks
uv run python scripts/parity_check.py --model smollm-135m  # Single model
uv run python scripts/parity_check.py --check canary       # Single suite
uv run python scripts/parity_check.py --ticket             # GitHub issue markdown
```

### Check Suites

| Suite | Checks | Description |
|-------|--------|-------------|
| `canary` | 12 | Int8 text must exactly match oracle |
| `token` | 24 | Int4 ≤5/32 mismatches, Int8 ≤3/32 mismatches |
| `drift` | 12 | Int8 mismatches ≤ Int4 mismatches + 1 |
| `roundtrip` | 6 | APR → GGUF → reimport produces identical tokens |
| `ppl` | 9 | PPL within ceiling, Int4/Int8 drift < 0.5 |

## Makefile Targets

| Target | Description |
|--------|-------------|
| `oracle` | Generate golden JSON from HuggingFace |
| `pull` | Download 3 models from HuggingFace |
| `convert` | Import to APR (Int4/Int8), export GGUF |
| `check` | Run all 59 parity checks |
| `check-canary` | Golden output regression only |
| `check-token` | Token parity bounds only |
| `check-drift` | Quantization drift ordering only |
| `check-roundtrip` | Format roundtrip only |
| `check-ppl` | Perplexity bounds only |
| `ticket` | Generate GitHub issue markdown for failures |
| `ci` | Full pipeline: pull + convert + check |
| `recheck` | After apr upgrades: clean + convert + check |
| `clean` | Remove generated model files |
