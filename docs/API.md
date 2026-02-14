# API Documentation

## Oracle Generator (`scripts/gen_oracle.py`)

### Usage

```bash
uv run python scripts/gen_oracle.py --all          # All 3 models
uv run python scripts/gen_oracle.py --model smollm-135m  # Single model

# GPU / precision variants
uv run python scripts/gen_oracle.py --all --device cuda --precision bfloat16
uv run python scripts/gen_oracle.py --all --device cuda --precision float16
uv run --extra gpu python scripts/gen_oracle.py --all --device cuda --precision bfloat16 --use-kernels
```

### CLI Flags

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--all` | — | — | Generate oracles for all models |
| `--model` | `smollm-135m`, `qwen2-0.5b`, `gpt2-124m` | — | Single model |
| `--device` | `cpu`, `cuda` | `cpu` | Inference device |
| `--precision` | `float32`, `bfloat16`, `float16` | `float32` | Model dtype |
| `--use-kernels` | — | off | Enable HF custom CUDA kernels |

### Models Registry

| Slug | HuggingFace ID |
|------|---------------|
| `smollm-135m` | `HuggingFaceTB/SmolLM-135M` |
| `qwen2-0.5b` | `Qwen/Qwen2-0.5B` |
| `gpt2-124m` | `openai-community/gpt2` |

### Output Format

Each CPU oracle JSON file (`oracle/{slug}/{prompt_name}.json`) contains:

| Field | Type | Description |
|-------|------|-------------|
| `model` | string | HuggingFace model ID |
| `model_slug` | string | Short model identifier |
| `prompt` | string | Input prompt text |
| `prompt_file` | string | Source prompt filename |
| `runtime` | string | Always `"transformers"` |
| `format` | string | Precision dtype (`"float32"`, `"bfloat16"`, `"float16"`) |
| `transformers_version` | string | HuggingFace transformers version |
| `torch_version` | string | PyTorch version |
| `tokens` | int[] | Generated token IDs (length = max_new_tokens) |
| `text` | string | Decoded text output |
| `token_count` | int | Number of generated tokens |
| `max_new_tokens` | int | Always 32 |
| `do_sample` | bool | Always false (greedy) |

GPU oracle files (`oracle-gpu/{slug}/{precision}/{prompt_name}.json`) include
all CPU fields plus:

| Field | Type | Description |
|-------|------|-------------|
| `device` | string | `"cuda"` |
| `precision` | string | `"bfloat16"` or `"float16"` |
| `cuda_version` | string | CUDA toolkit version (e.g. `"12.8"`) |
| `gpu_name` | string | GPU device name |
| `kernels_version` | string | HF kernels package version (if `--use-kernels`) |
| `kernels_enabled` | bool | Whether custom kernels were active (if `--use-kernels`) |

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
| `oracle-gpu` | Generate BF16/FP16 GPU oracles (requires CUDA) |
| `oracle-gpu-kernels` | GPU oracles with HF custom kernels |
| `clean` | Remove generated model files |
