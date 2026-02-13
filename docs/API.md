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

## Ruchy Test Helpers

Each `.ruchy` test file includes these inline helpers (ruchy has no file imports):

### `run_apr(args: string[]) -> Result<string>`
Execute `apr` CLI with arguments. Returns stdout on success, error on non-zero exit.

### `apr_run_json(model_path: string, prompt: string) -> Result<object>`
Run inference: `apr run <model> -p <prompt> -n 32 --json`. Returns parsed JSON.

### `load_oracle(slug: string, prompt_name: string) -> Result<object>`
Load oracle JSON from `oracle/{slug}/{prompt_name}.json`. Returns parsed JSON.

### `count_mismatches(a: int[], b: int[]) -> int`
Count token-level mismatches between two token arrays. Includes length difference.

### `is_available(bin: string) -> Result<bool>`
Check if a binary is available in PATH (via `which`).

## Makefile Targets

| Target | Dependencies | Description |
|--------|-------------|-------------|
| `oracle` | Python env | Generate golden JSON from HuggingFace |
| `pull` | Network | Download 3 models from HuggingFace |
| `convert` | `pull` | Import to APR (Int4/Int8), export GGUF |
| `test` | `convert` | Run all 6 ruchy test suites in parallel |
| `test-canary` | `convert` | Run canary regression tests only |
| `test-token` | `convert` | Run token parity tests only |
| `test-quant` | `convert` | Run quant drift tests only |
| `test-roundtrip` | `convert` | Run format roundtrip tests only |
| `test-runtime` | `convert` | Run runtime parity tests only |
| `test-ppl` | `convert` | Run perplexity tests only |
| `ci` | Network | Full pipeline: pull + convert + test |
| `clean` | None | Remove generated model files |
