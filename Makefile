.PHONY: oracle oracle-gpu oracle-gpu-kernels pull convert check check-canary check-token check-drift check-roundtrip check-ppl check-inspect check-validate check-tensors check-lint check-selftest check-diff check-tree check-oracle-id check-hex check-debug check-bench check-qa check-list check-rosetta-diff check-parity-gpu test test-parity test-safetensors test-operations test-canary test-token test-drift test-roundtrip test-ppl test-inspect test-validate test-tensors test-selftest coverage ticket ci recheck clean oracle-quantize oracle-finetune oracle-merge oracle-convert oracle-prune oracle-ops lint typecheck

oracle:
	uv run python scripts/gen_oracle.py --all

oracle-gpu:
	uv run python scripts/gen_oracle.py --all --device cuda --precision bfloat16
	uv run python scripts/gen_oracle.py --all --device cuda --precision float16

oracle-gpu-kernels:
	uv run --extra gpu python scripts/gen_oracle.py --all --device cuda --precision bfloat16 --use-kernels
	uv run --extra gpu python scripts/gen_oracle.py --all --device cuda --precision float16 --use-kernels

pull:
	apr pull hf://HuggingFaceTB/SmolLM-135M
	apr pull hf://Qwen/Qwen2-0.5B
	apr pull hf://openai-community/gpt2

convert:
	@mkdir -p models
	apr import hf://HuggingFaceTB/SmolLM-135M --quantize int4 -o models/smollm-135m-int4.apr
	apr import hf://HuggingFaceTB/SmolLM-135M --quantize int8 -o models/smollm-135m-int8.apr
	apr export models/smollm-135m-int4.apr --format gguf -o models/smollm-135m-int4.gguf
	apr import hf://Qwen/Qwen2-0.5B --quantize int4 -o models/qwen2-0.5b-int4.apr
	apr import hf://Qwen/Qwen2-0.5B --quantize int8 -o models/qwen2-0.5b-int8.apr
	apr export models/qwen2-0.5b-int4.apr --format gguf -o models/qwen2-0.5b-int4.gguf
	apr import hf://openai-community/gpt2 --quantize int4 -o models/gpt2-124m-int4.apr
	apr import hf://openai-community/gpt2 --quantize int8 -o models/gpt2-124m-int8.apr
	apr export models/gpt2-124m-int4.apr --format gguf --skip-contract -o models/gpt2-124m-int4.gguf

# Run all parity checks (shells out to apr run/eval)
check:
	uv run python scripts/parity_check.py

check-canary:
	uv run python scripts/parity_check.py --check canary

check-token:
	uv run python scripts/parity_check.py --check token

check-drift:
	uv run python scripts/parity_check.py --check drift

check-roundtrip:
	uv run python scripts/parity_check.py --check roundtrip

check-ppl:
	uv run python scripts/parity_check.py --check ppl

check-inspect:
	uv run python scripts/parity_check.py --check inspect

check-validate:
	uv run python scripts/parity_check.py --check validate

check-tensors:
	uv run python scripts/parity_check.py --check tensors

check-lint:
	uv run python scripts/parity_check.py --check lint

check-selftest:
	uv run python scripts/parity_check.py --check selftest

check-diff:
	uv run python scripts/parity_check.py --check diff

check-tree:
	uv run python scripts/parity_check.py --check tree

check-oracle-id:
	uv run python scripts/parity_check.py --check oracle-id

check-hex:
	uv run python scripts/parity_check.py --check hex

check-debug:
	uv run python scripts/parity_check.py --check debug

check-bench:
	uv run python scripts/parity_check.py --check bench

check-qa:
	uv run python scripts/parity_check.py --check qa

check-list:
	uv run python scripts/parity_check.py --check list

check-rosetta-diff:
	uv run python scripts/parity_check.py --check rosetta-diff

check-parity-gpu:
	uv run python scripts/parity_check.py --check parity-gpu

# pytest unit + property tests (no apr/model deps)
test:
	uv run --extra test pytest tests/ -v -m "not requires_apr"

# pytest parity tests (requires apr CLI + converted models)
test-parity:
	uv run --extra test pytest tests/ -v -m "requires_apr"

# coverage report (unit + property tests)
coverage:
	uv run --extra dev pytest tests/ -m "not requires_apr" --cov --cov-report=term-missing --cov-report=html -v

test-canary:
	uv run --extra test pytest tests/test_canary.py -v

test-token:
	uv run --extra test pytest tests/test_token_parity.py -v

test-drift:
	uv run --extra test pytest tests/test_quant_drift.py -v

test-roundtrip:
	uv run --extra test pytest tests/test_roundtrip.py -v

test-ppl:
	uv run --extra test pytest tests/test_perplexity.py -v

test-inspect:
	uv run --extra test pytest tests/test_inspect.py -v

test-validate:
	uv run --extra test pytest tests/test_validate.py -v

test-tensors:
	uv run --extra test pytest tests/test_tensors.py -v

test-selftest:
	uv run --extra test pytest tests/test_selftest.py -v

# safetensors parity: Python safetensors vs apr CLI (fast, metadata-only)
test-safetensors:
	uv run --extra test pytest tests/test_safetensors_parity.py -v

# operations parity: apr quantize/finetune/merge/prune/distill/convert vs oracles
test-operations:
	uv run --extra test pytest tests/test_operations_parity.py -v

# Generate GitHub issue markdown for failures
ticket:
	uv run python scripts/parity_check.py --ticket

# ── Operations Oracles ─────────────────────────────────────────────
# Python reference implementations for the 5 most common HF model operations.
# Each produces JSON oracle output in oracle-ops/ for future apr parity testing.

oracle-quantize:
	uv run --extra ops python scripts/oracle_quantize.py --all

oracle-finetune:
	uv run --extra ops python scripts/oracle_finetune.py --all

oracle-merge:
	uv run --extra ops python scripts/oracle_merge.py --all

oracle-convert:
	uv run --extra ops python scripts/oracle_convert.py --all

oracle-prune:
	uv run --extra ops python scripts/oracle_prune.py --all

# Run all operation oracles (slow — downloads models, runs training/quantization)
oracle-ops: oracle-quantize oracle-finetune oracle-merge oracle-convert oracle-prune

# ── apr equivalents (run when apr implements GH-243..247) ──────────
# These will fail until the corresponding apr features are implemented.
# Uncomment and adapt as each feature lands.

# apr-quantize:
# 	apr quantize models/smollm-135m-int8.apr --scheme gptq --bits 4 --json
# 	apr quantize models/smollm-135m-int8.apr --scheme int4 --json

# apr-finetune:
# 	apr finetune models/smollm-135m-int8.apr --method lora --rank 8 --data train.jsonl --json
# 	apr finetune models/smollm-135m-int8.apr --method qlora --rank 16 --data train.jsonl --json

# apr-merge:
# 	apr merge models/smollm-135m-int4.apr models/smollm-135m-int8.apr --strategy slerp --json
# 	apr merge models/smollm-135m-int4.apr models/smollm-135m-int8.apr --strategy dare --json

# apr-convert:
# 	apr export models/smollm-135m-int8.apr --format onnx --json
# 	apr export models/smollm-135m-int8.apr --format safetensors --json

# apr-prune:
# 	apr prune models/smollm-135m-int8.apr --method magnitude --target-ratio 0.3 --json
# 	apr distill models/smollm-135m-int8.apr --student pruned.apr --data train.jsonl --json

# ── Quality ────────────────────────────────────────────────────────
lint:
	uv run ruff check scripts/ tests/

typecheck:
	uv run ty check scripts/ tests/

# Full CI pipeline: download, convert, check
ci: pull convert check

# Reconvert and recheck (skip download, useful after apr upgrades)
recheck: clean convert check

clean:
	rm -f models/*.apr models/*.gguf models/*-roundtrip-tmp.apr
