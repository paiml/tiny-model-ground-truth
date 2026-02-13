.PHONY: oracle pull convert check check-canary check-token check-drift check-roundtrip check-ppl test test-canary test-token test-drift test-roundtrip test-ppl ticket ci recheck clean

oracle:
	uv run python scripts/gen_oracle.py --all

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

# pytest-based parity tests (same checks, pytest runner)
test:
	uv run --extra test pytest tests/ -v

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

# Generate GitHub issue markdown for failures
ticket:
	uv run python scripts/parity_check.py --ticket

# Full CI pipeline: download, convert, check
ci: pull convert check

# Reconvert and recheck (skip download, useful after apr upgrades)
recheck: clean convert check

clean:
	rm -f models/*.apr models/*.gguf models/*-roundtrip-tmp.apr
