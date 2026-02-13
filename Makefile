.PHONY: oracle pull convert check check-canary check-token check-drift check-roundtrip check-ppl ticket ci clean

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

# Real parity checks (actually runs apr inference)
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

# Generate GitHub issue markdown for failures
ticket:
	uv run python scripts/parity_check.py --ticket

ci: pull convert check

clean:
	rm -f models/*.apr models/*.gguf
