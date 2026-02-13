.PHONY: oracle pull convert test test-canary test-token test-quant test-roundtrip test-runtime test-ppl ci clean

oracle:
	uv run python scripts/gen_oracle.py --all

pull:
	apr pull hf://HuggingFaceTB/SmolLM-135M
	apr pull hf://Qwen/Qwen2-0.5B
	apr pull hf://openai-community/gpt2

convert:
	@mkdir -p models
	apr import hf://HuggingFaceTB/SmolLM-135M --quantize q4k -o models/smollm-135m-q4k.apr
	apr import hf://HuggingFaceTB/SmolLM-135M --quantize q6k -o models/smollm-135m-q6k.apr
	apr export models/smollm-135m-q4k.apr --format gguf -o models/smollm-135m-q4k.gguf
	apr import hf://Qwen/Qwen2-0.5B --quantize q4k -o models/qwen2-0.5b-q4k.apr
	apr import hf://Qwen/Qwen2-0.5B --quantize q6k -o models/qwen2-0.5b-q6k.apr
	apr export models/qwen2-0.5b-q4k.apr --format gguf -o models/qwen2-0.5b-q4k.gguf
	apr import hf://openai-community/gpt2 --quantize q4k -o models/gpt2-124m-q4k.apr
	apr import hf://openai-community/gpt2 --quantize q6k -o models/gpt2-124m-q6k.apr
	apr export models/gpt2-124m-q4k.apr --format gguf -o models/gpt2-124m-q4k.gguf

test:
	ruchy test tests/ --parallel

test-canary:
	ruchy test tests/test_canary.ruchy
test-token:
	ruchy test tests/test_token_parity.ruchy
test-quant:
	ruchy test tests/test_quant_drift.ruchy
test-roundtrip:
	ruchy test tests/test_format_roundtrip.ruchy
test-runtime:
	ruchy test tests/test_runtime_parity.ruchy
test-ppl:
	ruchy test tests/test_perplexity.ruchy

ci: pull convert test

clean:
	rm -f models/*.apr models/*.gguf
