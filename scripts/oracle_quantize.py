#!/usr/bin/env python3
"""Oracle: Quantization reference via HuggingFace ecosystem.

Produces reference quantized models and metadata JSON for parity testing
against future `apr quantize` (GH-243).

Usage:
    uv run --extra ops python scripts/oracle_quantize.py --all
    uv run --extra ops python scripts/oracle_quantize.py --model smollm-135m
    uv run --extra ops python scripts/oracle_quantize.py --model smollm-135m --method gptq
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "smollm-135m": "HuggingFaceTB/SmolLM-135M",
    "qwen2-0.5b": "Qwen/Qwen2-0.5B",
    "gpt2-124m": "openai-community/gpt2",
}

ORACLE_DIR = Path("oracle-ops/quantize")
MODELS_DIR = Path("models")

# Calibration sentences for GPTQ/AWQ
CALIBRATION_TEXTS = [
    "The capital of France is Paris, which is known for",
    "In mathematics, the Fibonacci sequence is defined as",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr",
    "The process of photosynthesis converts sunlight into",
]


def quantize_dynamic_int8(slug: str, hf_id: str) -> dict:
    """PyTorch dynamic quantization to Int8 (CPU, no calibration)."""
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.float32, device_map="cpu"
    )

    start = time.time()
    quantized = torch.ao.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    elapsed = time.time() - start

    # Get model size
    param_bytes = sum(
        p.nelement() * p.element_size() for p in quantized.parameters()
    )

    out_path = MODELS_DIR / f"{slug}-dynamic-int8.pt"
    torch.save(quantized.state_dict(), out_path)

    # Inference sanity check
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer("The capital of France is", return_tensors="pt")
    with torch.no_grad():
        outputs = quantized.generate(**inputs, max_new_tokens=16, do_sample=False)
    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return {
        "method": "dynamic_int8",
        "tool": "torch.ao.quantization.quantize_dynamic",
        "torch_version": torch.__version__,
        "model": hf_id,
        "slug": slug,
        "quantize_time_s": round(elapsed, 2),
        "param_bytes": param_bytes,
        "output_file": str(out_path),
        "sanity_text": text,
    }


def quantize_bnb_int8(slug: str, hf_id: str) -> dict:
    """bitsandbytes LLM.int8() quantization."""
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        return {"method": "bnb_int8", "error": "bitsandbytes not available"}

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    start = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, quantization_config=bnb_config, device_map="auto"
        )
    except Exception as e:
        return {"method": "bnb_int8", "error": str(e)}
    elapsed = time.time() - start

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return {
        "method": "bnb_int8",
        "tool": "bitsandbytes LLM.int8()",
        "torch_version": torch.__version__,
        "model": hf_id,
        "slug": slug,
        "quantize_time_s": round(elapsed, 2),
        "sanity_text": text,
    }


def quantize_bnb_int4(slug: str, hf_id: str) -> dict:
    """bitsandbytes NF4 quantization (QLoRA-style base)."""
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        return {"method": "bnb_nf4", "error": "bitsandbytes not available"}

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    start = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, quantization_config=bnb_config, device_map="auto"
        )
    except Exception as e:
        return {"method": "bnb_nf4", "error": str(e)}
    elapsed = time.time() - start

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return {
        "method": "bnb_nf4",
        "tool": "bitsandbytes NF4",
        "torch_version": torch.__version__,
        "model": hf_id,
        "slug": slug,
        "quantize_time_s": round(elapsed, 2),
        "sanity_text": text,
    }


def quantize_gptq(slug: str, hf_id: str) -> dict:
    """GPTQ quantization via auto-gptq."""
    try:
        from transformers import GPTQConfig
    except ImportError:
        return {"method": "gptq_int4", "error": "auto-gptq not available"}

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gptq_config = GPTQConfig(
        bits=4,
        dataset=CALIBRATION_TEXTS,
        tokenizer=tokenizer,
    )

    start = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, quantization_config=gptq_config, device_map="auto"
        )
    except Exception as e:
        return {"method": "gptq_int4", "error": str(e)}
    elapsed = time.time() - start

    out_dir = MODELS_DIR / f"{slug}-gptq-int4"
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return {
        "method": "gptq_int4",
        "tool": "auto-gptq via transformers GPTQConfig",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "model": hf_id,
        "slug": slug,
        "bits": 4,
        "quantize_time_s": round(elapsed, 2),
        "output_dir": str(out_dir),
        "sanity_text": text,
    }


def quantize_gguf(slug: str, hf_id: str) -> dict:
    """GGUF quantization via llama-cpp-python (if convert script available)."""
    # GGUF conversion typically uses llama.cpp's convert_hf_to_gguf.py
    # We document the command and record metadata
    return {
        "method": "gguf_q4_k_m",
        "tool": "llama.cpp convert_hf_to_gguf.py + quantize",
        "model": hf_id,
        "slug": slug,
        "commands": [
            f"python convert_hf_to_gguf.py {hf_id} --outfile models/{slug}-f16.gguf --outtype f16",
            f"./llama-quantize models/{slug}-f16.gguf models/{slug}-Q4_K_M.gguf Q4_K_M",
        ],
        "apr_equivalent": f"apr quantize {hf_id} --scheme q4_k_m --format gguf -o models/{slug}-Q4_K_M.gguf",
        "note": "Requires llama.cpp build. Use `apr import --quantize` for APR-native path.",
    }


METHODS = {
    "dynamic-int8": quantize_dynamic_int8,
    "bnb-int8": quantize_bnb_int8,
    "bnb-nf4": quantize_bnb_int4,
    "gptq": quantize_gptq,
    "gguf": quantize_gguf,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantization oracle generator")
    parser.add_argument("--all", action="store_true", help="All models")
    parser.add_argument("--model", choices=list(MODELS.keys()))
    parser.add_argument(
        "--method",
        choices=[*list(METHODS.keys()), "all"],
        default="all",
        help="Quantization method",
    )
    args = parser.parse_args()

    if not args.all and not args.model:
        parser.print_help()
        sys.exit(1)

    ORACLE_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    slugs = MODELS if args.all else {args.model: MODELS[args.model]}
    methods = METHODS if args.method == "all" else {args.method: METHODS[args.method]}

    for slug, hf_id in slugs.items():
        results = []
        for method_name, method_fn in methods.items():
            print(f"  {slug}/{method_name}...", end=" ", flush=True)
            result = method_fn(slug, hf_id)
            results.append(result)
            status = "OK" if "error" not in result else f"SKIP ({result['error'][:40]})"
            print(status)

        out_path = ORACLE_DIR / f"{slug}.json"
        out_path.write_text(json.dumps(results, indent=2) + "\n")
        print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()
