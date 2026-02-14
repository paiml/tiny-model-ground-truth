#!/usr/bin/env python3
"""Generate oracle golden JSON files from HuggingFace transformers.

Usage:
    uv run python scripts/gen_oracle.py --all
    uv run python scripts/gen_oracle.py --model smollm-135m
    uv run python scripts/gen_oracle.py --model qwen2-0.5b
    uv run python scripts/gen_oracle.py --model gpt2-124m

GPU / precision variants:
    uv run python scripts/gen_oracle.py --all --device cuda --precision bfloat16
    uv run python scripts/gen_oracle.py --all --device cuda --precision float16
    uv run --extra gpu python scripts/gen_oracle.py --all --device cuda --precision bfloat16 --use-kernels
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "smollm-135m": "HuggingFaceTB/SmolLM-135M",
    "qwen2-0.5b": "Qwen/Qwen2-0.5B",
    "gpt2-124m": "openai-community/gpt2",
}

PROMPTS_DIR = Path("prompts")
ORACLE_DIR = Path("oracle")
ORACLE_GPU_DIR = Path("oracle-gpu")
MAX_NEW_TOKENS = 32

DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

# Seed for reproducibility. Greedy decoding (do_sample=False) is fully
# deterministic regardless of seed, but we set it explicitly for provenance.
SEED = 42


def load_prompts() -> dict[str, str]:
    """Load all prompt files from prompts/ directory."""
    prompts = {}
    for path in sorted(PROMPTS_DIR.glob("*.txt")):
        prompts[path.stem] = path.read_text().strip()
    return prompts


def _gpu_metadata(device: str, precision: str, use_kernels: bool) -> dict:
    """Build GPU-specific provenance metadata."""
    meta: dict = {
        "device": device,
        "precision": precision,
    }
    if device == "cuda":
        meta["cuda_version"] = torch.version.cuda or "N/A"
        if torch.cuda.is_available():
            meta["gpu_name"] = torch.cuda.get_device_name(0)
        else:
            meta["gpu_name"] = "N/A"
    if use_kernels:
        try:
            import kernels  # type: ignore[import-untyped]

            meta["kernels_version"] = kernels.__version__
            meta["kernels_enabled"] = True
        except (ImportError, AttributeError):
            meta["kernels_version"] = "N/A"
            meta["kernels_enabled"] = False
    return meta


def generate_oracle(
    slug: str,
    hf_id: str,
    prompts: dict[str, str],
    *,
    device: str = "cpu",
    precision: str = "float32",
    use_kernels: bool = False,
) -> None:
    """Generate oracle JSON files for a single model."""
    dtype = DTYPE_MAP[precision]
    print(f"Loading model: {hf_id} (device={device}, precision={precision})")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=dtype,
        device_map=device,
    )
    model.eval()
    torch.manual_seed(SEED)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # CPU float32 → oracle/; GPU/precision variants → oracle-gpu/{slug}/{precision}/
    if device == "cpu" and precision == "float32":
        output_dir = ORACLE_DIR / slug
    else:
        output_dir = ORACLE_GPU_DIR / slug / precision
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_meta = _gpu_metadata(device, precision, use_kernels)

    with torch.no_grad():
        for prompt_name, prompt_text in prompts.items():
            print(f"  Generating: {slug}/{prompt_name}")
            inputs = tokenizer(prompt_text, return_tensors="pt")
            if device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            prompt_len = inputs["input_ids"].shape[1]

            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

            generated_ids = outputs[0][prompt_len:].tolist()
            generated_text = tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

            oracle = {
                "model": hf_id,
                "model_slug": slug,
                "prompt": prompt_text,
                "prompt_file": f"{prompt_name}.txt",
                "runtime": "transformers",
                "format": precision,
                "transformers_version": transformers.__version__,
                "torch_version": torch.__version__,
                **gpu_meta,
                "tokens": generated_ids,
                "text": generated_text,
                "token_count": len(generated_ids),
                "max_new_tokens": MAX_NEW_TOKENS,
                "do_sample": False,
            }

            output_path = output_dir / f"{prompt_name}.json"
            output_path.write_text(
                json.dumps(oracle, indent=2, ensure_ascii=False) + "\n"
            )
            print(f"    Wrote: {output_path} ({len(generated_ids)} tokens)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate oracle golden JSON from HuggingFace transformers"
    )
    parser.add_argument(
        "--all", action="store_true", help="Generate oracles for all models"
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        help="Generate oracle for a specific model",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for inference (default: cpu)",
    )
    parser.add_argument(
        "--precision",
        choices=list(DTYPE_MAP.keys()),
        default="float32",
        help="Model precision dtype (default: float32)",
    )
    parser.add_argument(
        "--use-kernels",
        action="store_true",
        help="Enable HF custom CUDA kernels (requires kernels package)",
    )
    args = parser.parse_args()

    if not args.all and not args.model:
        parser.print_help()
        sys.exit(1)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("Error: --device cuda requires CUDA-capable GPU", file=sys.stderr)
        sys.exit(1)

    if args.precision != "float32" and args.device == "cpu":
        print(
            "Warning: reduced precision on CPU may not match GPU results",
            file=sys.stderr,
        )

    prompts = load_prompts()
    if not prompts:
        print("Error: no prompt files found in prompts/", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(prompts)} prompts: {', '.join(prompts.keys())}")

    kwargs = {
        "device": args.device,
        "precision": args.precision,
        "use_kernels": args.use_kernels,
    }

    if args.all:
        for slug, hf_id in MODELS.items():
            generate_oracle(slug, hf_id, prompts, **kwargs)
    else:
        slug = args.model
        generate_oracle(slug, MODELS[slug], prompts, **kwargs)

    print("Done.")


if __name__ == "__main__":
    main()
