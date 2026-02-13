#!/usr/bin/env python3
"""Generate oracle golden JSON files from HuggingFace transformers.

Usage:
    uv run python scripts/gen_oracle.py --all
    uv run python scripts/gen_oracle.py --model smollm-135m
    uv run python scripts/gen_oracle.py --model qwen2-0.5b
    uv run python scripts/gen_oracle.py --model gpt2-124m
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
MAX_NEW_TOKENS = 32


def load_prompts() -> dict[str, str]:
    """Load all prompt files from prompts/ directory."""
    prompts = {}
    for path in sorted(PROMPTS_DIR.glob("*.txt")):
        prompts[path.stem] = path.read_text().strip()
    return prompts


def generate_oracle(slug: str, hf_id: str, prompts: dict[str, str]) -> None:
    """Generate oracle JSON files for a single model."""
    print(f"Loading model: {hf_id}")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    output_dir = ORACLE_DIR / slug
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for prompt_name, prompt_text in prompts.items():
            print(f"  Generating: {slug}/{prompt_name}")
            inputs = tokenizer(prompt_text, return_tensors="pt")
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
                "format": "float32",
                "transformers_version": transformers.__version__,
                "torch_version": torch.__version__,
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
    args = parser.parse_args()

    if not args.all and not args.model:
        parser.print_help()
        sys.exit(1)

    prompts = load_prompts()
    if not prompts:
        print("Error: no prompt files found in prompts/", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(prompts)} prompts: {', '.join(prompts.keys())}")

    if args.all:
        for slug, hf_id in MODELS.items():
            generate_oracle(slug, hf_id, prompts)
    else:
        slug = args.model
        generate_oracle(slug, MODELS[slug], prompts)

    print("Done.")


if __name__ == "__main__":
    main()
