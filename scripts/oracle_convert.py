#!/usr/bin/env python3
"""Oracle: Format conversion reference via HuggingFace Optimum + SafeTensors.

Produces reference converted models and metadata JSON for parity testing
against future `apr export --format onnx/mlx/openvino` (GH-246).

Usage:
    uv run --extra ops python scripts/oracle_convert.py --all
    uv run --extra ops python scripts/oracle_convert.py --model smollm-135m
    uv run --extra ops python scripts/oracle_convert.py --model smollm-135m --format onnx
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

ORACLE_DIR = Path("oracle-ops/convert")
MODELS_DIR = Path("models")


def convert_safetensors(slug: str, hf_id: str) -> dict:
    """Export to SafeTensors format (baseline — transformers default)."""
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.float32, device_map="cpu"
    )

    out_dir = MODELS_DIR / f"{slug}-safetensors"
    start = time.time()
    model.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))
    elapsed = time.time() - start

    st_files = list(out_dir.glob("*.safetensors"))
    total_bytes = sum(f.stat().st_size for f in st_files)

    return {
        "format": "safetensors",
        "tool": "transformers save_pretrained(safe_serialization=True)",
        "transformers_version": transformers.__version__,
        "model": hf_id,
        "slug": slug,
        "convert_time_s": round(elapsed, 2),
        "output_dir": str(out_dir),
        "file_count": len(st_files),
        "total_bytes": total_bytes,
        "apr_equivalent": f"apr export {slug}.apr --format safetensors -o {out_dir}",
    }


def convert_onnx(slug: str, hf_id: str) -> dict:
    """Export to ONNX format via Optimum."""
    try:
        from optimum.exporters.onnx import main_export
    except ImportError:
        return {"format": "onnx", "error": "optimum not available"}

    out_dir = MODELS_DIR / f"{slug}-onnx"

    start = time.time()
    try:
        main_export(
            hf_id,
            output=str(out_dir),
            task="text-generation",
            no_post_process=True,
        )
    except Exception as e:
        return {"format": "onnx", "error": str(e), "model": hf_id, "slug": slug}
    elapsed = time.time() - start

    onnx_files = list(out_dir.glob("**/*.onnx"))
    total_bytes = sum(f.stat().st_size for f in onnx_files)

    # Validate with onnxruntime
    sanity = None
    try:
        import onnxruntime as ort

        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        session = ort.InferenceSession(str(onnx_files[0]))
        input_names = [inp.name for inp in session.get_inputs()]
        sanity = f"ONNX session loaded, inputs: {input_names}"
    except Exception as e:
        sanity = f"validation failed: {e}"

    return {
        "format": "onnx",
        "tool": "optimum.exporters.onnx.main_export",
        "transformers_version": transformers.__version__,
        "model": hf_id,
        "slug": slug,
        "convert_time_s": round(elapsed, 2),
        "output_dir": str(out_dir),
        "file_count": len(onnx_files),
        "total_bytes": total_bytes,
        "sanity_check": sanity,
        "apr_equivalent": f"apr export {slug}.apr --format onnx -o {out_dir}",
    }


def convert_torch_pt(slug: str, hf_id: str) -> dict:
    """Export to PyTorch .pt format (pickle-based, for comparison)."""
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.float32, device_map="cpu"
    )

    out_path = MODELS_DIR / f"{slug}-pytorch.pt"
    start = time.time()
    torch.save(model.state_dict(), out_path)
    elapsed = time.time() - start

    return {
        "format": "pytorch_pt",
        "tool": "torch.save(state_dict)",
        "torch_version": torch.__version__,
        "model": hf_id,
        "slug": slug,
        "convert_time_s": round(elapsed, 2),
        "output_file": str(out_path),
        "total_bytes": out_path.stat().st_size,
        "note": "pickle-based — SafeTensors preferred for safety",
    }


def convert_gguf_reference(slug: str, hf_id: str) -> dict:
    """Document GGUF conversion commands (requires llama.cpp binary)."""
    return {
        "format": "gguf",
        "tool": "llama.cpp convert_hf_to_gguf.py",
        "model": hf_id,
        "slug": slug,
        "commands": [
            f"python convert_hf_to_gguf.py {hf_id} --outfile models/{slug}-f16.gguf --outtype f16",
            f"./llama-quantize models/{slug}-f16.gguf models/{slug}-Q4_K_M.gguf Q4_K_M",
            f"./llama-quantize models/{slug}-f16.gguf models/{slug}-Q8_0.gguf Q8_0",
        ],
        "apr_equivalent": f"apr export {slug}.apr --format gguf -o models/{slug}.gguf",
        "note": "Already tested via existing convert target. Listed here for completeness.",
    }


FORMATS = {
    "safetensors": convert_safetensors,
    "onnx": convert_onnx,
    "pytorch": convert_torch_pt,
    "gguf": convert_gguf_reference,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Format conversion oracle generator")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--model", choices=list(MODELS.keys()))
    parser.add_argument(
        "--format",
        choices=[*list(FORMATS.keys()), "all"],
        default="all",
    )
    args = parser.parse_args()

    if not args.all and not args.model:
        parser.print_help()
        sys.exit(1)

    ORACLE_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    slugs = MODELS if args.all else {args.model: MODELS[args.model]}
    formats = FORMATS if args.format == "all" else {args.format: FORMATS[args.format]}

    for slug, hf_id in slugs.items():
        results = []
        for fmt_name, fmt_fn in formats.items():
            print(f"  {slug}/{fmt_name}...", end=" ", flush=True)
            result = fmt_fn(slug, hf_id)
            status = "OK" if "error" not in result else f"SKIP ({result['error'][:40]})"
            print(status)
            results.append(result)

        out_path = ORACLE_DIR / f"{slug}.json"
        out_path.write_text(json.dumps(results, indent=2) + "\n")
        print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()
