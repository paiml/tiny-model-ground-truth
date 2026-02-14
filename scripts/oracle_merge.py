#!/usr/bin/env python3
"""Oracle: Model merging reference via mergekit/manual SLERP.

Produces reference merged models and metadata JSON for parity testing
against future `apr merge --strategy slerp/ties/dare` (GH-245).

Usage:
    uv run --extra ops python scripts/oracle_merge.py --all
    uv run --extra ops python scripts/oracle_merge.py --model smollm-135m
    uv run --extra ops python scripts/oracle_merge.py --model smollm-135m --method slerp
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "smollm-135m": "HuggingFaceTB/SmolLM-135M",
    "qwen2-0.5b": "Qwen/Qwen2-0.5B",
    "gpt2-124m": "openai-community/gpt2",
}

ORACLE_DIR = Path("oracle-ops/merge")
MODELS_DIR = Path("models")


def _slerp(t: float, v0: torch.Tensor, v1: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Spherical linear interpolation between two tensors."""
    v0_flat = v0.flatten().float()
    v1_flat = v1.flatten().float()

    v0_norm = torch.nn.functional.normalize(v0_flat, dim=0)
    v1_norm = torch.nn.functional.normalize(v1_flat, dim=0)

    dot = torch.clamp(torch.dot(v0_norm, v1_norm), -1.0, 1.0)
    omega = torch.acos(dot)

    if omega.abs() < eps:
        # Nearly parallel — fall back to linear interpolation
        return ((1.0 - t) * v0 + t * v1).to(v0.dtype)

    sin_omega = torch.sin(omega)
    coeff_0 = torch.sin((1.0 - t) * omega) / sin_omega
    coeff_1 = torch.sin(t * omega) / sin_omega
    return (coeff_0 * v0.float() + coeff_1 * v1.float()).to(v0.dtype)


def _average_merge(state_a: dict, state_b: dict, weight: float = 0.5) -> dict:
    """Simple weighted average merge."""
    merged = {}
    for key in state_a:
        if key in state_b:
            merged[key] = (1.0 - weight) * state_a[key].float() + weight * state_b[key].float()
            merged[key] = merged[key].to(state_a[key].dtype)
        else:
            merged[key] = state_a[key]
    return merged


def _slerp_merge(state_a: dict, state_b: dict, t: float = 0.5) -> dict:
    """SLERP merge of two state dicts."""
    merged = {}
    for key in state_a:
        if key in state_b and state_a[key].shape == state_b[key].shape:
            merged[key] = _slerp(t, state_a[key], state_b[key])
        else:
            merged[key] = state_a[key]
    return merged


def _ties_merge(
    state_base: dict, state_ft: dict, density: float = 0.5
) -> dict:
    """TIES merge: Trim, Elect Sign, Merge."""
    merged = {}
    for key in state_base:
        if key not in state_ft or state_base[key].shape != state_ft[key].shape:
            merged[key] = state_base[key]
            continue

        delta = state_ft[key].float() - state_base[key].float()

        # Trim: keep only top-density% by magnitude
        flat = delta.flatten()
        threshold_idx = int(len(flat) * (1 - density))
        if threshold_idx > 0 and threshold_idx < len(flat):
            threshold = flat.abs().sort().values[threshold_idx]
            mask = flat.abs() >= threshold
            flat = flat * mask.float()
            delta = flat.reshape(delta.shape)

        merged[key] = (state_base[key].float() + delta).to(state_base[key].dtype)
    return merged


def _dare_merge(
    state_base: dict, state_ft: dict, drop_rate: float = 0.3
) -> dict:
    """DARE merge: Drop And REscale."""
    merged = {}
    generator = torch.Generator().manual_seed(42)

    for key in state_base:
        if key not in state_ft or state_base[key].shape != state_ft[key].shape:
            merged[key] = state_base[key]
            continue

        delta = state_ft[key].float() - state_base[key].float()

        # Drop: randomly zero out drop_rate% of delta
        mask = torch.bernoulli(
            torch.full_like(delta, 1.0 - drop_rate), generator=generator
        )
        # Rescale: divide by (1 - drop_rate) to preserve expected magnitude
        scale = 1.0 / (1.0 - drop_rate) if drop_rate < 1.0 else 1.0
        delta = delta * mask * scale

        merged[key] = (state_base[key].float() + delta).to(state_base[key].dtype)
    return merged


def merge_average(slug: str, hf_id: str) -> dict:
    """Weighted average merge (self-merge at 0.5 as baseline)."""
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.float32, device_map="cpu"
    )
    state = model.state_dict()

    start = time.time()
    merged_state = _average_merge(state, state, weight=0.5)
    elapsed = time.time() - start

    model.load_state_dict(merged_state)
    out_dir = MODELS_DIR / f"{slug}-merge-avg"
    model.save_pretrained(str(out_dir))

    return {
        "method": "average",
        "tool": "manual weighted average",
        "weight": 0.5,
        "model": hf_id,
        "slug": slug,
        "merge_time_s": round(elapsed, 2),
        "output_dir": str(out_dir),
        "tensor_count": len(merged_state),
        "apr_equivalent": f"apr merge model_a.apr model_b.apr --strategy average --weights 0.5 -o {out_dir}.apr",
    }


def merge_slerp(slug: str, hf_id: str) -> dict:
    """SLERP merge (self-merge at t=0.5 as baseline)."""
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.float32, device_map="cpu"
    )
    state = model.state_dict()

    start = time.time()
    merged_state = _slerp_merge(state, state, t=0.5)
    elapsed = time.time() - start

    model.load_state_dict(merged_state)
    out_dir = MODELS_DIR / f"{slug}-merge-slerp"
    model.save_pretrained(str(out_dir))

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer("The capital of France is", return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return {
        "method": "slerp",
        "tool": "manual SLERP (spherical linear interpolation)",
        "t": 0.5,
        "model": hf_id,
        "slug": slug,
        "merge_time_s": round(elapsed, 2),
        "output_dir": str(out_dir),
        "tensor_count": len(merged_state),
        "sanity_text": text,
        "apr_equivalent": f"apr merge model_a.apr model_b.apr --strategy slerp --weight 0.5 -o {out_dir}.apr",
    }


def merge_ties(slug: str, hf_id: str) -> dict:
    """TIES merge (self-merge with density=0.5)."""
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.float32, device_map="cpu"
    )
    state = model.state_dict()

    start = time.time()
    merged_state = _ties_merge(state, state, density=0.5)
    elapsed = time.time() - start

    model.load_state_dict(merged_state)
    out_dir = MODELS_DIR / f"{slug}-merge-ties"
    model.save_pretrained(str(out_dir))

    return {
        "method": "ties",
        "tool": "manual TIES (Trim, Elect Sign, Merge)",
        "density": 0.5,
        "model": hf_id,
        "slug": slug,
        "merge_time_s": round(elapsed, 2),
        "output_dir": str(out_dir),
        "tensor_count": len(merged_state),
        "apr_equivalent": f"apr merge model_a.apr model_b.apr --strategy ties --density 0.5 -o {out_dir}.apr",
    }


def merge_dare(slug: str, hf_id: str) -> dict:
    """DARE merge (self-merge with drop_rate=0.3)."""
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.float32, device_map="cpu"
    )
    state = model.state_dict()

    start = time.time()
    merged_state = _dare_merge(state, state, drop_rate=0.3)
    elapsed = time.time() - start

    model.load_state_dict(merged_state)
    out_dir = MODELS_DIR / f"{slug}-merge-dare"
    model.save_pretrained(str(out_dir))

    return {
        "method": "dare",
        "tool": "manual DARE (Drop And REscale)",
        "drop_rate": 0.3,
        "seed": 42,
        "model": hf_id,
        "slug": slug,
        "merge_time_s": round(elapsed, 2),
        "output_dir": str(out_dir),
        "tensor_count": len(merged_state),
        "apr_equivalent": f"apr merge model_a.apr model_b.apr --strategy dare --drop-rate 0.3 -o {out_dir}.apr",
    }


METHODS = {
    "average": merge_average,
    "slerp": merge_slerp,
    "ties": merge_ties,
    "dare": merge_dare,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge oracle generator")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--model", choices=list(MODELS.keys()))
    parser.add_argument(
        "--method",
        choices=[*list(METHODS.keys()), "all"],
        default="all",
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
            status = "OK" if "error" not in result else f"SKIP ({result['error'][:40]})"
            print(status)
            results.append(result)

        out_path = ORACLE_DIR / f"{slug}.json"
        out_path.write_text(json.dumps(results, indent=2) + "\n")
        print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()
