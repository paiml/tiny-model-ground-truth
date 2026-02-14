#!/usr/bin/env python3
"""Oracle: Pruning + knowledge distillation reference.

Produces reference pruned/distilled models and metadata JSON for parity
testing against future `apr prune` and `apr distill` (GH-247).

Usage:
    uv run --extra ops python scripts/oracle_prune.py --all
    uv run --extra ops python scripts/oracle_prune.py --model smollm-135m
    uv run --extra ops python scripts/oracle_prune.py --model smollm-135m --method magnitude
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

ORACLE_DIR = Path("oracle-ops/prune")
MODELS_DIR = Path("models")

# Calibration for importance scoring
CALIBRATION_TEXTS = [
    "The capital of France is Paris, which is known for",
    "In mathematics, the Fibonacci sequence is defined as",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr",
    "The process of photosynthesis converts sunlight into",
]


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _count_nonzero(model: torch.nn.Module) -> int:
    return sum(p.nonzero().shape[0] for p in model.parameters())


def _sparsity(model: torch.nn.Module) -> float:
    total = _count_params(model)
    nonzero = _count_nonzero(model)
    return round(1.0 - nonzero / total, 4) if total > 0 else 0.0


def prune_magnitude(slug: str, hf_id: str) -> dict:
    """Unstructured magnitude pruning (zero out smallest weights)."""
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.float32, device_map="cpu"
    )

    target_sparsity = 0.3
    start = time.time()

    for name, param in model.named_parameters():
        if "weight" in name and param.dim() >= 2:
            flat = param.data.flatten().abs()
            k = int(len(flat) * target_sparsity)
            if k > 0:
                threshold = flat.sort().values[k]
                mask = param.data.abs() >= threshold
                param.data *= mask.float()

    elapsed = time.time() - start

    actual_sparsity = _sparsity(model)
    out_dir = MODELS_DIR / f"{slug}-pruned-mag30"
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    inputs = tokenizer("The capital of France is", return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return {
        "method": "magnitude",
        "tool": "manual magnitude pruning (threshold sort)",
        "model": hf_id,
        "slug": slug,
        "target_sparsity": target_sparsity,
        "actual_sparsity": actual_sparsity,
        "total_params": _count_params(model),
        "prune_time_s": round(elapsed, 2),
        "output_dir": str(out_dir),
        "sanity_text": text,
        "apr_equivalent": f"apr prune {hf_id} --method magnitude --target-ratio 0.3 -o {out_dir}.apr",
    }


def prune_structured_heads(slug: str, hf_id: str) -> dict:
    """Structured pruning: remove attention heads by L1 norm."""
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.float32, device_map="cpu"
    )

    # Score attention heads by their Q projection weight L1 norm
    head_scores = []
    config = model.config
    num_heads = getattr(config, "num_attention_heads", 12)
    hidden = getattr(config, "hidden_size", 768)
    head_dim = hidden // num_heads

    for name, param in model.named_parameters():
        if (
            ("q_proj.weight" in name or "c_attn.weight" in name)
            and param.shape[0] >= hidden
        ):
            for h in range(num_heads):
                start_idx = h * head_dim
                end_idx = (h + 1) * head_dim
                head_weight = param.data[start_idx:end_idx]
                score = head_weight.abs().mean().item()
                layer = name.split(".")[2]
                head_scores.append({
                    "layer": layer,
                    "head": h,
                    "score": round(score, 6),
                    "param": name,
                })

    # Sort by score (lowest = least important)
    head_scores.sort(key=lambda x: x["score"])
    heads_to_prune = len(head_scores) // 4  # Remove bottom 25%

    return {
        "method": "structured_heads",
        "tool": "manual attention head scoring (L1 norm of Q projection)",
        "model": hf_id,
        "slug": slug,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "total_heads_scored": len(head_scores),
        "heads_to_prune": heads_to_prune,
        "bottom_5_heads": head_scores[:5],
        "top_5_heads": head_scores[-5:],
        "apr_equivalent": f"apr prune {hf_id} --method structured --analyze --json",
        "note": "Analysis only — actual head removal requires architecture surgery",
    }


def distill_logit_matching(slug: str, hf_id: str) -> dict:
    """Knowledge distillation via logit matching (teacher=self, 1 epoch)."""
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.float32, device_map="cpu"
    )
    student = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.float32, device_map="cpu"
    )
    teacher.eval()

    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-5)
    kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    temperature = 2.0

    start = time.time()
    total_loss = 0.0
    steps = 0

    for text in CALIBRATION_TEXTS:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)

        with torch.no_grad():
            teacher_logits = teacher(**inputs).logits

        student_logits = student(**inputs).logits

        # KL divergence on softened logits
        t_soft = torch.nn.functional.log_softmax(teacher_logits / temperature, dim=-1)
        s_soft = torch.nn.functional.log_softmax(student_logits / temperature, dim=-1)
        loss = kl_loss_fn(s_soft, t_soft) * (temperature ** 2)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        steps += 1

    elapsed = time.time() - start

    out_dir = MODELS_DIR / f"{slug}-distilled"
    student.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    inputs = tokenizer("The capital of France is", return_tensors="pt")
    with torch.no_grad():
        outputs = student.generate(**inputs, max_new_tokens=16, do_sample=False)
    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return {
        "method": "distill_logit",
        "tool": "manual KL-divergence logit matching",
        "model": hf_id,
        "slug": slug,
        "temperature": temperature,
        "steps": steps,
        "avg_loss": round(total_loss / steps, 6) if steps > 0 else 0,
        "distill_time_s": round(elapsed, 2),
        "output_dir": str(out_dir),
        "sanity_text": text,
        "apr_equivalent": f"apr distill {hf_id} --student pruned.apr --data train.jsonl -o {out_dir}.apr",
    }


def prune_and_distill(slug: str, hf_id: str) -> dict:
    """Combined pipeline: magnitude prune → distill (Minitron-style)."""
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.float32, device_map="cpu"
    )
    student = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.float32, device_map="cpu"
    )
    teacher.eval()

    # Step 1: Prune student
    target_sparsity = 0.2
    for name, param in student.named_parameters():
        if "weight" in name and param.dim() >= 2:
            flat = param.data.flatten().abs()
            k = int(len(flat) * target_sparsity)
            if k > 0:
                threshold = flat.sort().values[k]
                mask = param.data.abs() >= threshold
                param.data *= mask.float()

    sparsity_after_prune = _sparsity(student)

    # Step 2: Distill
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-5)
    kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    temperature = 2.0

    start = time.time()
    total_loss = 0.0
    steps = 0

    for text in CALIBRATION_TEXTS:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)

        with torch.no_grad():
            teacher_logits = teacher(**inputs).logits
        student_logits = student(**inputs).logits

        t_soft = torch.nn.functional.log_softmax(teacher_logits / temperature, dim=-1)
        s_soft = torch.nn.functional.log_softmax(student_logits / temperature, dim=-1)
        loss = kl_loss_fn(s_soft, t_soft) * (temperature ** 2)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        steps += 1

    elapsed = time.time() - start

    sparsity_after_distill = _sparsity(student)
    out_dir = MODELS_DIR / f"{slug}-prune-distill"
    student.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    inputs = tokenizer("The capital of France is", return_tensors="pt")
    with torch.no_grad():
        outputs = student.generate(**inputs, max_new_tokens=16, do_sample=False)
    text_out = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return {
        "method": "prune_and_distill",
        "tool": "magnitude pruning → KL distillation (Minitron-style)",
        "model": hf_id,
        "slug": slug,
        "target_sparsity": target_sparsity,
        "sparsity_after_prune": sparsity_after_prune,
        "sparsity_after_distill": sparsity_after_distill,
        "temperature": temperature,
        "steps": steps,
        "avg_loss": round(total_loss / steps, 6) if steps > 0 else 0,
        "total_time_s": round(elapsed, 2),
        "output_dir": str(out_dir),
        "sanity_text": text_out,
        "apr_equivalent": (
            f"apr prune {hf_id} --method magnitude --target-ratio 0.2 "
            f"--distill --data train.jsonl -o {out_dir}.apr"
        ),
    }


METHODS = {
    "magnitude": prune_magnitude,
    "structured": prune_structured_heads,
    "distill": distill_logit_matching,
    "prune-distill": prune_and_distill,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pruning/distillation oracle generator")
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
