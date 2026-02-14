#!/usr/bin/env python3
"""Oracle: LoRA/QLoRA fine-tuning reference via PEFT.

Produces reference adapter weights and metadata JSON for parity testing
against future `apr finetune` (GH-244).

Usage:
    uv run --extra ops python scripts/oracle_finetune.py --all
    uv run --extra ops python scripts/oracle_finetune.py --model smollm-135m
    uv run --extra ops python scripts/oracle_finetune.py --model smollm-135m --method qlora
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

MODELS = {
    "smollm-135m": "HuggingFaceTB/SmolLM-135M",
    "qwen2-0.5b": "Qwen/Qwen2-0.5B",
    "gpt2-124m": "openai-community/gpt2",
}

ORACLE_DIR = Path("oracle-ops/finetune")
MODELS_DIR = Path("models")

# Tiny training set (enough for oracle generation, not real fine-tuning)
TRAIN_EXAMPLES = [
    {"text": "The Eiffel Tower is located in Paris, France."},
    {"text": "Python is a high-level programming language."},
    {"text": "The speed of light is approximately 3e8 m/s."},
    {"text": "Machine learning models learn patterns from data."},
]


def make_dataset(tokenizer: AutoTokenizer):
    """Create a tiny dataset from TRAIN_EXAMPLES."""
    from datasets import Dataset

    texts = [e["text"] for e in TRAIN_EXAMPLES]
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=64)
    encodings["labels"] = encodings["input_ids"].copy()
    return Dataset.from_dict(encodings)


def finetune_lora(slug: str, hf_id: str) -> dict:
    """LoRA fine-tuning via PEFT."""
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError:
        return {"method": "lora", "error": "peft not available"}

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.float32, device_map="cpu"
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )

    peft_model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())

    dataset = make_dataset(tokenizer)
    out_dir = MODELS_DIR / f"{slug}-lora-r8"

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        no_cuda=True,
    )

    from transformers import Trainer

    start = time.time()
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    elapsed = time.time() - start

    peft_model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # Inference check
    inputs = tokenizer("The capital of France is", return_tensors="pt")
    with torch.no_grad():
        outputs = peft_model.generate(**inputs, max_new_tokens=16, do_sample=False)
    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return {
        "method": "lora",
        "tool": "peft LoraConfig + transformers Trainer",
        "peft_version": _peft_version(),
        "transformers_version": transformers.__version__,
        "model": hf_id,
        "slug": slug,
        "rank": 8,
        "alpha": 16,
        "target_modules": ["q_proj", "v_proj"],
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 2),
        "train_time_s": round(elapsed, 2),
        "output_dir": str(out_dir),
        "adapter_files": _list_adapter_files(out_dir),
        "sanity_text": text,
        "apr_equivalent": f"apr finetune {hf_id} --method lora --rank 8 --data train.jsonl -o {out_dir}",
    }


def finetune_qlora(slug: str, hf_id: str) -> dict:
    """QLoRA fine-tuning (4-bit base + fp16 adapters)."""
    try:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig
    except ImportError:
        return {"method": "qlora", "error": "peft or bitsandbytes not available"}

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, quantization_config=bnb_config, device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)
    except Exception as e:
        return {"method": "qlora", "error": str(e)}

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )

    peft_model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())

    dataset = make_dataset(tokenizer)
    out_dir = MODELS_DIR / f"{slug}-qlora-r16"

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    from transformers import Trainer

    start = time.time()
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    elapsed = time.time() - start

    peft_model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    return {
        "method": "qlora",
        "tool": "peft LoraConfig + bitsandbytes NF4 + transformers Trainer",
        "peft_version": _peft_version(),
        "transformers_version": transformers.__version__,
        "model": hf_id,
        "slug": slug,
        "rank": 16,
        "alpha": 32,
        "base_quant": "nf4",
        "target_modules": ["q_proj", "v_proj"],
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 2),
        "train_time_s": round(elapsed, 2),
        "output_dir": str(out_dir),
        "adapter_files": _list_adapter_files(out_dir),
        "apr_equivalent": f"apr finetune {hf_id} --method qlora --rank 16 --data train.jsonl -o {out_dir}",
    }


def finetune_full(slug: str, hf_id: str) -> dict:
    """Full fine-tuning (all parameters, no adapters)."""
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.float32, device_map="cpu"
    )

    total = sum(p.numel() for p in model.parameters())
    dataset = make_dataset(tokenizer)
    out_dir = MODELS_DIR / f"{slug}-full-ft"

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        no_cuda=True,
    )

    from transformers import Trainer

    start = time.time()
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()
    elapsed = time.time() - start

    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    inputs = tokenizer("The capital of France is", return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return {
        "method": "full",
        "tool": "transformers Trainer (all params unfrozen)",
        "transformers_version": transformers.__version__,
        "model": hf_id,
        "slug": slug,
        "total_params": total,
        "train_time_s": round(elapsed, 2),
        "output_dir": str(out_dir),
        "sanity_text": text,
        "apr_equivalent": f"apr finetune {hf_id} --method full --data train.jsonl -o {out_dir}",
    }


def _peft_version() -> str:
    try:
        import peft
        return peft.__version__
    except (ImportError, AttributeError):
        return "unknown"


def _list_adapter_files(path: Path) -> list[str]:
    if not path.exists():
        return []
    return sorted(str(f.relative_to(path)) for f in path.rglob("*") if f.is_file())


METHODS = {
    "lora": finetune_lora,
    "qlora": finetune_qlora,
    "full": finetune_full,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning oracle generator")
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
