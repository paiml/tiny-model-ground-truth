# Model Card — tiny-model-ground-truth

## Dataset Purpose

This repository uses 3 tiny language models as test fixtures for Popperian falsification of model format conversions. The models are NOT trained or fine-tuned here — they are downloaded from HuggingFace and used read-only.

## Models Used

### SmolLM-135M
- **Source**: [HuggingFaceTB/SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M)
- **Parameters**: 135M (30 layers, 576 hidden, 9 attention heads)
- **Architecture**: LLaMA-style
- **License**: Apache 2.0
- **SHA256**: Verified via `apr pull` content hash
- **Random seed**: Not applicable (greedy decode, no training)

### Qwen2-0.5B
- **Source**: [Qwen/Qwen2-0.5B](https://huggingface.co/Qwen/Qwen2-0.5B)
- **Parameters**: 500M (24 layers, 896 hidden, 14 attention heads, 2 KV heads)
- **Architecture**: Qwen (Grouped Query Attention)
- **License**: Apache 2.0
- **SHA256**: Verified via `apr pull` content hash
- **Random seed**: Not applicable (greedy decode, no training)

### GPT-2 124M
- **Source**: [openai-community/gpt2](https://huggingface.co/openai-community/gpt2)
- **Parameters**: 124M (12 layers, 768 hidden, 12 attention heads)
- **Architecture**: GPT-2
- **License**: MIT
- **SHA256**: Verified via `apr pull` content hash
- **Random seed**: Not applicable (greedy decode, no training)

## Prompt Dataset

4 deterministic prompts, committed to version control in `prompts/`:

| ID | Category | Text | Tokens (approx) |
|----|----------|------|-----------------|
| arithmetic | Math | `What is 2+2? Answer:` | 8 |
| completion | NLP | `The capital of France is` | 6 |
| code | Programming | `def fibonacci(n):` | 5 |
| greeting | Social | `Hello, my name is` | 5 |

## Inference Configuration

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `do_sample` | `false` | Deterministic greedy decoding |
| `max_new_tokens` | `32` | Sufficient to detect divergence |
| `temperature` | `0.0` | Greedy argmax |
| `torch_dtype` | `float32` | Maximum precision oracle |
| `device_map` | `cpu` | Avoids GPU non-determinism |
| Random seed | Not set | Greedy decode is fully deterministic; seed has no effect |

## Ethical Considerations

- Models are used for infrastructure testing only, not for generation of content
- All prompts are benign and factual
- No personal data is processed
- No model training or fine-tuning is performed
