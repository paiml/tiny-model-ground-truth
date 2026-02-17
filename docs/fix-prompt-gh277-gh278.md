# Fix Prompt: apr GGUF Compatibility (GH-277, GH-278)

> Give this prompt to Claude Code in the `aprender` repo to fix both issues.

---

## Context

Two issues were filed from Popperian falsification testing in `tiny-model-ground-truth`
(Layer 4: llama.cpp cross-runtime parity). The test harness feeds the same GGUF files to
both `apr` and `llama-completion` (llama.cpp) with greedy decoding and compares outputs.

**Falsification test repo**: `../tiny-model-ground-truth`
**Test file**: `tests/test_llamacpp_parity.py`
**Test runner**: `make test-llamacpp` (requires llama-completion in PATH + native GGUFs)

## Issue 1: GH-277 — apr-exported GGUFs rejected by llama.cpp

### Problem

`apr export --format gguf` produces GGUF files that llama.cpp cannot load. Three models,
three distinct failure modes:

| Model | Error | Root Cause |
|-------|-------|------------|
| SmolLM-135M | `unknown pre-tokenizer type: 'llama'` | `tokenizer.ggml.pre` hardcoded to `"llama"` instead of `"default"` |
| Qwen2-0.5B | `GGML_ASSERT(id_to_token.size() == token_to_id.size()) failed` → crash | Token table has duplicate/missing entries |
| GPT-2 | `key not found: gpt2.attention.layer_norm_epsilon` | apr writes `layer_norm_rms_epsilon` but GPT-2 uses `layer_norm_epsilon` (standard LayerNorm, not RMSNorm) |

### Five-Whys (SmolLM pre-tokenizer)

1. Why does llama.cpp reject? → `unknown pre-tokenizer type: 'llama'`
2. Why is it set to `'llama'`? → apr hardcodes `tokenizer.ggml.pre` based on `general.architecture`
3. Why doesn't that work? → Pre-tokenizer type is a **tokenizer** property, not an architecture property. SmolLM uses GPT-2 BPE, not LLaMA SentencePiece.
4. Why does llama.cpp care? → It uses `tokenizer.ggml.pre` to select regex pre-tokenization patterns
5. Why is this hard? → The mapping from HF tokenizer → GGUF pre-tokenizer is a ~50-entry lookup table in `convert_hf_to_gguf.py`. apr must replicate it.

### Five-Whys (Qwen2 vocab crash)

1. Why crash? → `id_to_token.size() != token_to_id.size()`
2. Why sizes differ? → Duplicate token entries in the exported table
3. Why duplicates? → apr doesn't handle Qwen2's added/special tokens correctly (151,936 vocab with overlapping special tokens)
4. Why is Qwen2 special? → Large vocab with `<|im_start|>` etc. that may overlap base vocabulary indices
5. Why not caught? → No post-export validation that token table is bijective

### Five-Whys (GPT-2 hyperparameter)

1. Why rejected? → `key not found: gpt2.attention.layer_norm_epsilon`
2. Why missing? → apr writes `gpt2.attention.layer_norm_rms_epsilon` instead
3. Why wrong key? → GGUF export uses LLaMA-style keys for all architectures
4. Why wrong for GPT-2? → GPT-2 uses standard LayerNorm (not RMSNorm); GGUF spec has different key
5. Why not caught? → Export path only tested against LLaMA-family architectures

### Popperian Falsification

**Claim**: "apr-exported GGUFs are valid GGUF files loadable by any GGUF-compatible runtime."
**Test**: Load in llama-completion (the reference GGUF implementation).
**Result**: **FALSIFIED** — 3/3 models fail, each for a different reason.
**Implication**: The GGUF export path was not tested against any external consumer.

### What to Fix

1. **`tokenizer.ggml.pre` mapping**: Build a lookup from HF tokenizer class/config to GGUF pre-tokenizer type. Reference: `convert_hf_to_gguf.py` in llama.cpp source (`/home/noah/src/llama.cpp/convert_hf_to_gguf.py`). Key mappings:
   - SmolLM (GPT2 tokenizer) → `"default"`
   - Qwen2 (Qwen2Tokenizer) → `"qwen2"`
   - GPT-2 (GPT2Tokenizer) → `"gpt2"`

2. **Qwen2 token table**: Ensure exported `tokenizer.ggml.tokens` array has no duplicates and `len(tokens) == len(set(token_to_id.values()))`. Debug by comparing apr's exported token count vs llama.cpp's.

3. **Architecture-specific GGUF keys**: GPT-2 must use `gpt2.attention.layer_norm_epsilon` (not `layer_norm_rms_epsilon`). Each architecture has spec-defined keys — reference `llama.cpp/src/llama-arch.cpp` for the canonical key names.

4. **Post-export validation**: After writing the GGUF, validate it loads in gguf-py or llama.cpp's C API. At minimum, verify:
   - All required keys present for the declared architecture
   - Token table is bijective (no duplicate IDs or strings)
   - `tokenizer.ggml.pre` is in the set of known values

### Verification

```bash
# After fixing, rebuild apr
cargo install --path crates/apr-cli

# Re-run Layer 4b tests (should go from xfail → pass)
cd ../tiny-model-ground-truth
make convert  # re-export GGUFs with fixed apr
PATH="/home/noah/src/llama.cpp/build/bin:${PATH}" CUDA_VISIBLE_DEVICES="" \
  uv run --extra test pytest tests/test_llamacpp_parity.py::test_apr_gguf_loads_in_llamacpp -v

# Expected: 3 passed (was: 3 xfailed)
```

---

## Issue 2: GH-278 — Cross-runtime text mismatch (apr vs llama-completion)

### Problem

Same llama.cpp-native GGUF, greedy decoding, same prompt → apr and llama-completion
produce different text. 11/12 test cases fail, only SmolLM/completion matches.

### Results Matrix

| Model | Prompt | Match? | Severity |
|-------|--------|--------|----------|
| SmolLM | arithmetic | DIFFER | Coherent but different |
| SmolLM | code | DIFFER | Coherent but different |
| SmolLM | completion | **MATCH** | Exact text match |
| SmolLM | greeting | DIFFER | Coherent but different |
| Qwen2 | arithmetic | DIFFER | Both coherent, diverge after ~20 chars |
| Qwen2 | code | DIFFER | Start identical, diverge after ~60 chars |
| Qwen2 | completion | DIFFER | Start identical, diverge after 1 sentence |
| Qwen2 | greeting | DIFFER | apr empty, llama produces text |
| GPT-2 | arithmetic | DIFFER | **apr degenerate** (`"I in in in in"`) |
| GPT-2 | code | DIFFER | **apr degenerate** (`"in in ways ways"`) |
| GPT-2 | completion | DIFFER | **apr degenerate** (`"all all all all"`) |
| GPT-2 | greeting | DIFFER | **apr degenerate** (`"and we we we we"`) |

### Five-Whys (GPT-2 degenerate — most severe)

1. Why does apr produce `"all all all all..."` from GPT-2 GGUF? → Stuck in repetition loop
2. Why stuck? → Logits computed incorrectly; argmax always returns same token
3. Why logits wrong? → apr misinterprets GPT-2 weight layout or applies wrong ops
4. Why wrong ops? → GPT-2 uses standard LayerNorm + learned position embeddings (not RoPE), different from LLaMA
5. Why not caught? → apr GGUF inference was only tested against LLaMA-family models

### Five-Whys (Qwen2 divergence)

1. Why diverge after initial agreement? → First tokens high-confidence (same argmax), then numerical differences accumulate
2. Why accumulate? → Autoregressive: small logit diff at token N changes input to N+1, cascading
3. Why any logit diff? → Different dequantization precision, matrix multiply order, or attention scaling
4. Why dequant differs? → apr implements own Q8_0 dequant; llama.cpp uses ggml's. Different rounding or SIMD paths.
5. Why does epsilon matter? → Greedy decoding: 1e-7 logit diff can flip argmax when two tokens have near-equal probability

### Popperian Falsification

**Claim**: "Given the same GGUF and greedy decoding, apr and llama.cpp produce identical text."
**Test**: 3 models × 4 prompts, llama.cpp-native Q8_0 GGUFs, `--temp 0 --top-k 1`.
**Result**: **FALSIFIED** for 11/12 cases. Falsification gradient:
1. **GPT-2**: Total failure. apr produces degenerate text. This is a correctness bug, not precision.
2. **Qwen2**: Partial agreement → divergence. Likely numerical or tokenizer issue.
3. **SmolLM**: 1/4 exact match. Closest to parity. Remaining 3 diverge but are coherent.

### Debugging Strategy

**Priority 1: GPT-2 degenerate output** (correctness bug, not precision)

The fact that apr produces `"all all all all"` while llama-completion produces coherent
text from the same GGUF means apr's GPT-2 inference is fundamentally broken. Debug by:

1. Compare first-token logits between apr and llama.cpp for the prompt `"Hello"`:
   - apr: `apr run models/gpt2-124m-q8_0.gguf --prompt "Hello" --json --max-tokens 1`
   - llama.cpp: add `--logits-all` flag or instrument llama.cpp
2. Check if apr is applying RoPE to GPT-2 (GPT-2 uses learned position embeddings)
3. Check if apr handles `wte` (token embeddings) and `wpe` (position embeddings) correctly for GPT-2 architecture
4. Check if LayerNorm vs RMSNorm is correctly selected based on architecture

**Priority 2: SmolLM remaining 3 prompts**

Since 1/4 already matches, the core inference is nearly correct. Compare token-by-token
to find where the first divergence occurs:

```bash
# Get apr tokens
apr run models/smollm-135m-q8_0.gguf --prompt "Hello" --json --max-tokens 32 | jq .tokens

# llama.cpp doesn't output token IDs by default, so compare decoded text character-by-character
```

**Priority 3: Qwen2**

If SmolLM is fixed, Qwen2 may also improve (shared LLaMA-style architecture). If not,
compare attention patterns or GQA implementation.

### Verification

```bash
# After fixing, rebuild apr
cargo install --path crates/apr-cli

# Re-run Layer 4c tests (should improve from 1/12 → ≥9/12)
cd ../tiny-model-ground-truth
PATH="/home/noah/src/llama.cpp/build/bin:${PATH}" CUDA_VISIBLE_DEVICES="" \
  uv run --extra test pytest tests/test_llamacpp_parity.py::test_cross_runtime_text_match -v

# Expected: majority pass (was: 1 xpassed, 11 xfailed)
```

---

## Full Verification (both issues)

```bash
# Rebuild apr
cd ../aprender && cargo install --path crates/apr-cli

# Re-export GGUFs with fixed apr
cd ../tiny-model-ground-truth
make convert

# Run ALL Layer 4 tests
PATH="/home/noah/src/llama.cpp/build/bin:${PATH}" CUDA_VISIBLE_DEVICES="" \
  uv run --extra test pytest tests/test_llamacpp_parity.py -v --timeout 120

# Run full test suite (no regression)
uv run --extra test pytest tests/ -m "not requires_apr and not requires_llamacpp" --timeout 60

# Expected final state:
# - test_llamacpp_loads_gguf: 6 passed (unchanged)
# - test_llamacpp_q8_text_vs_oracle: mix of xpass/xfail (precision-dependent)
# - test_llamacpp_q4_text_vs_oracle: mix of xpass/xfail (precision-dependent)
# - test_apr_gguf_loads_in_llamacpp: 3 passed (was: 3 xfailed) ← GH-277
# - test_cross_runtime_text_match: ≥9 passed (was: 1 xpassed, 11 xfailed) ← GH-278
# - Unit tests: 357 passed, 96.77% coverage (no regression)
```

## Reference Files

- **llama.cpp pre-tokenizer mapping**: `/home/noah/src/llama.cpp/convert_hf_to_gguf.py` (search for `get_vocab_base_pre`)
- **llama.cpp architecture keys**: `/home/noah/src/llama.cpp/src/llama-arch.cpp` (search for `LLM_KV`)
- **llama.cpp vocab loader**: `/home/noah/src/llama.cpp/src/llama-vocab.cpp` (line 2126 for Qwen2 crash)
- **Test oracle files**: `../tiny-model-ground-truth/oracle/{smollm-135m,qwen2-0.5b,gpt2-124m}/*.json`
- **Native GGUFs**: `../tiny-model-ground-truth/models/*-q8_0.gguf` (llama.cpp-native, known-good)
- **apr-exported GGUFs**: `../tiny-model-ground-truth/models/*-int4.gguf` (apr-exported, broken)
