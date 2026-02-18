"""Layer 4: llama.cpp cross-runtime parity tests.

Layer 4a: llama.cpp native GGUF -> llama-completion inference -> compare vs oracle
Layer 4b: apr-exported GGUF -> llama-completion load test (GH-277 fixed in apr 0.2.18)
Layer 4c: llama.cpp native GGUF -> both apr + llama-completion -> cross-runtime text match

Falsification findings:
  - GGML Q8_0/Q4_0 quantization diverges significantly from transformers float32
    at the character level due to different quantization schemes and autoregressive
    cascade. Word-level overlap shows semantic similarity for some models (SmolLM)
    but total divergence for others (Qwen2 produces Chinese at Q4).
  - apr and llama-completion produce different text from the same GGUF with greedy
    decoding, indicating different weight interpretation or sampling paths (GH-278).

Sample size: n = 12 per test group (3 models x 4 prompts).
"""

import pytest
from helpers import (
    LLAMACPP_MODELS,
    MODELS,
    PROMPTS,
    apr_run_json,
    llamacpp_run,
    load_oracle,
)

pytestmark = [
    pytest.mark.requires_llamacpp,
    pytest.mark.llamacpp_parity,
]

# ── Parametrization ──────────────────────────────────────────────

LLAMACPP_SLUGS = list(LLAMACPP_MODELS.keys())

LLAMACPP_QUANT_PARAMS = [
    (slug, quant)
    for slug in LLAMACPP_SLUGS
    for quant in ["q4_0", "q8_0"]
]

LLAMACPP_MODEL_PROMPT_PARAMS = [
    (slug, prompt) for slug in LLAMACPP_SLUGS for prompt in PROMPTS
]


def _word_overlap(a: str, b: str) -> float:
    """Jaccard word overlap ratio between two strings."""
    words_a = set(a.split())
    words_b = set(b.split())
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


# ── Layer 4a: llama-completion loads native GGUF ─────────────────

@pytest.mark.parametrize("slug,quant", LLAMACPP_QUANT_PARAMS)
def test_llamacpp_loads_gguf(slug: str, quant: str):
    """llama-completion can load each native GGUF and produce output (exit 0)."""
    gguf_path = LLAMACPP_MODELS[slug][quant]
    output, err = llamacpp_run(str(gguf_path), "Hello", n_tokens=4)
    assert err is None, f"llama-completion failed to load {slug}/{quant}: {err}"
    assert output["text"], f"llama-completion produced empty output for {slug}/{quant}"


# ── Layer 4a: llama-completion Q8 text vs oracle (word overlap) ──
# xfail(strict=False): XPASS when overlap is good, XFAIL when divergent

Q8_WORD_OVERLAP = 0.25  # ≥25% word overlap with oracle

@pytest.mark.xfail(strict=False, reason="GGML Q8_0 quantization diverges from float32 oracle")
@pytest.mark.parametrize("slug,prompt_name", LLAMACPP_MODEL_PROMPT_PARAMS)
def test_llamacpp_q8_text_vs_oracle(slug: str, prompt_name: str):
    """llama-completion Q8_0 produces text with ≥25% word overlap vs oracle."""
    oracle = load_oracle(slug, prompt_name)
    gguf_path = LLAMACPP_MODELS[slug]["q8_0"]
    output, err = llamacpp_run(str(gguf_path), oracle["prompt"])
    assert err is None, f"llama-completion failed: {err}"

    actual = output["text"]
    assert actual, "llama-completion produced empty output"
    overlap = _word_overlap(actual, oracle["text"])
    assert overlap >= Q8_WORD_OVERLAP, (
        f"Q8_0 {slug}/{prompt_name}: word overlap {overlap:.2f} < {Q8_WORD_OVERLAP}\n"
        f"  oracle: {oracle['text']!r}\n"
        f"  got:    {actual!r}"
    )


# ── Layer 4a: llama-completion Q4 text vs oracle (word overlap) ──

Q4_WORD_OVERLAP = 0.15  # ≥15% word overlap with oracle

@pytest.mark.xfail(strict=False, reason="GGML Q4_0 quantization diverges from float32 oracle")
@pytest.mark.parametrize("slug,prompt_name", LLAMACPP_MODEL_PROMPT_PARAMS)
def test_llamacpp_q4_text_vs_oracle(slug: str, prompt_name: str):
    """llama-completion Q4_0 produces text with ≥15% word overlap vs oracle."""
    oracle = load_oracle(slug, prompt_name)
    gguf_path = LLAMACPP_MODELS[slug]["q4_0"]
    output, err = llamacpp_run(str(gguf_path), oracle["prompt"])
    assert err is None, f"llama-completion failed: {err}"

    actual = output["text"]
    assert actual, "llama-completion produced empty output"
    overlap = _word_overlap(actual, oracle["text"])
    assert overlap >= Q4_WORD_OVERLAP, (
        f"Q4_0 {slug}/{prompt_name}: word overlap {overlap:.2f} < {Q4_WORD_OVERLAP}\n"
        f"  oracle: {oracle['text']!r}\n"
        f"  got:    {actual!r}"
    )


# ── Layer 4b: apr-exported GGUF in llama-completion ──────────────

@pytest.mark.parametrize("slug", LLAMACPP_SLUGS)
def test_apr_gguf_loads_in_llamacpp(slug: str):
    """apr-exported GGUF loads in llama-completion — GH-277 fixed in apr 0.2.18."""
    apr_gguf = MODELS[slug]["gguf"]
    _output, err = llamacpp_run(str(apr_gguf), "Hello", n_tokens=4)
    assert err is None, f"llama-completion rejected apr GGUF for {slug}: {err}"


# ── Layer 4c: cross-runtime text match ───────────────────────────

@pytest.mark.xfail(reason="apr and llama-completion interpret same GGUF weights differently")
@pytest.mark.requires_apr
@pytest.mark.parametrize("slug,prompt_name", LLAMACPP_MODEL_PROMPT_PARAMS)
def test_cross_runtime_text_match(slug: str, prompt_name: str):
    """Same native GGUF, greedy decoding: apr vs llama-completion must produce same text."""
    oracle = load_oracle(slug, prompt_name)
    gguf_path = str(LLAMACPP_MODELS[slug]["q8_0"])

    apr_out, apr_err = apr_run_json(gguf_path, oracle["prompt"])
    assert apr_err is None, f"apr run failed: {apr_err}"

    llama_out, llama_err = llamacpp_run(gguf_path, oracle["prompt"])
    assert llama_err is None, f"llama-completion failed: {llama_err}"

    apr_text = apr_out.get("text", "") or apr_out.get("generated_text", "")
    llama_text = llama_out["text"]
    assert apr_text == llama_text, (
        f"Cross-runtime mismatch for {slug}/{prompt_name}\n"
        f"  apr:       {apr_text!r}\n"
        f"  llama-completion: {llama_text!r}"
    )
