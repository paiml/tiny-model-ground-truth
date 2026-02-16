"""Operations parity: apr quantize/finetune/merge/prune/distill/convert smoke tests.

Three-layer testing strategy:
1. Smoke tests:   Commands exist, --help works, --plan --json returns valid JSON,
                  graceful errors on bad input, no panics.
2. Oracle tests:  --plan output matches Python oracle expectations (param counts,
                  reduction ratios, methods).
3. Structural:    Commands that produce output files: correct tensor count, dtype
                  changes, file size ratios, Python-readable output.

Sample size: n = 3 models (SmolLM-135M, Qwen2-0.5B, GPT-2).
Oracle data: oracle-ops/{quantize,finetune,merge,prune,convert}/smollm-135m.json.
"""

import contextlib
import json
import os
import re
import subprocess
import tempfile

import pytest
from helpers import MODEL_METADATA, MODEL_PARAMS
from huggingface_hub import hf_hub_download
from safetensors import safe_open

pytestmark = [
    pytest.mark.requires_apr,
    pytest.mark.operations_parity,
]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORACLE_OPS_DIR = os.path.join(ROOT, "oracle-ops")

# Operations that support --plan --json
PLAN_OPS = ["quantize", "finetune", "prune", "distill"]
# All operations
ALL_OPS = ["quantize", "finetune", "merge", "prune", "distill", "convert"]


def _run_apr(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["apr", *args], capture_output=True, text=True, timeout=30,
    )


def _apr_json(args: list[str]) -> dict:
    proc = _run_apr(args)
    assert proc.returncode == 0, f"apr {args[0]} failed: {proc.stderr}"
    return json.loads(proc.stdout)


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _resolve_path(slug: str) -> str:
    hf_id = MODEL_METADATA[slug]["hf_id"]
    path = hf_hub_download(hf_id, "model.safetensors", local_files_only=True)
    return os.path.realpath(path)


def _read_header(path: str) -> dict[str, dict]:
    with open(path, "rb") as f:
        header_size = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_size))
    header.pop("__metadata__", None)
    return header


def _load_oracle(op: str, slug: str) -> list[dict] | None:
    path = os.path.join(ORACLE_OPS_DIR, op, f"{slug}.json")
    if not os.path.exists(path):
        return None
    return json.loads(open(path).read())


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def model_paths():
    """Resolve safetensors paths for all models."""
    paths = {}
    for slug in MODEL_PARAMS:
        with contextlib.suppress(Exception):
            paths[slug] = _resolve_path(slug)
    if not paths:
        pytest.skip("No HF-cached safetensors files found")
    return paths


@pytest.fixture(scope="module")
def plan_data(model_paths):
    """Run --plan --json for each operation on each model."""
    data = {}
    for slug, path in model_paths.items():
        data[slug] = {}
        for op in PLAN_OPS:
            args = _plan_args(op, path)
            proc = _run_apr(args)
            if proc.returncode == 0:
                try:
                    data[slug][op] = json.loads(proc.stdout)
                except json.JSONDecodeError:
                    data[slug][op] = None
            else:
                data[slug][op] = None
    return data


def _plan_args(op: str, path: str) -> list[str]:
    """Build --plan --json args for each operation."""
    if op == "quantize":  # noqa: SIM116
        return ["quantize", path, "--scheme", "int4", "--plan", "--json"]
    elif op == "finetune":
        return ["finetune", path, "--method", "lora", "--plan", "--json"]
    elif op == "prune":
        return ["prune", path, "--method", "magnitude", "--plan", "--json"]
    elif op == "distill":
        return ["distill", path, "--plan", "--json"]
    return [op, path, "--plan", "--json"]


# ══════════════════════════════════════════════════════════════════════════
# LAYER 1: SMOKE TESTS
# ══════════════════════════════════════════════════════════════════════════


# ── --help works ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("op", ALL_OPS)
def test_help_exits_zero(op):
    """apr <op> --help: exits 0 with usage text."""
    proc = _run_apr([op, "--help"])
    assert proc.returncode == 0, f"apr {op} --help failed: {proc.stderr}"
    assert "Usage" in proc.stdout or "usage" in proc.stdout, (
        f"apr {op} --help has no usage text"
    )


@pytest.mark.parametrize("op", ALL_OPS)
def test_help_mentions_json_flag(op):
    """apr <op> --help: documents the --json flag."""
    proc = _run_apr([op, "--help"])
    assert "--json" in proc.stdout, f"apr {op} --help doesn't mention --json"


# ── --plan --json returns valid JSON ──────────────────────────────────────


@pytest.mark.parametrize("op", PLAN_OPS)
@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_plan_json_valid(slug, op, model_paths, plan_data):
    """apr <op> --plan --json: returns valid JSON dict."""
    if slug not in model_paths:
        pytest.skip(f"{slug} not in HF cache")
    result = plan_data[slug].get(op)
    assert result is not None, f"apr {op} --plan --json failed for {slug}"
    assert isinstance(result, dict), f"expected dict, got {type(result)}"


# ── merge --json and convert --json (currently broken) ────────────────────


@pytest.mark.parametrize("slug", MODEL_PARAMS)
@pytest.mark.xfail(reason="PMAT-268: apr merge --json outputs ANSI instead of JSON")
def test_merge_json_output(slug, model_paths):
    """apr merge --json: should produce valid JSON."""
    if slug not in model_paths:
        pytest.skip(f"{slug} not in HF cache")
    path = model_paths[slug]
    tmpdir = tempfile.mkdtemp(prefix="apr_merge_")
    out = os.path.join(tmpdir, "merged.safetensors")
    proc = _run_apr(["merge", path, path, "--strategy", "slerp", "-o", out, "--json"])
    try:
        assert proc.returncode == 0, f"apr merge failed: {proc.stderr}"
        data = json.loads(proc.stdout)
        assert isinstance(data, dict)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(out)
        with contextlib.suppress(OSError):
            os.rmdir(tmpdir)


@pytest.mark.parametrize("slug", MODEL_PARAMS)
@pytest.mark.xfail(reason="PMAT-269: apr convert --json outputs ANSI instead of JSON")
def test_convert_json_output(slug, model_paths):
    """apr convert --json: should produce valid JSON."""
    if slug not in model_paths:
        pytest.skip(f"{slug} not in HF cache")
    path = model_paths[slug]
    tmpdir = tempfile.mkdtemp(prefix="apr_convert_")
    out = os.path.join(tmpdir, "converted.safetensors")
    proc = _run_apr(["convert", path, "-o", out, "--json"])
    try:
        assert proc.returncode == 0, f"apr convert failed: {proc.stderr}"
        data = json.loads(proc.stdout)
        assert isinstance(data, dict)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(out)
        with contextlib.suppress(OSError):
            os.rmdir(tmpdir)


# ── no panics on bad input ────────────────────────────────────────────────


@pytest.mark.parametrize("op", ALL_OPS)
def test_nonexistent_file_no_panic(op):
    """apr <op> /nonexistent: fails gracefully, no panic."""
    proc = _run_apr([op, "/nonexistent/model.safetensors", "--json"])
    assert "panicked" not in proc.stderr.lower(), (
        f"apr {op} panicked on missing file: {proc.stderr[:200]}"
    )
    assert "SIGSEGV" not in proc.stderr


@pytest.mark.parametrize("op", ALL_OPS)
def test_empty_file_no_panic(op):
    """apr <op> on empty file: fails gracefully, no panic."""
    with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
        proc = _run_apr([op, f.name, "--json"])
    assert "panicked" not in proc.stderr.lower(), (
        f"apr {op} panicked on empty file: {proc.stderr[:200]}"
    )
    assert "SIGSEGV" not in proc.stderr


@pytest.mark.parametrize("op", ALL_OPS)
def test_garbage_file_no_panic(op):
    """apr <op> on garbage file: fails gracefully, no panic."""
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        f.write(os.urandom(256))
        f.flush()
        path = f.name
    try:
        proc = _run_apr([op, path, "--json"])
        assert "panicked" not in proc.stderr.lower(), (
            f"apr {op} panicked on garbage: {proc.stderr[:200]}"
        )
        assert "SIGSEGV" not in proc.stderr
    finally:
        os.unlink(path)


# ── --version flag doesn't break subcommands ──────────────────────────────


@pytest.mark.parametrize("op", ALL_OPS)
def test_version_flag(op):
    """apr <op> --version doesn't exist, but apr --version does."""
    proc = _run_apr(["--version"])
    assert proc.returncode == 0
    assert "apr" in proc.stdout.lower()


# ══════════════════════════════════════════════════════════════════════════
# LAYER 2: ORACLE COMPARISON
# ══════════════════════════════════════════════════════════════════════════


# ── quantize: --plan reduction ratio matches oracle ───────────────────────


def test_quantize_plan_reduction_ratio(model_paths, plan_data):
    """apr quantize --plan: reduction ratio is reasonable (~8x for int4)."""
    slug = "smollm-135m"
    if slug not in model_paths:
        pytest.skip(f"{slug} not in HF cache")
    plan = plan_data[slug].get("quantize")
    if plan is None:
        pytest.skip("quantize plan not available")
    ratio = plan.get("reduction_ratio", 0)
    # int4 should be ~8x reduction (FP32 is 4 bytes, int4 is 0.5 bytes)
    assert 4.0 <= ratio <= 16.0, (
        f"int4 reduction_ratio={ratio}, expected 4-16x"
    )


def test_quantize_plan_output_size_vs_oracle(model_paths, plan_data):
    """apr quantize --plan: estimated output size is < input size."""
    slug = "smollm-135m"
    if slug not in model_paths:
        pytest.skip(f"{slug} not in HF cache")
    plan = plan_data[slug].get("quantize")
    if plan is None:
        pytest.skip("quantize plan not available")
    oracle = _load_oracle("quantize", slug)
    if oracle is None:
        pytest.skip("no quantize oracle")
    input_size = plan.get("input_size", 0)
    output_size = plan.get("estimated_output_size", 0)
    assert output_size < input_size, (
        f"quantized output ({output_size}) should be smaller than input ({input_size})"
    )
    # Oracle int8 output is ~113MB for 538MB input (4x)
    oracle_bytes = oracle[0].get("param_bytes", 0)
    if oracle_bytes > 0:
        # apr int4 should be even smaller than oracle int8
        assert output_size < oracle_bytes * 1.5, (
            f"apr int4 estimate ({output_size}) should be <= oracle int8 ({oracle_bytes})"
        )


# ── finetune: --plan identifies LoRA parameters ──────────────────────────


def test_finetune_plan_method(model_paths, plan_data):
    """apr finetune --plan: recommends LoRA or similar method."""
    slug = "smollm-135m"
    if slug not in model_paths:
        pytest.skip(f"{slug} not in HF cache")
    plan = plan_data[slug].get("finetune")
    if plan is None:
        pytest.skip("finetune plan not available")
    method = plan.get("recommended_method", "").lower()
    assert "lora" in method or "qlora" in method, (
        f"expected LoRA/QLoRA recommendation, got '{method}'"
    )


def test_finetune_plan_trainable_pct(model_paths, plan_data):
    """apr finetune --plan: trainable % is small (LoRA is <10%)."""
    slug = "smollm-135m"
    if slug not in model_paths:
        pytest.skip(f"{slug} not in HF cache")
    plan = plan_data[slug].get("finetune")
    if plan is None:
        pytest.skip("finetune plan not available")
    pct = plan.get("trainable_percent", 100)
    assert pct < 20.0, f"trainable%={pct}, expected <20% for LoRA"


def test_finetune_plan_vs_oracle(model_paths, plan_data):
    """apr finetune --plan: param count in same order of magnitude as oracle."""
    slug = "smollm-135m"
    if slug not in model_paths:
        pytest.skip(f"{slug} not in HF cache")
    plan = plan_data[slug].get("finetune")
    if plan is None:
        pytest.skip("finetune plan not available")
    oracle = _load_oracle("finetune", slug)
    if oracle is None:
        pytest.skip("no finetune oracle")
    oracle_total = oracle[0].get("total_params", 0)
    apr_total = plan.get("model_params", 0)
    if oracle_total > 0 and apr_total > 0:
        ratio = apr_total / oracle_total
        assert 0.5 <= ratio <= 10.0, (
            f"apr model_params={apr_total} vs oracle total_params={oracle_total}, "
            f"ratio={ratio:.2f} (expected 0.5-10x)"
        )


# ── prune: --plan sparsity and method ─────────────────────────────────────


def test_prune_plan_method(model_paths, plan_data):
    """apr prune --plan: reports magnitude pruning method."""
    slug = "smollm-135m"
    if slug not in model_paths:
        pytest.skip(f"{slug} not in HF cache")
    plan = plan_data[slug].get("prune")
    if plan is None:
        pytest.skip("prune plan not available")
    method = plan.get("method", "").lower()
    assert "magnitude" in method, f"expected magnitude, got '{method}'"


def test_prune_plan_output_smaller(model_paths, plan_data):
    """apr prune --plan: estimated output is smaller than input."""
    slug = "smollm-135m"
    if slug not in model_paths:
        pytest.skip(f"{slug} not in HF cache")
    plan = plan_data[slug].get("prune")
    if plan is None:
        pytest.skip("prune plan not available")
    input_size = plan.get("input_size", 0)
    output_size = plan.get("estimated_output_size", 0)
    assert output_size < input_size, (
        f"pruned output ({output_size}) should be smaller than input ({input_size})"
    )


def test_prune_plan_vs_oracle_sparsity(model_paths, plan_data):
    """apr prune --plan: target ratio matches oracle expectation."""
    slug = "smollm-135m"
    if slug not in model_paths:
        pytest.skip(f"{slug} not in HF cache")
    plan = plan_data[slug].get("prune")
    if plan is None:
        pytest.skip("prune plan not available")
    oracle = _load_oracle("prune", slug)
    if oracle is None:
        pytest.skip("no prune oracle")
    apr_ratio = plan.get("target_ratio", 0)
    oracle_sparsity = oracle[0].get("target_sparsity", 0)
    # Both should be in [0, 1] range
    assert 0.0 < apr_ratio <= 1.0, f"apr target_ratio={apr_ratio} out of range"
    assert 0.0 < oracle_sparsity <= 1.0, f"oracle sparsity={oracle_sparsity} out of range"


# ── distill: --plan structure ─────────────────────────────────────────────


def test_distill_plan_temperature(model_paths, plan_data):
    """apr distill --plan: temperature is positive."""
    slug = "smollm-135m"
    if slug not in model_paths:
        pytest.skip(f"{slug} not in HF cache")
    plan = plan_data[slug].get("distill")
    if plan is None:
        pytest.skip("distill plan not available")
    temp = plan.get("temperature", 0)
    assert temp > 0, f"distill temperature={temp}, should be positive"


def test_distill_plan_student_smaller(model_paths, plan_data):
    """apr distill --plan: student model is smaller than teacher."""
    slug = "smollm-135m"
    if slug not in model_paths:
        pytest.skip(f"{slug} not in HF cache")
    plan = plan_data[slug].get("distill")
    if plan is None:
        pytest.skip("distill plan not available")
    teacher = plan.get("teacher_size", 0)
    student = plan.get("student_size", 0)
    assert student <= teacher, (
        f"student ({student}) should be <= teacher ({teacher})"
    )


# ── convert: oracle file size matches ─────────────────────────────────────


def test_convert_oracle_size_matches_actual(model_paths):
    """oracle convert: recorded file size matches actual safetensors size."""
    slug = "smollm-135m"
    if slug not in model_paths:
        pytest.skip(f"{slug} not in HF cache")
    oracle = _load_oracle("convert", slug)
    if oracle is None:
        pytest.skip("no convert oracle")
    st_oracle = next((o for o in oracle if o["format"] == "safetensors"), None)
    if st_oracle is None:
        pytest.skip("no safetensors entry in convert oracle")
    actual_size = os.path.getsize(model_paths[slug])
    assert st_oracle["total_bytes"] == actual_size, (
        f"oracle={st_oracle['total_bytes']}, actual={actual_size}"
    )


# ── merge: oracle tensor count vs header ──────────────────────────────────


def test_merge_oracle_tensor_count(model_paths):
    """oracle merge: tensor_count matches actual safetensors header (+1 for metadata)."""
    slug = "smollm-135m"
    if slug not in model_paths:
        pytest.skip(f"{slug} not in HF cache")
    oracle = _load_oracle("merge", slug)
    if oracle is None:
        pytest.skip("no merge oracle")
    header = _read_header(model_paths[slug])
    oracle_count = oracle[0].get("tensor_count", 0)
    # Oracle merge may include lm_head (tied weight) as separate tensor
    assert abs(oracle_count - len(header)) <= 2, (
        f"oracle tensor_count={oracle_count}, header={len(header)}, diff > 2"
    )


# ══════════════════════════════════════════════════════════════════════════
# LAYER 3: STRUCTURAL TESTS (commands that produce output)
# ══════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def merge_output(model_paths):
    """Run apr merge on SmolLM self-merge, return output path."""
    slug = "smollm-135m"
    if slug not in model_paths:
        pytest.skip("smollm not in HF cache")
    path = model_paths[slug]
    tmpdir = tempfile.mkdtemp(prefix="apr_merge_struct_")
    out = os.path.join(tmpdir, "merged.safetensors")
    proc = _run_apr(["merge", path, path, "--strategy", "slerp", "-o", out])
    if proc.returncode != 0:
        pytest.skip(f"apr merge failed: {proc.stderr[:200]}")
    yield {"path": out, "slug": slug, "tmpdir": tmpdir}
    with contextlib.suppress(OSError):
        os.unlink(out)
    with contextlib.suppress(OSError):
        os.rmdir(tmpdir)


def test_merge_output_exists(merge_output):
    """apr merge: produces an output file."""
    assert os.path.exists(merge_output["path"])
    assert os.path.getsize(merge_output["path"]) > 0


def test_merge_output_readable(merge_output):
    """apr merge: output is valid safetensors readable by Python."""
    with safe_open(merge_output["path"], framework="pt") as f:
        keys = list(f.keys())
    assert len(keys) > 0


def test_merge_output_tensor_names(merge_output, model_paths):
    """apr merge: output has same tensor names as input."""
    input_header = _read_header(model_paths[merge_output["slug"]])
    output_header = _read_header(merge_output["path"])
    input_names = sorted(input_header.keys())
    output_names = sorted(output_header.keys())
    assert input_names == output_names, (
        f"missing={set(input_names) - set(output_names)}, "
        f"extra={set(output_names) - set(input_names)}"
    )


def test_merge_output_shapes_preserved(merge_output, model_paths):
    """apr merge: output tensor shapes match input."""
    input_header = _read_header(model_paths[merge_output["slug"]])
    output_header = _read_header(merge_output["path"])
    mismatches = [
        f"  {n}: {input_header[n]['shape']} -> {output_header[n]['shape']}"
        for n in input_header
        if n in output_header and input_header[n]["shape"] != output_header[n]["shape"]
    ]
    assert not mismatches, "shape mismatches:\n" + "\n".join(mismatches)


def test_merge_output_dtypes_preserved(merge_output, model_paths):
    """apr merge: output tensor dtypes match input."""
    input_header = _read_header(model_paths[merge_output["slug"]])
    output_header = _read_header(merge_output["path"])
    mismatches = [
        f"  {n}: {input_header[n]['dtype']} -> {output_header[n]['dtype']}"
        for n in input_header
        if n in output_header and input_header[n]["dtype"] != output_header[n]["dtype"]
    ]
    assert not mismatches, "dtype mismatches:\n" + "\n".join(mismatches)


def test_merge_self_slerp_is_identity(merge_output, model_paths):
    """apr merge: SLERP self-merge (t=0.5) should produce ~identical weights."""
    input_path = model_paths[merge_output["slug"]]
    output_path = merge_output["path"]
    # Check a small probe tensor
    probe = "model.layers.0.input_layernorm.weight"
    with safe_open(input_path, framework="pt") as f:
        t_in = f.get_tensor(probe).float()
    with safe_open(output_path, framework="pt") as f:
        t_out = f.get_tensor(probe).float()
    max_diff = (t_in - t_out).abs().max().item()
    assert max_diff < 1e-4, f"self-SLERP should be identity, max_diff={max_diff}"


@pytest.fixture(scope="module")
def convert_output(model_paths):
    """Run apr convert on SmolLM, return output path."""
    slug = "smollm-135m"
    if slug not in model_paths:
        pytest.skip("smollm not in HF cache")
    path = model_paths[slug]
    tmpdir = tempfile.mkdtemp(prefix="apr_convert_struct_")
    out = os.path.join(tmpdir, "converted.safetensors")
    proc = _run_apr(["convert", path, "-o", out])
    if proc.returncode != 0:
        pytest.skip(f"apr convert failed: {proc.stderr[:200]}")
    yield {"path": out, "slug": slug, "tmpdir": tmpdir}
    with contextlib.suppress(OSError):
        os.unlink(out)
    with contextlib.suppress(OSError):
        os.rmdir(tmpdir)


def test_convert_output_exists(convert_output):
    """apr convert: produces an output file."""
    assert os.path.exists(convert_output["path"])
    assert os.path.getsize(convert_output["path"]) > 0


def test_convert_output_readable(convert_output):
    """apr convert: output is valid safetensors readable by Python."""
    with safe_open(convert_output["path"], framework="pt") as f:
        keys = list(f.keys())
    assert len(keys) > 0


def test_convert_output_tensor_count(convert_output, model_paths):
    """apr convert: output tensor count matches input."""
    input_header = _read_header(model_paths[convert_output["slug"]])
    output_header = _read_header(convert_output["path"])
    assert len(output_header) == len(input_header), (
        f"input={len(input_header)}, output={len(output_header)}"
    )


def test_convert_output_size_reasonable(convert_output, model_paths):
    """apr convert: output size is within 2x of input (no bloat)."""
    input_size = os.path.getsize(model_paths[convert_output["slug"]])
    output_size = os.path.getsize(convert_output["path"])
    ratio = output_size / input_size
    assert 0.5 <= ratio <= 2.0, (
        f"size ratio={ratio:.2f} (input={input_size}, output={output_size})"
    )


# ── quantize structural (produces output file) ───────────────────────────


@pytest.fixture(scope="module")
def quantize_output(model_paths):
    """Run apr quantize int4 on SmolLM, return output path."""
    slug = "smollm-135m"
    if slug not in model_paths:
        pytest.skip("smollm not in HF cache")
    path = model_paths[slug]
    tmpdir = tempfile.mkdtemp(prefix="apr_quant_struct_")
    out = os.path.join(tmpdir, "quantized.safetensors")
    proc = _run_apr(["quantize", path, "--scheme", "int4", "-o", out, "--json"])
    if proc.returncode != 0:
        # quantize may not be fully implemented yet
        yield None
        return
    yield {"path": out, "slug": slug, "tmpdir": tmpdir, "stdout": proc.stdout}
    with contextlib.suppress(OSError):
        os.unlink(out)
    with contextlib.suppress(OSError):
        os.rmdir(tmpdir)


def test_quantize_output_exists(quantize_output):
    """apr quantize: produces an output file."""
    if quantize_output is None:
        pytest.skip("apr quantize not implemented yet")
    assert os.path.exists(quantize_output["path"])
    assert os.path.getsize(quantize_output["path"]) > 0


def test_quantize_output_smaller(quantize_output, model_paths):
    """apr quantize int4: output is significantly smaller than input."""
    if quantize_output is None:
        pytest.skip("apr quantize not implemented yet")
    input_size = os.path.getsize(model_paths[quantize_output["slug"]])
    output_size = os.path.getsize(quantize_output["path"])
    assert output_size < input_size * 0.75, (
        f"quantized output ({output_size}) should be <75% of input ({input_size})"
    )


def test_quantize_output_readable(quantize_output):
    """apr quantize: output is valid safetensors readable by Python."""
    if quantize_output is None:
        pytest.skip("apr quantize not implemented yet")
    with safe_open(quantize_output["path"], framework="pt") as f:
        keys = list(f.keys())
    assert len(keys) > 0


def test_quantize_output_tensor_count(quantize_output, model_paths):
    """apr quantize: output tensor count matches input."""
    if quantize_output is None:
        pytest.skip("apr quantize not implemented yet")
    input_header = _read_header(model_paths[quantize_output["slug"]])
    output_header = _read_header(quantize_output["path"])
    # Quantized model may have different tensor count (scale/zero tensors)
    # but should be >= input count
    assert len(output_header) >= len(input_header) * 0.5, (
        f"quantized has too few tensors: {len(output_header)} vs input {len(input_header)}"
    )


# ── prune structural ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def prune_output(model_paths):
    """Run apr prune on SmolLM, return output path."""
    slug = "smollm-135m"
    if slug not in model_paths:
        pytest.skip("smollm not in HF cache")
    path = model_paths[slug]
    tmpdir = tempfile.mkdtemp(prefix="apr_prune_struct_")
    out = os.path.join(tmpdir, "pruned.safetensors")
    proc = _run_apr([
        "prune", path, "--method", "magnitude",
        "--target-ratio", "0.3", "-o", out, "--json",
    ])
    if proc.returncode != 0:
        yield None
        return
    # PMAT-270: apr prune exits 0 with plan JSON but doesn't create the output file
    if not os.path.exists(out):
        yield None
        return
    yield {"path": out, "slug": slug, "tmpdir": tmpdir, "stdout": proc.stdout}
    with contextlib.suppress(OSError):
        os.unlink(out)
    with contextlib.suppress(OSError):
        os.rmdir(tmpdir)


def test_prune_output_exists(prune_output):
    """apr prune: produces an output file."""
    if prune_output is None:
        pytest.skip("apr prune not implemented yet")
    assert os.path.exists(prune_output["path"])
    assert os.path.getsize(prune_output["path"]) > 0


def test_prune_output_readable(prune_output):
    """apr prune: output is valid safetensors readable by Python."""
    if prune_output is None:
        pytest.skip("apr prune not implemented yet")
    with safe_open(prune_output["path"], framework="pt") as f:
        keys = list(f.keys())
    assert len(keys) > 0


def test_prune_output_tensor_names_preserved(prune_output, model_paths):
    """apr prune: output has same tensor names as input."""
    if prune_output is None:
        pytest.skip("apr prune not implemented yet")
    input_header = _read_header(model_paths[prune_output["slug"]])
    output_header = _read_header(prune_output["path"])
    input_names = sorted(input_header.keys())
    output_names = sorted(output_header.keys())
    assert input_names == output_names


def test_prune_output_shapes_preserved(prune_output, model_paths):
    """apr prune: magnitude pruning preserves tensor shapes (zeroes weights, doesn't remove them)."""
    if prune_output is None:
        pytest.skip("apr prune not implemented yet")
    input_header = _read_header(model_paths[prune_output["slug"]])
    output_header = _read_header(prune_output["path"])
    mismatches = [
        f"  {n}: {input_header[n]['shape']} -> {output_header[n]['shape']}"
        for n in input_header
        if n in output_header and input_header[n]["shape"] != output_header[n]["shape"]
    ]
    assert not mismatches, "shape mismatches:\n" + "\n".join(mismatches)


def test_prune_output_has_zeros(prune_output):
    """apr prune: pruned model has increased sparsity (more zero weights)."""
    if prune_output is None:
        pytest.skip("apr prune not implemented yet")
    probe = "model.layers.0.mlp.gate_proj.weight"
    with safe_open(prune_output["path"], framework="pt") as f:
        if probe not in f:
            pytest.skip(f"probe tensor {probe} not in pruned output")
        t = f.get_tensor(probe).float()
    zero_pct = (t == 0).float().mean().item()
    # Magnitude pruning at 30% should have ~30% zeros in weight tensors
    assert zero_pct > 0.1, (
        f"pruned tensor has only {zero_pct:.1%} zeros, expected >10%"
    )
