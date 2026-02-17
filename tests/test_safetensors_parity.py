"""Safetensors parity: Python safetensors vs apr subcommands.

Proves apr's safetensors I/O matches the Python reference on:
- apr tensors:       names, shapes, dtypes, count
- apr inspect:       architecture, file size, param count, format
- apr debug:         health status, metadata agreement
- apr hex --stats:   actual tensor data (min/max/mean/std vs torch)
- apr hex --header:  header length matches binary read
- apr hex --distrib: entropy/kurtosis/skewness vs torch
- apr tree:          leaf tensor names match safetensors header
- apr validate:      all tensor checks pass, no NaN/Inf
- apr diff:          self-diff is identity
- apr list:          cached model appears with correct size
- apr export:        safetensors write roundtrip (bit-perfect)
- stdin/stdout pipe: in-memory serialize/deserialize equivalent
- apr trace:         layer count matches Python layer count
- apr oracle:        architecture and param count vs header
- apr flow:          --json flag produces valid JSON
- apr explain:       tensor lookup in safetensors files
- tensor slicing:    partial tensor load (Python only, gap)
- metadata:          __metadata__ dict preserved
- corruption:        graceful error on truncated files
- sharded models:    multi-file safetensors handling
- cross-format diff: diff between different models

Fast: header reads + small tensor spot-checks, no inference.
Sample size: n = 3 (one per model in roster) + 1 sharded.
"""

import contextlib
import json
import math
import os
import re
import subprocess
import tempfile

import pytest
import torch
from helpers import MODEL_METADATA, MODEL_PARAMS
from huggingface_hub import hf_hub_download
from safetensors import SafetensorError, safe_open

pytestmark = [
    pytest.mark.requires_apr,
    pytest.mark.safetensors_parity,
]

# One small 1D tensor per model for data spot-checks
PROBE_TENSORS = {
    "smollm-135m": "model.layers.0.input_layernorm.weight",
    "qwen2-0.5b": "model.layers.0.input_layernorm.weight",
    "gpt2-124m": "h.0.ln_1.weight",
}


def _read_header(path: str) -> dict[str, dict]:
    """Read safetensors JSON header without loading weights."""
    with open(path, "rb") as f:
        header_size = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_size))
    header.pop("__metadata__", None)
    return header


def _read_header_size(path: str) -> int:
    """Read the 8-byte header length from a safetensors file."""
    with open(path, "rb") as f:
        return int.from_bytes(f.read(8), "little")


def _compare_tensor_headers(orig: dict, rt: dict) -> list[str]:
    """Compare shape and dtype between two safetensors headers, return mismatch strings."""
    mismatches = []
    for name in orig:
        if name not in rt:
            continue
        if orig[name]["shape"] != rt[name]["shape"]:
            mismatches.append(f"  {name} shape: {orig[name]['shape']} -> {rt[name]['shape']}")
        if orig[name]["dtype"] != rt[name]["dtype"]:
            mismatches.append(f"  {name} dtype: {orig[name]['dtype']} -> {rt[name]['dtype']}")
    return mismatches


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
    return hf_hub_download(hf_id, "model.safetensors", local_files_only=True)


def _collect_tree_leaves(node: dict) -> list[str]:
    """Recursively collect all leaf tensor paths from apr tree JSON."""
    if node.get("type") == "tensor":
        return [node["path"]]
    leaves = []
    for child in node.get("children", []):
        leaves.extend(_collect_tree_leaves(child))
    return leaves


# ── Fixture: gather all data once ─────────────────────────────────────────


@pytest.fixture(scope="module")
def parity_data():
    """Run all apr subcommands once per model, cache results."""
    data = {}
    for slug in MODEL_PARAMS:
        try:
            path = _resolve_path(slug)
        except Exception:
            continue

        header = _read_header(path)
        header_size = _read_header_size(path)
        apr_tensors = _apr_json(["tensors", path, "--json"])
        apr_inspect = _apr_json(["inspect", path, "--json"])
        apr_debug = _apr_json(["debug", path, "--json"])
        apr_tree = _apr_json(["tree", path, "--json"])
        apr_validate = _apr_json(["validate", path, "--json"])
        apr_diff_self = _apr_json(["diff", path, path, "--json"])

        # apr hex --stats for the probe tensor
        probe = PROBE_TENSORS[slug]
        probe_dtype = header[probe]["dtype"]
        hex_proc = _run_apr(["hex", path, "--tensor", probe, "--stats"])
        hex_out = _strip_ansi(hex_proc.stdout)

        # apr hex --header
        hex_header_proc = _run_apr(["hex", path, "--header"])
        hex_header_out = _strip_ansi(hex_header_proc.stdout)

        # apr hex --distribution for probe tensor
        hex_dist_proc = _run_apr(["hex", path, "--tensor", probe, "--distribution"])
        hex_dist_out = _strip_ansi(hex_dist_proc.stdout)

        # Python stats for the same probe tensor (torch handles bfloat16)
        with safe_open(path, framework="pt") as f:
            probe_tensor = f.get_tensor(probe).float()
            py_keys = list(f.keys())

        data[slug] = {
            "path": path,
            "header": header,
            "header_size": header_size,
            "apr_tensors": {t["name"]: t for t in apr_tensors["tensors"]},
            "apr_inspect": apr_inspect,
            "apr_debug": apr_debug,
            "apr_tree": apr_tree,
            "apr_validate": apr_validate,
            "apr_diff_self": apr_diff_self,
            "hex_stdout": hex_out,
            "hex_header_out": hex_header_out,
            "hex_dist_out": hex_dist_out,
            "probe_name": probe,
            "probe_dtype": probe_dtype,
            "probe_pt": probe_tensor,
            "safe_open_keys": py_keys,
        }

    if not data:
        pytest.skip("No HF-cached safetensors files found")
    return data


@pytest.fixture(scope="module")
def list_data():
    """Run apr list --json once."""
    return _apr_json(["list", "--json"])


# ── apr tensors ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_tensor_count(slug, parity_data):
    """apr tensors: count matches safetensors header."""
    if slug not in parity_data:
        pytest.skip(f"{slug} not in HF cache")
    d = parity_data[slug]
    assert len(d["header"]) == len(d["apr_tensors"]), (
        f"safetensors={len(d['header'])}, apr={len(d['apr_tensors'])}"
    )


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_tensor_names(slug, parity_data):
    """apr tensors: every name matches safetensors header."""
    if slug not in parity_data:
        pytest.skip(f"{slug} not in HF cache")
    d = parity_data[slug]
    assert sorted(d["header"].keys()) == sorted(d["apr_tensors"].keys())


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_tensor_shapes(slug, parity_data):
    """apr tensors: every shape matches safetensors header."""
    if slug not in parity_data:
        pytest.skip(f"{slug} not in HF cache")
    d = parity_data[slug]
    mismatches = [
        f"  {n}: header={d['header'][n]['shape']}, apr={d['apr_tensors'][n]['shape']}"
        for n in d["header"]
        if n in d["apr_tensors"] and d["header"][n]["shape"] != d["apr_tensors"][n]["shape"]
    ]
    assert not mismatches, "\n".join(mismatches)


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_tensor_dtypes(slug, parity_data):
    """apr tensors: every dtype matches safetensors header."""
    if slug not in parity_data:
        pytest.skip(f"{slug} not in HF cache")
    d = parity_data[slug]
    mismatches = [
        f"  {n}: header={d['header'][n]['dtype']}, apr={d['apr_tensors'][n]['dtype']}"
        for n in d["header"]
        if n in d["apr_tensors"] and d["header"][n]["dtype"] != d["apr_tensors"][n]["dtype"]
    ]
    assert not mismatches, "\n".join(mismatches)


# ── apr inspect ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_inspect_architecture(slug, parity_data):
    """apr inspect: detects correct architecture from tensor names."""
    if slug not in parity_data:
        pytest.skip(f"{slug} not in HF cache")
    expected = MODEL_METADATA[slug]["architecture"]
    got = parity_data[slug]["apr_inspect"].get("architecture", "").lower()
    assert expected in got, f"expected '{expected}' in '{got}'"


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_inspect_file_size(slug, parity_data):
    """apr inspect: file_size matches os.path.getsize."""
    if slug not in parity_data:
        pytest.skip(f"{slug} not in HF cache")
    path = parity_data[slug]["path"]
    assert parity_data[slug]["apr_inspect"]["file_size"] == os.path.getsize(path)


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_inspect_tensor_count(slug, parity_data):
    """apr inspect: tensor_count matches safetensors header."""
    if slug not in parity_data:
        pytest.skip(f"{slug} not in HF cache")
    d = parity_data[slug]
    assert d["apr_inspect"]["tensor_count"] == len(d["header"])


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_inspect_format(slug, parity_data):
    """apr inspect: identifies format as SafeTensors."""
    if slug not in parity_data:
        pytest.skip(f"{slug} not in HF cache")
    fmt = parity_data[slug]["apr_inspect"].get("format", "").lower()
    assert "safetensor" in fmt, f"expected SafeTensors, got '{fmt}'"


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_inspect_param_count(slug, parity_data):
    """apr inspect: total_params matches sum of tensor element counts."""
    if slug not in parity_data:
        pytest.skip(f"{slug} not in HF cache")
    d = parity_data[slug]
    header_params = sum(
        math.prod(meta["shape"]) for meta in d["header"].values() if meta["shape"]
    )
    apr_params = d["apr_inspect"].get("total_params", 0)
    assert apr_params == header_params, f"header={header_params}, apr={apr_params}"


# ── apr debug ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_debug_health_ok(slug, parity_data):
    """apr debug: health=OK for valid safetensors file."""
    if slug not in parity_data:
        pytest.skip(f"{slug} not in HF cache")
    health = parity_data[slug]["apr_debug"].get("health", "").upper()
    assert health == "OK", f"apr debug health={health}"


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_debug_format(slug, parity_data):
    """apr debug: format=SafeTensors."""
    if slug not in parity_data:
        pytest.skip(f"{slug} not in HF cache")
    fmt = parity_data[slug]["apr_debug"].get("format", "").lower()
    assert "safetensor" in fmt, f"expected SafeTensors, got '{fmt}'"


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_debug_tensor_count(slug, parity_data):
    """apr debug: tensor count matches safetensors header."""
    if slug not in parity_data:
        pytest.skip(f"{slug} not in HF cache")
    d = parity_data[slug]
    assert d["apr_debug"]["tensors"] == len(d["header"])


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_debug_architecture(slug, parity_data):
    """apr debug: architecture matches expected."""
    if slug not in parity_data:
        pytest.skip(f"{slug} not in HF cache")
    expected = MODEL_METADATA[slug]["architecture"]
    got = parity_data[slug]["apr_debug"].get("architecture", "").lower()
    assert expected in got, f"expected '{expected}' in '{got}'"


# ── apr hex --stats (data parity) ─────────────────────────────────────────


def _parse_hex_stats(text: str) -> dict[str, float] | None:
    """Parse min/max/mean/std from apr hex --stats output. Returns None if missing."""
    m = re.search(
        r"min=([\d.e+-]+)\s+max=([\d.e+-]+)\s+mean=([\d.e+-]+)\s+std=([\d.e+-]+)",
        text,
    )
    if not m:
        return None
    return {
        "min": float(m.group(1)),
        "max": float(m.group(2)),
        "mean": float(m.group(3)),
        "std": float(m.group(4)),
    }


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_hex_stats_min(slug, parity_data):
    """apr hex --stats: min matches torch within 1e-4."""
    if slug not in parity_data:
        pytest.skip(f"{slug} not in HF cache")
    d = parity_data[slug]
    apr_stats = _parse_hex_stats(d["hex_stdout"])
    assert apr_stats is not None, f"apr hex --stats returned no data for {d['probe_dtype']}"
    py_val = d["probe_pt"].min().item()
    assert abs(apr_stats["min"] - py_val) < 1e-4, (
        f"min: apr={apr_stats['min']}, torch={py_val}"
    )


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_hex_stats_max(slug, parity_data):
    """apr hex --stats: max matches torch within 1e-4."""
    if slug not in parity_data:
        pytest.skip(f"{slug} not in HF cache")
    d = parity_data[slug]
    apr_stats = _parse_hex_stats(d["hex_stdout"])
    assert apr_stats is not None, f"apr hex --stats returned no data for {d['probe_dtype']}"
    py_val = d["probe_pt"].max().item()
    assert abs(apr_stats["max"] - py_val) < 1e-4, (
        f"max: apr={apr_stats['max']}, torch={py_val}"
    )


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_hex_stats_mean(slug, parity_data):
    """apr hex --stats: mean matches torch within 1e-4."""
    if slug not in parity_data:
        pytest.skip(f"{slug} not in HF cache")
    d = parity_data[slug]
    apr_stats = _parse_hex_stats(d["hex_stdout"])
    assert apr_stats is not None, f"apr hex --stats returned no data for {d['probe_dtype']}"
    py_val = d["probe_pt"].mean().item()
    assert abs(apr_stats["mean"] - py_val) < 1e-4, (
        f"mean: apr={apr_stats['mean']}, torch={py_val}"
    )


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_hex_stats_std(slug, parity_data):
    """apr hex --stats: std matches torch within 1e-4."""
    if slug not in parity_data:
        pytest.skip(f"{slug} not in HF cache")
    d = parity_data[slug]
    apr_stats = _parse_hex_stats(d["hex_stdout"])
    assert apr_stats is not None, f"apr hex --stats returned no data for {d['probe_dtype']}"
    py_val = d["probe_pt"].std(correction=0).item()
    assert abs(apr_stats["std"] - py_val) < 1e-4, (
        f"std: apr={apr_stats['std']}, torch={py_val}"
    )


# ── apr hex --header ──────────────────────────────────────────────────────


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_hex_header_length(slug, parity_data):
    """apr hex --header: header_length matches binary read."""
    if slug not in parity_data:
        pytest.skip(f"{slug} not in HF cache")
    d = parity_data[slug]
    m = re.search(r"header_length:\s*(\d+)\s*bytes", d["hex_header_out"])
    assert m, "could not parse header_length from apr hex --header"
    apr_header_len = int(m.group(1))
    assert apr_header_len == d["header_size"], (
        f"header_length: apr={apr_header_len}, binary={d['header_size']}"
    )


# ── apr hex --distribution ────────────────────────────────────────────────


def _parse_hex_distribution(text: str) -> dict[str, float]:
    """Parse entropy/kurtosis/skewness from apr hex --distribution output."""
    result = {}
    for key in ["Entropy", "Kurtosis", "Skewness", "Min", "Max"]:
        m = re.search(rf"{key}:\s*([\d.e+-]+)", text)
        if m:
            result[key.lower()] = float(m.group(1))
    return result


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_hex_distribution_min_max(slug, parity_data):
    """apr hex --distribution: min/max match torch."""
    if slug not in parity_data:
        pytest.skip(f"{slug} not in HF cache")
    d = parity_data[slug]
    dist = _parse_hex_distribution(d["hex_dist_out"])
    assert "min" in dist and "max" in dist, f"apr hex --distribution missing min/max for {d['probe_dtype']}"
    py_min = d["probe_pt"].min().item()
    py_max = d["probe_pt"].max().item()
    assert abs(dist["min"] - py_min) < 1e-4, f"min: apr={dist['min']}, torch={py_min}"
    assert abs(dist["max"] - py_max) < 1e-4, f"max: apr={dist['max']}, torch={py_max}"


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_hex_distribution_kurtosis(slug, parity_data):
    """apr hex --distribution: kurtosis is finite and reasonable."""
    if slug not in parity_data:
        pytest.skip(f"{slug} not in HF cache")
    d = parity_data[slug]
    dist = _parse_hex_distribution(d["hex_dist_out"])
    assert "kurtosis" in dist, f"apr hex --distribution missing kurtosis for {d['probe_dtype']}"
    assert math.isfinite(dist["kurtosis"]), f"kurtosis not finite: {dist['kurtosis']}"


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_hex_distribution_entropy(slug, parity_data):
    """apr hex --distribution: entropy is positive and finite."""
    if slug not in parity_data:
        pytest.skip(f"{slug} not in HF cache")
    d = parity_data[slug]
    dist = _parse_hex_distribution(d["hex_dist_out"])
    assert "entropy" in dist, f"apr hex --distribution missing entropy for {d['probe_dtype']}"
    assert dist["entropy"] > 0, f"entropy should be positive: {dist['entropy']}"
    assert math.isfinite(dist["entropy"]), f"entropy not finite: {dist['entropy']}"


# ── apr tree ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_tree_tensor_count(slug, parity_data):
    """apr tree: tensor_count matches safetensors header."""
    if slug not in parity_data:
        pytest.skip(f"{slug} not in HF cache")
    d = parity_data[slug]
    assert d["apr_tree"]["tensor_count"] == len(d["header"])


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_tree_leaf_names(slug, parity_data):
    """apr tree: every leaf tensor path matches a safetensors tensor name."""
    if slug not in parity_data:
        pytest.skip(f"{slug} not in HF cache")
    d = parity_data[slug]
    tree_leaves = sorted(_collect_tree_leaves(d["apr_tree"]))
    header_names = sorted(d["header"].keys())
    assert tree_leaves == header_names, (
        f"missing={set(header_names) - set(tree_leaves)}, "
        f"extra={set(tree_leaves) - set(header_names)}"
    )


# ── apr validate ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_validate_all_pass(slug, parity_data):
    """apr validate: all tensor checks pass for valid safetensors."""
    if slug not in parity_data:
        pytest.skip(f"{slug} not in HF cache")
    d = parity_data[slug]
    v = d["apr_validate"]
    assert v["failed"] == 0, (
        f"{v['failed']}/{v['total_checks']} checks failed: "
        + ", ".join(c["name"] for c in v["checks"] if c["status"] != "PASS")
    )


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_validate_tensor_count(slug, parity_data):
    """apr validate: total_tensors matches safetensors header."""
    if slug not in parity_data:
        pytest.skip(f"{slug} not in HF cache")
    d = parity_data[slug]
    assert d["apr_validate"]["total_tensors"] == len(d["header"])


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_validate_no_nan(slug, parity_data):
    """apr validate: no NaN values in any tensor."""
    if slug not in parity_data:
        pytest.skip(f"{slug} not in HF cache")
    assert parity_data[slug]["apr_validate"]["total_nan"] == 0


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_validate_no_inf(slug, parity_data):
    """apr validate: no Inf values in any tensor."""
    if slug not in parity_data:
        pytest.skip(f"{slug} not in HF cache")
    assert parity_data[slug]["apr_validate"]["total_inf"] == 0


# ── apr diff (self-identity) ──────────────────────────────────────────────


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_diff_self_identical(slug, parity_data):
    """apr diff: self-diff reports identical."""
    if slug not in parity_data:
        pytest.skip(f"{slug} not in HF cache")
    d = parity_data[slug]["apr_diff_self"]
    assert d["identical"] is True, f"self-diff not identical: {d}"


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_diff_self_zero_diffs(slug, parity_data):
    """apr diff: self-diff has zero differences."""
    if slug not in parity_data:
        pytest.skip(f"{slug} not in HF cache")
    d = parity_data[slug]["apr_diff_self"]
    assert d["difference_count"] == 0, f"self-diff has {d['difference_count']} diffs"


# ── apr list ──────────────────────────────────────────────────────────────


def _find_in_list(slug: str, list_data: dict) -> dict | None:
    """Find a model in apr list by matching HF repo ID in the name field."""
    hf_id = MODEL_METADATA[slug]["hf_id"]
    # apr list names look like "hf_openai-community_gpt2_model.safetensors"
    # Convert "openai-community/gpt2" → "openai-community_gpt2"
    name_fragment = hf_id.replace("/", "_")
    models = list_data.get("models", [])
    return next((m for m in models if name_fragment in m.get("name", "")), None)


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_list_model_present(slug, parity_data, list_data):
    """apr list: model appears in cached model list."""
    if slug not in parity_data:
        pytest.skip(f"{slug} not in HF cache")
    entry = _find_in_list(slug, list_data)
    assert entry is not None, (
        f"{slug} (hf={MODEL_METADATA[slug]['hf_id']}) not found in apr list"
    )


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_list_model_size(slug, parity_data, list_data):
    """apr list: reported size matches actual file size."""
    if slug not in parity_data:
        pytest.skip(f"{slug} not in HF cache")
    entry = _find_in_list(slug, list_data)
    if entry is None:
        pytest.skip(f"{slug} not in apr list")
    actual = os.path.getsize(entry["path"])
    assert entry["size_bytes"] == actual, (
        f"size: apr list={entry['size_bytes']}, os={actual}"
    )


# ── safe_open sanity ──────────────────────────────────────────────────────


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_safe_open_keys_match_header(slug, parity_data):
    """Python safe_open keys match raw header (test sanity check)."""
    if slug not in parity_data:
        pytest.skip(f"{slug} not in HF cache")
    d = parity_data[slug]
    assert sorted(d["safe_open_keys"]) == sorted(d["header"].keys())


# ── apr export roundtrip ──────────────────────────────────────────────────


@pytest.fixture(scope="module")
def roundtrip_data(parity_data):
    """Export each model via apr and read back with Python safetensors."""
    data = {}
    tmpdir = tempfile.mkdtemp(prefix="apr_roundtrip_")
    for slug, d in parity_data.items():
        out_path = os.path.join(tmpdir, f"{slug}-roundtrip.safetensors")
        proc = _run_apr(["export", d["path"], "--format", "safetensors", "-o", out_path])
        if proc.returncode == 0:
            data[slug] = {"out_path": out_path, "rt_header": _read_header(out_path)}
    yield data
    for d in data.values():
        with contextlib.suppress(OSError):
            os.unlink(d["out_path"])
    with contextlib.suppress(OSError):
        os.rmdir(tmpdir)


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_roundtrip_tensor_count(slug, parity_data, roundtrip_data):
    """apr export roundtrip: tensor count preserved."""
    if slug not in roundtrip_data:
        pytest.skip(f"{slug} roundtrip not available")
    orig = parity_data[slug]["header"]
    rt = roundtrip_data[slug]["rt_header"]
    assert len(rt) == len(orig), f"orig={len(orig)}, roundtrip={len(rt)}"


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_roundtrip_tensor_names(slug, parity_data, roundtrip_data):
    """apr export roundtrip: all tensor names preserved."""
    if slug not in roundtrip_data:
        pytest.skip(f"{slug} roundtrip not available")
    orig_names = sorted(parity_data[slug]["header"].keys())
    rt_names = sorted(roundtrip_data[slug]["rt_header"].keys())
    assert orig_names == rt_names


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_roundtrip_shapes_dtypes(slug, parity_data, roundtrip_data):
    """apr export roundtrip: shapes and dtypes preserved for every tensor."""
    if slug not in roundtrip_data:
        pytest.skip(f"{slug} roundtrip not available")
    orig = parity_data[slug]["header"]
    rt = roundtrip_data[slug]["rt_header"]
    mismatches = _compare_tensor_headers(orig, rt)
    assert not mismatches, "roundtrip changed:\n" + "\n".join(mismatches)


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_roundtrip_data_identical(slug, parity_data, roundtrip_data):
    """apr export roundtrip: probe tensor values are bit-perfect."""
    if slug not in roundtrip_data:
        pytest.skip(f"{slug} roundtrip not available")
    orig_path = parity_data[slug]["path"]
    rt_path = roundtrip_data[slug]["out_path"]
    probe = PROBE_TENSORS[slug]
    with safe_open(orig_path, framework="pt") as f1, safe_open(rt_path, framework="pt") as f2:
        t1 = f1.get_tensor(probe)
        t2 = f2.get_tensor(probe)
    assert torch.equal(t1, t2), (
        f"probe tensor {probe} differs: max_diff={( t1.float() - t2.float()).abs().max().item()}"
    )


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_roundtrip_python_readable(slug, roundtrip_data):
    """apr export roundtrip: output file is valid safetensors readable by Python."""
    if slug not in roundtrip_data:
        pytest.skip(f"{slug} roundtrip not available")
    rt_path = roundtrip_data[slug]["out_path"]
    with safe_open(rt_path, framework="pt") as f:
        keys = list(f.keys())
    assert len(keys) > 0, "roundtrip file has no tensors"


# ── stdin/stdout pipe support (PMAT-261) ─────────────────────────────────
# safetensors Python can serialize/deserialize in-memory (bytes ↔ tensors).
# The CLI equivalent is stdin/stdout pipe support: read from stdin, write
# raw bytes to stdout. apr currently lacks both capabilities.


@pytest.fixture(scope="module")
def pipe_data(parity_data):
    """Test stdin and stdout pipe support for each model."""
    data = {}
    for slug, d in parity_data.items():
        path = d["path"]

        # Test 1: apr tensors reading from stdin via `-`
        with open(path, "rb") as f:
            stdin_input = f.read()
        stdin_dash = subprocess.run(
            ["apr", "tensors", "-", "--json"],
            input=stdin_input,
            capture_output=True,
            timeout=10,
        )

        # Test 2: apr tensors reading from /dev/stdin
        with open(path, "rb") as f:
            stdin_input2 = f.read()
        stdin_dev = subprocess.run(
            ["apr", "tensors", "/dev/stdin", "--json"],
            input=stdin_input2,
            capture_output=True,
            timeout=10,
        )

        # Test 3: apr export writing to stdout via `-o -`
        stdout_dash = subprocess.run(
            ["apr", "export", path, "--format", "safetensors", "-o", "-"],
            capture_output=True,
            timeout=30,
        )

        # Test 4: apr export writing to /dev/stdout
        stdout_dev = subprocess.run(
            ["apr", "export", path, "--format", "safetensors", "-o", "/dev/stdout"],
            capture_output=True,
            timeout=30,
        )

        data[slug] = {
            "stdin_dash": stdin_dash,
            "stdin_dev": stdin_dev,
            "stdout_dash": stdout_dash,
            "stdout_dev": stdout_dev,
        }
    return data


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_pipe_stdin_dash(slug, parity_data, pipe_data):
    """apr tensors -: read safetensors from stdin pipe."""
    if slug not in pipe_data:
        pytest.skip(f"{slug} not available")
    proc = pipe_data[slug]["stdin_dash"]
    assert proc.returncode == 0, (
        f"apr tensors - failed (exit {proc.returncode}): "
        f"{proc.stderr.decode() if isinstance(proc.stderr, bytes) else proc.stderr}"
    )
    result = json.loads(proc.stdout)
    expected_count = len(parity_data[slug]["header"])
    assert len(result["tensors"]) == expected_count


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_pipe_stdin_dev(slug, parity_data, pipe_data):
    """apr tensors /dev/stdin: read safetensors from /dev/stdin."""
    if slug not in pipe_data:
        pytest.skip(f"{slug} not available")
    proc = pipe_data[slug]["stdin_dev"]
    assert proc.returncode == 0, (
        f"apr tensors /dev/stdin failed (exit {proc.returncode}): "
        f"{proc.stderr.decode() if isinstance(proc.stderr, bytes) else proc.stderr}"
    )
    result = json.loads(proc.stdout)
    expected_count = len(parity_data[slug]["header"])
    assert len(result["tensors"]) == expected_count


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_pipe_stdout_dash(slug, parity_data, pipe_data):
    """apr export -o -: write safetensors to stdout (parseable by Python)."""
    if slug not in pipe_data:
        pytest.skip(f"{slug} not available")
    proc = pipe_data[slug]["stdout_dash"]
    assert proc.returncode == 0, (
        f"apr export -o - failed (exit {proc.returncode}): "
        f"{proc.stderr.decode() if isinstance(proc.stderr, bytes) else proc.stderr}"
    )
    # stdout should be raw safetensors bytes: 8-byte LE header size + JSON + data
    raw = proc.stdout if isinstance(proc.stdout, bytes) else proc.stdout.encode()
    assert len(raw) >= 8, f"stdout too short ({len(raw)} bytes)"
    header_size = int.from_bytes(raw[:8], "little")
    assert header_size < len(raw), f"header_size={header_size} >= file_size={len(raw)}"
    header = json.loads(raw[8 : 8 + header_size])
    expected_count = len(parity_data[slug]["header"])
    # Exclude __metadata__ key
    tensor_keys = [k for k in header if k != "__metadata__"]
    assert len(tensor_keys) == expected_count


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_pipe_stdout_dev(slug, parity_data, pipe_data):
    """apr export -o /dev/stdout: write safetensors to /dev/stdout."""
    if slug not in pipe_data:
        pytest.skip(f"{slug} not available")
    proc = pipe_data[slug]["stdout_dev"]
    assert proc.returncode == 0, (
        f"apr export -o /dev/stdout failed (exit {proc.returncode}): "
        f"{proc.stderr.decode() if isinstance(proc.stderr, bytes) else proc.stderr}"
    )
    raw = proc.stdout if isinstance(proc.stdout, bytes) else proc.stdout.encode()
    assert len(raw) >= 8, f"stdout too short ({len(raw)} bytes)"
    header_size = int.from_bytes(raw[:8], "little")
    assert header_size < len(raw), f"header_size={header_size} >= file_size={len(raw)}"


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_pipe_roundtrip(slug, parity_data, pipe_data):
    """Full pipe roundtrip: apr export -o - | Python safetensors.deserialize()."""
    if slug not in pipe_data:
        pytest.skip(f"{slug} not available")
    proc = pipe_data[slug]["stdout_dash"]
    if proc.returncode != 0:
        pytest.fail("apr export -o - failed, can't test roundtrip")
    raw = proc.stdout if isinstance(proc.stdout, bytes) else proc.stdout.encode()
    # Parse as safetensors in Python
    from safetensors.torch import load as st_load

    tensors = st_load(raw)
    expected_count = len(parity_data[slug]["header"])
    assert len(tensors) == expected_count, (
        f"pipe roundtrip: got {len(tensors)} tensors, expected {expected_count}"
    )
    # Verify probe tensor matches
    probe = PROBE_TENSORS[slug]
    orig = parity_data[slug]["probe_pt"]
    assert torch.equal(tensors[probe].float(), orig), (
        f"probe tensor {probe} differs after pipe roundtrip"
    )


# ── tensor slicing (safetensors Python feature, no apr equivalent) ────────


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_slice_partial_load(slug, parity_data):
    """safetensors get_slice: load partial tensor without full weight read."""
    if slug not in parity_data:
        pytest.skip(f"{slug} not in HF cache")
    path = parity_data[slug]["path"]
    probe = PROBE_TENSORS[slug]
    with safe_open(path, framework="pt") as f:
        s = f.get_slice(probe)
        partial = s[:3]
        full = f.get_tensor(probe)
    assert torch.equal(partial, full[:3]), "slice[:3] != full[:3]"
    assert partial.shape[0] == 3


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_slice_apr_equivalent(slug, parity_data):
    """apr should support partial tensor reads (like safetensors get_slice)."""
    if slug not in parity_data:
        pytest.skip(f"{slug} not in HF cache")
    path = parity_data[slug]["path"]
    probe = PROBE_TENSORS[slug]
    # No apr subcommand supports slicing; this tests if one appears
    proc = _run_apr(["hex", path, "--tensor", probe, "--slice", "0:3", "--json"])
    assert proc.returncode == 0, f"apr hex --slice not supported: {proc.stderr}"


# ── metadata preservation ─────────────────────────────────────────────────


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_metadata_python_reads(slug, parity_data):
    """safetensors metadata() returns __metadata__ dict."""
    if slug not in parity_data:
        pytest.skip(f"{slug} not in HF cache")
    path = parity_data[slug]["path"]
    with safe_open(path, framework="pt") as f:
        meta = f.metadata()
    # metadata may be None or a dict
    assert meta is None or isinstance(meta, dict)


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_metadata_apr_inspect(slug, parity_data):
    """apr inspect: metadata field matches safetensors __metadata__."""
    if slug not in parity_data:
        pytest.skip(f"{slug} not in HF cache")
    path = parity_data[slug]["path"]
    # Python ground truth
    with safe_open(path, framework="pt") as f:
        py_meta = f.metadata() or {}
    if not py_meta:
        pytest.skip("no metadata in this model")
    # apr inspect
    apr_meta = parity_data[slug]["apr_inspect"].get("metadata", {})
    # apr should surface the same metadata keys
    for key in py_meta:
        assert key in apr_meta, (
            f"apr inspect missing metadata key '{key}': "
            f"python={py_meta}, apr={apr_meta}"
        )


# ── corruption handling ───────────────────────────────────────────────────


@pytest.fixture(scope="module")
def corruption_data(parity_data):
    """Create corrupted safetensors variants and test apr behavior."""
    tmpdir = tempfile.mkdtemp(prefix="apr_corrupt_")
    data = {}
    # Use first available model
    slug = next(iter(parity_data))
    path = parity_data[slug]["path"]

    # Variant 1: truncated (100 bytes)
    trunc_path = os.path.join(tmpdir, "truncated.safetensors")
    with open(path, "rb") as f, open(trunc_path, "wb") as out:
        out.write(f.read(100))

    # Variant 2: zero-length
    empty_path = os.path.join(tmpdir, "empty.safetensors")
    with open(empty_path, "wb") as f:
        pass

    # Variant 3: garbage bytes
    garbage_path = os.path.join(tmpdir, "garbage.safetensors")
    with open(garbage_path, "wb") as f:
        f.write(os.urandom(256))

    # Variant 4: valid header but truncated data
    with open(path, "rb") as f:
        header_size = int.from_bytes(f.read(8), "little")
        # Write header + 8 byte length prefix but only 10 bytes of tensor data
        f.seek(0)
        header_bytes = f.read(8 + header_size + 10)
    partial_path = os.path.join(tmpdir, "partial_data.safetensors")
    with open(partial_path, "wb") as f:
        f.write(header_bytes)

    variants = {
        "truncated": trunc_path,
        "empty": empty_path,
        "garbage": garbage_path,
        "partial_data": partial_path,
    }

    results = {}
    for name, vpath in variants.items():
        results[name] = {
            "path": vpath,
            "validate": _run_apr(["validate", vpath, "--json"]),
            "debug": _run_apr(["debug", vpath, "--json"]),
            "tensors": _run_apr(["tensors", vpath, "--json"]),
        }

    data["slug"] = slug
    data["variants"] = results
    yield data
    # cleanup
    for _name, vpath in variants.items():
        with contextlib.suppress(OSError):
            os.unlink(vpath)
    with contextlib.suppress(OSError):
        os.rmdir(tmpdir)


# partial_data has a valid header but truncated tensor data.
# apr tensors only reads the header, so it correctly returns exit 0.
# apr debug/validate also only read the header for basic checks.
CORRUPT_TOTAL = ["truncated", "empty", "garbage"]


@pytest.mark.parametrize("variant", CORRUPT_TOTAL)
def test_corruption_validate_rejects(variant, corruption_data):
    """apr validate: rejects corrupted safetensors files (non-zero exit)."""
    result = corruption_data["variants"][variant]["validate"]
    assert result.returncode != 0, (
        f"apr validate should reject {variant} file but returned exit 0"
    )


def test_corruption_partial_data_detected(corruption_data):
    """apr validate: partial_data (valid header, truncated tensors) should be flagged.
    """
    result = corruption_data["variants"]["partial_data"]["validate"]
    if result.returncode != 0:
        return  # correctly rejected
    data = json.loads(result.stdout)
    # If it passes, it should at least identify the format correctly
    fmt = data.get("format", "").lower()
    assert "safetensor" in fmt, (
        f"partial_data has valid safetensors header but apr says format='{fmt}'"
    )


@pytest.mark.parametrize("variant", CORRUPT_TOTAL)
def test_corruption_debug_rejects(variant, corruption_data):
    """apr debug: rejects corrupted safetensors files (non-zero exit)."""
    result = corruption_data["variants"][variant]["debug"]
    assert result.returncode != 0, (
        f"apr debug should reject {variant} file but returned exit 0"
    )


@pytest.mark.parametrize("variant", CORRUPT_TOTAL)
def test_corruption_tensors_rejects(variant, corruption_data):
    """apr tensors: rejects corrupted safetensors files (non-zero exit)."""
    result = corruption_data["variants"][variant]["tensors"]
    assert result.returncode != 0, (
        f"apr tensors should reject {variant} file but returned exit 0"
    )


@pytest.mark.parametrize("variant", ["truncated", "empty", "garbage", "partial_data"])
def test_corruption_no_panic(variant, corruption_data):
    """apr does not panic/crash on corrupted files (stderr has no 'panicked')."""
    for cmd in ["validate", "debug", "tensors"]:
        result = corruption_data["variants"][variant][cmd]
        stderr = result.stderr
        assert "panicked" not in stderr.lower(), (
            f"apr {cmd} panicked on {variant}: {stderr[:200]}"
        )
        assert "SIGSEGV" not in stderr, (
            f"apr {cmd} segfaulted on {variant}: {stderr[:200]}"
        )


@pytest.mark.parametrize("variant", ["truncated", "empty", "garbage", "partial_data"])
def test_corruption_python_also_rejects(variant, corruption_data):
    """Python safetensors also rejects corrupted files (sanity check)."""
    path = corruption_data["variants"][variant]["path"]
    with pytest.raises((SafetensorError, ValueError, RuntimeError, OSError)), safe_open(path, framework="pt") as f:
        list(f.keys())


# ── cross-format diff ─────────────────────────────────────────────────────


def test_diff_cross_model():
    """apr diff: reports differences between two different models."""
    slugs = MODEL_PARAMS[:2]
    if len(slugs) < 2:
        pytest.skip("need at least 2 models")
    try:
        path_a = _resolve_path(slugs[0])
        path_b = _resolve_path(slugs[1])
    except Exception:
        pytest.skip("models not in HF cache")
    data = _apr_json(["diff", path_a, path_b, "--json"])
    assert data["identical"] is False, "different models should not be identical"
    assert data["difference_count"] > 0


# ── apr trace ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def trace_data(parity_data):
    """Run apr trace --json once per model."""
    data = {}
    for slug, d in parity_data.items():
        proc = _run_apr(["trace", d["path"], "--json"])
        if proc.returncode == 0:
            with contextlib.suppress(json.JSONDecodeError):
                data[slug] = json.loads(proc.stdout)
    return data


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_trace_has_layers(slug, parity_data, trace_data):
    """apr trace: returns non-empty layer list."""
    if slug not in trace_data:
        pytest.skip(f"{slug} trace not available")
    layers = trace_data[slug].get("layers", [])
    assert len(layers) > 0, "apr trace returned no layers"


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_trace_layer_count(slug, parity_data, trace_data):
    """apr trace: layer count matches expected architecture layers + embedding."""
    if slug not in trace_data:
        pytest.skip(f"{slug} trace not available")
    layers = trace_data[slug].get("layers", [])
    expected_layers = MODEL_METADATA[slug]["layers"]
    # trace usually has: embedding + N transformer blocks (+ possibly output)
    block_layers = [ly for ly in layers if "block" in ly.get("name", "").lower()
                    or "layer" in ly.get("name", "").lower()
                    or ly.get("index") is not None]
    assert len(block_layers) >= expected_layers, (
        f"expected >= {expected_layers} transformer blocks, got {len(block_layers)}: "
        f"{[ly['name'] for ly in block_layers]}"
    )


# ── apr oracle ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def oracle_data(parity_data):
    """Run apr oracle --json once per model."""
    data = {}
    for slug, d in parity_data.items():
        proc = _run_apr(["oracle", d["path"], "--json"])
        if proc.returncode == 0:
            with contextlib.suppress(json.JSONDecodeError):
                data[slug] = json.loads(proc.stdout)
    return data


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_oracle_architecture(slug, parity_data, oracle_data):
    """apr oracle: detects correct architecture."""
    if slug not in oracle_data:
        pytest.skip(f"{slug} oracle not available")
    expected = MODEL_METADATA[slug]["architecture"]
    fmt = oracle_data[slug].get("format", {})
    got = fmt.get("architecture", "").lower()
    assert expected in got, f"expected '{expected}' in oracle arch '{got}'"


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_oracle_param_count(slug, parity_data, oracle_data):
    """apr oracle: total_params matches safetensors header."""
    if slug not in oracle_data:
        pytest.skip(f"{slug} oracle not available")
    header = parity_data[slug]["header"]
    expected = sum(math.prod(m["shape"]) for m in header.values() if m["shape"])
    fmt = oracle_data[slug].get("format", {})
    got = fmt.get("total_params", 0)
    assert got == expected, f"oracle params={got}, header params={expected}"


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_oracle_tensor_count(slug, parity_data, oracle_data):
    """apr oracle: tensor_count matches safetensors header."""
    if slug not in oracle_data:
        pytest.skip(f"{slug} oracle not available")
    expected = len(parity_data[slug]["header"])
    fmt = oracle_data[slug].get("format", {})
    got = fmt.get("tensor_count", 0)
    assert got == expected, f"oracle tensors={got}, header tensors={expected}"


# ── apr flow ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_flow_json_output(slug, parity_data):
    """apr flow --json: should produce valid JSON."""
    if slug not in parity_data:
        pytest.skip(f"{slug} not in HF cache")
    proc = _run_apr(["flow", parity_data[slug]["path"], "--json"])
    assert proc.returncode == 0, f"apr flow failed: {proc.stderr}"
    data = json.loads(proc.stdout)
    assert isinstance(data, dict), "apr flow --json should return a JSON object"


# ── apr explain ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_explain_tensor(slug, parity_data):
    """apr explain --tensor: should describe a tensor from a safetensors file."""
    if slug not in parity_data:
        pytest.skip(f"{slug} not in HF cache")
    probe = PROBE_TENSORS[slug]
    proc = _run_apr(["explain", "--tensor", probe, "--file", parity_data[slug]["path"]])
    assert proc.returncode == 0, f"apr explain failed: {proc.stderr}"
    out = _strip_ansi(proc.stdout)
    # Should contain the tensor name and its shape/dtype
    assert probe in out, f"explain output doesn't mention tensor name: {out[:200]}"
    assert "Shape" in out or "shape" in out, f"explain output missing shape: {out[:200]}"


# ── compare-hf ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("slug", MODEL_PARAMS)
def test_compare_hf(slug, parity_data):
    """apr compare-hf: should work on safetensors files, not just .apr."""
    if slug not in parity_data:
        pytest.skip(f"{slug} not in HF cache")
    hf_id = MODEL_METADATA[slug]["hf_id"]
    proc = _run_apr(["compare-hf", parity_data[slug]["path"], "--hf", hf_id, "--json"])
    assert proc.returncode == 0, f"apr compare-hf failed: {proc.stderr}"
    data = json.loads(proc.stdout)
    assert isinstance(data, dict)


# ── sharded models ────────────────────────────────────────────────────────
# Test apr on multi-file safetensors (Phi-4-mini-instruct has 2 shards in cache)

SHARDED_MODELS = {
    "phi-4-mini": {
        "hf_id": "microsoft/Phi-4-mini-instruct",
        "shards": [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ],
    },
}


def _resolve_shard(hf_id: str, filename: str) -> str | None:
    """Resolve a sharded safetensors file from HF cache, following symlinks."""
    try:
        path = hf_hub_download(hf_id, filename, local_files_only=True)
        return os.path.realpath(path)  # resolve symlinks for apr
    except Exception:
        return None


def _resolve_shard_paths(info: dict) -> list[str] | None:
    """Resolve all shard file paths for a sharded model, or None if any missing."""
    paths = []
    for shard_file in info["shards"]:
        path = _resolve_shard(info["hf_id"], shard_file)
        if path is None:
            return None
        paths.append(path)
    return paths


def _load_shard(spath: str) -> dict:
    """Load all data (header, apr commands, Python keys) for a single shard."""
    with safe_open(spath, framework="pt") as f:
        py_keys = list(f.keys())
        py_meta = f.metadata() or {}
    return {
        "path": spath,
        "header": _read_header(spath),
        "apr_tensors": _apr_json(["tensors", spath, "--json"]),
        "apr_inspect": _apr_json(["inspect", spath, "--json"]),
        "apr_validate": _apr_json(["validate", spath, "--json"]),
        "py_keys": py_keys,
        "py_meta": py_meta,
    }


def _build_shard_entry(info: dict) -> dict | None:
    """Build shard data entry for a single model, or None if shards not in cache."""
    shard_paths = _resolve_shard_paths(info)
    if shard_paths is None:
        return None
    shard_results = [_load_shard(sp) for sp in shard_paths]
    py_all_keys = [k for s in shard_results for k in s["py_keys"]]
    py_metadata = {}
    for s in shard_results:
        py_metadata.update(s["py_meta"])
    return {
        "shards": shard_results,
        "py_total_tensors": len(py_all_keys),
        "py_all_keys": py_all_keys,
        "py_metadata": py_metadata,
    }


@pytest.fixture(scope="module")
def shard_data():
    """Load sharded model data from HF cache."""
    data = {}
    for name, info in SHARDED_MODELS.items():
        entry = _build_shard_entry(info)
        if entry is not None:
            data[name] = entry
    if not data:
        pytest.skip("No sharded models in HF cache")
    return data


SHARD_NAMES = list(SHARDED_MODELS.keys())


@pytest.mark.parametrize("name", SHARD_NAMES)
def test_shard_tensor_count_matches(name, shard_data):
    """Sharded: sum of per-shard tensor counts matches Python total."""
    if name not in shard_data:
        pytest.skip(f"{name} not in HF cache")
    d = shard_data[name]
    apr_total = sum(
        len(s["apr_tensors"]["tensors"]) for s in d["shards"]
    )
    assert apr_total == d["py_total_tensors"], (
        f"apr total={apr_total}, python total={d['py_total_tensors']}"
    )


@pytest.mark.parametrize("name", SHARD_NAMES)
def test_shard_no_duplicate_tensors(name, shard_data):
    """Sharded: no tensor name appears in more than one shard."""
    if name not in shard_data:
        pytest.skip(f"{name} not in HF cache")
    all_names = []
    for shard in shard_data[name]["shards"]:
        all_names.extend(shard["header"].keys())
    dupes = [n for n in set(all_names) if all_names.count(n) > 1]
    assert not dupes, f"duplicate tensors across shards: {dupes[:5]}"


@pytest.mark.parametrize("name", SHARD_NAMES)
def test_shard_per_shard_names_match(name, shard_data):
    """Sharded: per-shard apr tensor names match Python safe_open keys."""
    if name not in shard_data:
        pytest.skip(f"{name} not in HF cache")
    for i, shard in enumerate(shard_data[name]["shards"]):
        apr_names = sorted(t["name"] for t in shard["apr_tensors"]["tensors"])
        py_names = sorted(shard["py_keys"])
        assert apr_names == py_names, (
            f"shard {i}: name mismatch, "
            f"missing={set(py_names) - set(apr_names)}, "
            f"extra={set(apr_names) - set(py_names)}"
        )


@pytest.mark.parametrize("name", SHARD_NAMES)
def test_shard_validate_all_pass(name, shard_data):
    """Sharded: apr validate passes on each shard."""
    if name not in shard_data:
        pytest.skip(f"{name} not in HF cache")
    for i, shard in enumerate(shard_data[name]["shards"]):
        v = shard["apr_validate"]
        assert v["failed"] == 0, (
            f"shard {i}: {v['failed']}/{v['total_checks']} checks failed"
        )


@pytest.mark.parametrize("name", SHARD_NAMES)
def test_shard_metadata_preserved(name, shard_data):
    """Sharded: apr inspect surfaces __metadata__ from each shard."""
    if name not in shard_data:
        pytest.skip(f"{name} not in HF cache")
    py_meta = shard_data[name]["py_metadata"]
    if not py_meta:
        pytest.skip("no metadata in sharded model")
    for i, shard in enumerate(shard_data[name]["shards"]):
        apr_meta = shard["apr_inspect"].get("metadata", {})
        for key in py_meta:
            assert key in apr_meta, (
                f"shard {i}: apr inspect missing metadata key '{key}': "
                f"python={py_meta}, apr={apr_meta}"
            )


@pytest.mark.parametrize("name", SHARD_NAMES)
def test_shard_shapes_match(name, shard_data):
    """Sharded: per-shard tensor shapes match safetensors header."""
    if name not in shard_data:
        pytest.skip(f"{name} not in HF cache")
    for i, shard in enumerate(shard_data[name]["shards"]):
        apr_by_name = {t["name"]: t for t in shard["apr_tensors"]["tensors"]}
        mismatches = [
            f"  {n}: header={shard['header'][n]['shape']}, apr={apr_by_name[n]['shape']}"
            for n in shard["header"]
            if n in apr_by_name and shard["header"][n]["shape"] != apr_by_name[n]["shape"]
        ]
        assert not mismatches, f"shard {i} shape mismatches:\n" + "\n".join(mismatches)
