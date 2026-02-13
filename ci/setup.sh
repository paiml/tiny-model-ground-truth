#!/usr/bin/env bash
# CI setup script — install apr and Python deps
set -euo pipefail

echo "=== Installing apr-cli ==="
cargo install apr-cli --locked

echo "=== Installing Python deps ==="
pip install uv
uv sync

echo "=== Versions ==="
apr --version || echo "apr: not found"
uv --version || echo "uv: not found"
python3 --version || echo "python3: not found"
