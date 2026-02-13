#!/usr/bin/env bash
# CI setup script — install apr, ruchy, and optionally llama.cpp
set -euo pipefail

echo "=== Installing apr-cli ==="
cargo install apr-cli --locked

echo "=== Installing ruchy ==="
cargo install ruchy --locked

# Optional: build llama.cpp if cmake is available
if command -v cmake &>/dev/null; then
    echo "=== Building llama.cpp ==="
    if [ ! -d /tmp/llama.cpp ]; then
        git clone --depth 1 https://github.com/ggerganov/llama.cpp /tmp/llama.cpp
    fi
    cmake -B /tmp/llama.cpp/build -S /tmp/llama.cpp -DCMAKE_BUILD_TYPE=Release
    cmake --build /tmp/llama.cpp/build --config Release -j "$(nproc)"
    cp /tmp/llama.cpp/build/bin/llama-cli "$HOME/.cargo/bin/"
    echo "llama-cli installed"
else
    echo "cmake not found — skipping llama.cpp (runtime parity tests will be skipped)"
fi

echo "=== Versions ==="
apr --version || echo "apr: not found"
ruchy --version || echo "ruchy: not found"
llama-cli --version 2>/dev/null || echo "llama-cli: not installed (optional)"
