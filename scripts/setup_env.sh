#!/usr/bin/env bash
# Install the RSI framework + the heavy GPU deps for Blackwell (RTX 5090, sm_120).
#
# PyTorch + Triton for sm_120 need a recent CUDA-12.8+ build. We try the cu128 index
# first; adjust the index URL if NVIDIA/PyTorch ship a newer channel.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Installing the rsi package (Claude Agent SDK only)…"
python3 -m pip install -e .

echo "==> Installing PyTorch + Triton for Blackwell sm_120 (CUDA 12.8)…"
# Triton ships inside the PyTorch wheels; install torch from the cu128 channel.
python3 -m pip install --index-url https://download.pytorch.org/whl/cu128 torch || {
  echo "cu128 channel failed; trying the nightly cu128 channel…"
  python3 -m pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu128 torch
}

# Ensure a standalone triton is present (some torch wheels bundle it; this is a no-op then).
python3 -m pip install -q triton || true
python3 -m pip install -q numpy pandas

echo "==> Optional: vLLM (for external baseline context) — skipped by default."
echo "    To install:  python3 -m pip install vllm"

echo "==> Verifying torch sees the GPU…"
python3 - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability(0))
PY

echo "==> Done. Next:  python3 scripts/smoke_test.py"
