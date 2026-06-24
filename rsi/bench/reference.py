"""Pure-PyTorch reference implementations of the target ops.

These serve double duty:
  1. **Correctness oracle** — the kernel must match `reference(*inputs)` within tolerance.
  2. **Speed baseline** — eager PyTorch (multiple kernel launches) is what we time against;
     beating it is the per-kernel speedup the framework reports.

Each reference computes reductions in float32 then casts back to the input dtype, matching
HF / vLLM numerics, so a correct kernel that accumulates in fp32 lands within tolerance.

torch is imported lazily by callers (these functions are only invoked at runtime on the GPU).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1. Fused residual-add + RMSNorm   (memory-bound; classic fusion win)
# --------------------------------------------------------------------------- #
def rmsnorm_residual(x: torch.Tensor, residual: torch.Tensor,
                     weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """out = RMSNorm(x + residual) * weight, over the last dim."""
    h = (x + residual).to(torch.float32)
    var = h.pow(2).mean(dim=-1, keepdim=True)
    out = h * torch.rsqrt(var + eps)
    return (out.to(x.dtype) * weight)


# --------------------------------------------------------------------------- #
# 2. SwiGLU activation   (pointwise fusion; gate/up GEMMs left to cuBLAS)
# --------------------------------------------------------------------------- #
def swiglu_act(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """out = silu(gate) * up."""
    return F.silu(gate) * up


# --------------------------------------------------------------------------- #
# 3. RoPE   (rotary positional embedding applied to Q or K)
# --------------------------------------------------------------------------- #
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [T, H, D]; cos/sin: [T, D] (broadcast over heads). out = x*cos + rotate_half(x)*sin."""
    cos = cos.unsqueeze(1)  # [T, 1, D]
    sin = sin.unsqueeze(1)
    xf = x.to(torch.float32)
    out = xf * cos + _rotate_half(xf) * sin
    return out.to(x.dtype)


# --------------------------------------------------------------------------- #
# 4. Row-wise softmax   (the numerically-stable reduction inside attention)
# --------------------------------------------------------------------------- #
def softmax(x: torch.Tensor) -> torch.Tensor:
    """Row softmax over the last dim, computed in fp32."""
    return torch.softmax(x.to(torch.float32), dim=-1).to(x.dtype)


# --------------------------------------------------------------------------- #
# 5. GQA decode attention (STRETCH)   single query position, KV cache
# --------------------------------------------------------------------------- #
def gqa_decode_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """q: [B, Hq, D]; k/v: [B, Hkv, S, D] (Hq % Hkv == 0). out: [B, Hq, D].

    Single-query (decode) attention with grouped KV heads. softmax over S in fp32.
    """
    B, Hq, D = q.shape
    Hkv, S = k.shape[1], k.shape[2]
    group = Hq // Hkv
    qf = q.to(torch.float32)
    kf = k.to(torch.float32)
    vf = v.to(torch.float32)
    # expand kv heads to match q heads (GQA)
    kf = kf.repeat_interleave(group, dim=1)   # [B, Hq, S, D]
    vf = vf.repeat_interleave(group, dim=1)
    scale = 1.0 / (D ** 0.5)
    scores = torch.einsum("bhd,bhsd->bhs", qf, kf) * scale   # [B, Hq, S]
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhs,bhsd->bhd", probs, vf)           # [B, Hq, D]
    return out.to(q.dtype)
