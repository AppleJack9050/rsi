"""Our Triton kernels, wrapped to vLLM's exact layer signatures AND registered as
torch.library custom ops so they are traceable by inductor and capturable in vLLM's
CUDA graphs (needed for a stable full-model throughput A/B).

vLLM calls (bf16 hidden states):
  RMSNorm.forward(x, residual) ->
      residual is None: rms_norm(x)*w
      else:             (rmsnorm(x+residual)*w, x+residual)
  SiluAndMul.forward(x[..., :d] | x[..., d:]) -> silu(gate)*up , d = x.shape[-1]//2

Kernels accumulate in fp32 and store in the output tensor's dtype (fp16 + bf16 safe).
They allocate via torch.empty (graph-pool-safe) and contain no host syncs, so they are
CUDA-graph capturable. No @triton.autotune (fixed configs) so capture never benchmarks.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------------- #
# Triton kernels
# --------------------------------------------------------------------------- #
@triton.jit
def _add_rmsnorm_2out(x_ptr, r_ptr, w_ptr, o_ptr, h_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    base = row * n_cols + offs
    x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + base, mask=mask, other=0.0).to(tl.float32)
    h = x + r
    var = tl.sum(h * h, axis=0) / n_cols
    inv = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(o_ptr + base, ((h * inv) * w).to(o_ptr.dtype.element_ty), mask=mask)
    tl.store(h_ptr + base, h.to(h_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _rmsnorm_1out(x_ptr, w_ptr, o_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    base = row * n_cols + offs
    x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / n_cols
    inv = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(o_ptr + base, ((x * inv) * w).to(o_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _silu_and_mul_flat(x_ptr, o_ptr, N, d, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N
    row = idx // d
    col = idx % d
    g = tl.load(x_ptr + row * 2 * d + col, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(x_ptr + row * 2 * d + d + col, mask=mask, other=0.0)
    o = (g * tl.sigmoid(g)).to(o_ptr.dtype.element_ty) * u
    tl.store(o_ptr + idx, o, mask=mask)


def _nw(H):
    return 8 if H >= 2048 else 4


# --------------------------------------------------------------------------- #
# torch.library custom ops (graph-capturable, inductor-traceable)
# --------------------------------------------------------------------------- #
@torch.library.custom_op("rsi::rmsnorm", mutates_args=())
def rmsnorm_op(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    H = x.shape[-1]
    x2 = x.reshape(-1, H)
    out = torch.empty_like(x2)
    _rmsnorm_1out[(x2.shape[0],)](x2, weight, out, H, eps,
                                  BLOCK=triton.next_power_of_2(H), num_warps=_nw(H))
    return out.reshape_as(x)


@rmsnorm_op.register_fake
def _(x, weight, eps):
    return torch.empty_like(x)


@torch.library.custom_op("rsi::fused_add_rmsnorm", mutates_args=())
def fused_add_rmsnorm_op(x: torch.Tensor, residual: torch.Tensor,
                         weight: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    H = x.shape[-1]
    x2 = x.reshape(-1, H)
    r2 = residual.reshape(-1, H)
    out = torch.empty_like(x2)
    h = torch.empty_like(x2)
    _add_rmsnorm_2out[(x2.shape[0],)](x2, r2, weight, out, h, H, eps,
                                      BLOCK=triton.next_power_of_2(H), num_warps=_nw(H))
    return out.reshape_as(x), h.reshape_as(residual)


@fused_add_rmsnorm_op.register_fake
def _(x, residual, weight, eps):
    return torch.empty_like(x), torch.empty_like(residual)


@torch.library.custom_op("rsi::silu_and_mul", mutates_args=())
def silu_and_mul_op(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x2 = x.reshape(-1, 2 * d).contiguous()
    T = x2.shape[0]
    out = torch.empty((T, d), device=x.device, dtype=x.dtype)
    N = T * d
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK']),)
    _silu_and_mul_flat[grid](x2, out, N, d, BLOCK=2048, num_warps=8)
    return out.reshape(*x.shape[:-1], d)


@silu_and_mul_op.register_fake
def _(x):
    d = x.shape[-1] // 2
    return x.new_empty((*x.shape[:-1], d))


# --------------------------------------------------------------------------- #
# Public API used by the vLLM source hooks
# --------------------------------------------------------------------------- #
def rms_forward(x, weight, eps, residual=None):
    if residual is None:
        return torch.ops.rsi.rmsnorm(x, weight, eps)
    return torch.ops.rsi.fused_add_rmsnorm(x, residual, weight, eps)


def silu_and_mul(x):
    return torch.ops.rsi.silu_and_mul(x)
