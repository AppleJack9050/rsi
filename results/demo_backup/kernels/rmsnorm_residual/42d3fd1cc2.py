
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4,  num_stages=1),
        triton.Config({}, num_warps=8,  num_stages=1),
        triton.Config({}, num_warps=16, num_stages=1),
        triton.Config({}, num_warps=32, num_stages=1),
        triton.Config({}, num_warps=4,  num_stages=2),
        triton.Config({}, num_warps=8,  num_stages=2),
        triton.Config({}, num_warps=16, num_stages=2),
        triton.Config({}, num_warps=32, num_stages=2),
        triton.Config({}, num_warps=4,  num_stages=3),
        triton.Config({}, num_warps=8,  num_stages=3),
        triton.Config({}, num_warps=16, num_stages=3),
        triton.Config({}, num_warps=32, num_stages=3),
    ],
    key=['T', 'H'],
)
@triton.jit
def rmsnorm_residual_kernel(
    x_ptr, residual_ptr, weight_ptr, out_ptr,
    T, H,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < H

    # Weight is read by every row but never changes — keep it in L1 as long
    # as possible so subsequent CTAs on the same SM hit L1 instead of L2/DRAM.
    # x and residual are streaming (one row each, never reused) — evict them
    # first to avoid displacing weight from L1.
    w_vals   = tl.load(weight_ptr   + offs,           mask=mask, other=0.0,
                       eviction_policy='evict_last').to(tl.float32)
    x_vals   = tl.load(x_ptr       + row * H + offs, mask=mask, other=0.0,
                       eviction_policy='evict_first').to(tl.float32)
    res_vals = tl.load(residual_ptr + row * H + offs, mask=mask, other=0.0,
                       eviction_policy='evict_first').to(tl.float32)

    h = x_vals + res_vals

    # fp32 reduction → scale → cast back to fp16
    sum_sq = tl.sum(h * h, axis=0)
    rrms   = tl.rsqrt(sum_sq / H + eps)

    out = (h * rrms * w_vals).to(tl.float16)
    tl.store(out_ptr + row * H + offs, out, mask=mask,
             eviction_policy='evict_first')


def run(x, residual, weight):
    T, H = x.shape
    out = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(H)  # H=2048 → BLOCK_SIZE=2048

    rmsnorm_residual_kernel[(T,)](
        x, residual, weight, out,
        T, H,
        1e-6,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out
