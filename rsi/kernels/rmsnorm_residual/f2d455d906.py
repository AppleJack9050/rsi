
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # ROWS_PER_CTA=1: maximize row-level parallelism (best for decode T=32)
        triton.Config({'BLOCK_SIZE': 2048, 'ROWS_PER_CTA': 1}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048, 'ROWS_PER_CTA': 1}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048, 'ROWS_PER_CTA': 1}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048, 'ROWS_PER_CTA': 1}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048, 'ROWS_PER_CTA': 1}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048, 'ROWS_PER_CTA': 1}, num_warps=16, num_stages=3),
        # ROWS_PER_CTA=2: share weight load across 2 rows (may help prefill)
        triton.Config({'BLOCK_SIZE': 2048, 'ROWS_PER_CTA': 2}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048, 'ROWS_PER_CTA': 2}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048, 'ROWS_PER_CTA': 2}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048, 'ROWS_PER_CTA': 2}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048, 'ROWS_PER_CTA': 2}, num_warps=16, num_stages=3),
        # ROWS_PER_CTA=4: share weight load across 4 rows (best weight amortization for prefill)
        triton.Config({'BLOCK_SIZE': 2048, 'ROWS_PER_CTA': 4}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048, 'ROWS_PER_CTA': 4}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048, 'ROWS_PER_CTA': 4}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048, 'ROWS_PER_CTA': 4}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048, 'ROWS_PER_CTA': 4}, num_warps=16, num_stages=3),
    ],
    key=['T'],
)
@triton.jit
def rmsnorm_residual_kernel(
    x_ptr, residual_ptr, weight_ptr, out_ptr,
    T, H,
    stride_xT, stride_rT, stride_oT,
    eps,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_CTA: tl.constexpr,
):
    cta_id = tl.program_id(0)
    row_start = cta_id * ROWS_PER_CTA
    offs = tl.arange(0, BLOCK_SIZE)

    # Load weight ONCE per CTA into registers — shared across all ROWS_PER_CTA rows.
    # evict_last hints the L1 cache to keep this alive across iterations.
    w = tl.load(weight_ptr + offs, eviction_policy='evict_last')  # fp16

    for r in tl.static_range(ROWS_PER_CTA):
        row = row_start + r
        # evict_first: these are streaming inputs that won't be reused
        x   = tl.load(x_ptr       + row * stride_xT + offs, eviction_policy='evict_first').to(tl.float32)
        res = tl.load(residual_ptr + row * stride_rT + offs, eviction_policy='evict_first').to(tl.float32)
        h   = x + res

        # RMS normalization in fp32.
        # Use compile-time constant (1.0/BLOCK_SIZE) instead of runtime division by H.
        var = tl.sum(h * h, axis=0) * (1.0 / BLOCK_SIZE)
        rms = tl.rsqrt(var + eps)

        # Cast to fp16 and apply weight (fp16 * fp16)
        out = (h * rms).to(tl.float16) * w
        tl.store(out_ptr + row * stride_oT + offs, out, eviction_policy='evict_first')


def run(x, residual, weight):
    T, H = x.shape
    out  = torch.empty_like(x)

    def grid(meta):
        return (triton.cdiv(T, meta['ROWS_PER_CTA']),)

    rmsnorm_residual_kernel[grid](
        x, residual, weight, out,
        T, H,
        x.stride(0), residual.stride(0), out.stride(0),
        1e-6,
    )
    return out
