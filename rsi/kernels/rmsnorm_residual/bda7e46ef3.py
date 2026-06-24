
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16, num_stages=3),
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
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)

    # Single-pass: load x and residual once, keep h in registers
    x   = tl.load(x_ptr       + row * stride_xT + offs).to(tl.float32)
    res = tl.load(residual_ptr + row * stride_rT + offs).to(tl.float32)
    h   = x + res

    # RMS normalization (accumulate in fp32)
    var = tl.sum(h * h, axis=0) / H
    rms = tl.rsqrt(var + eps)

    # Load weight (keep in fp16), cast h*rms to fp16 and multiply fp16*fp16
    w   = tl.load(weight_ptr + offs)          # fp16
    out = (h * rms).to(tl.float16) * w        # fp16 output

    tl.store(out_ptr + row * stride_oT + offs, out)


def run(x, residual, weight):
    T, H = x.shape
    out  = torch.empty_like(x)
    rmsnorm_residual_kernel[(T,)](
        x, residual, weight, out,
        T, H,
        x.stride(0), residual.stride(0), out.stride(0),
        1e-6,
    )
    return out
