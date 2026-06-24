
import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_residual_kernel(
    x_ptr, residual_ptr, weight_ptr, out_ptr,
    T, H,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # One program per row
    row = tl.program_id(0)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < H

    # Load x and residual, add them in fp32 (single pass)
    x_vals   = tl.load(x_ptr       + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    res_vals = tl.load(residual_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    h = x_vals + res_vals

    # Compute RMS over the hidden dim
    sum_sq = tl.sum(h * h, axis=0)
    rrms   = tl.rsqrt(sum_sq / H + eps)

    # Load weight (fp16 -> fp32)
    w_vals = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Scale and store as fp16
    out = (h * rrms * w_vals).to(tl.float16)
    tl.store(out_ptr + row * H + offs, out, mask=mask)


def run(x, residual, weight):
    T, H = x.shape
    out = torch.empty_like(x)

    # H=2048 -> BLOCK_SIZE=2048 (one tile covers the whole row)
    BLOCK_SIZE = triton.next_power_of_2(H)

    rmsnorm_residual_kernel[(T,)](
        x, residual, weight, out,
        T, H,
        1e-6,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=1,
    )

    return out
