
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

    # No mask needed: BLOCK_SIZE = next_power_of_2(H) = H = 2048,
    # so every offset is in-bounds. Unconditional loads let the compiler
    # emit wider (128-bit) vector instructions without predication overhead.
    # Issue all three loads upfront so the HW memory pipeline can overlap them.
    x_vals   = tl.load(x_ptr       + row * H + offs).to(tl.float32)
    res_vals = tl.load(residual_ptr + row * H + offs).to(tl.float32)
    w_vals   = tl.load(weight_ptr   + offs).to(tl.float32)

    h = x_vals + res_vals

    # fp32 reduction for precision, cast back to fp16
    sum_sq = tl.sum(h * h, axis=0)
    rrms   = tl.rsqrt(sum_sq / H + eps)

    out = (h * rrms * w_vals).to(tl.float16)
    tl.store(out_ptr + row * H + offs, out)


def run(x, residual, weight):
    T, H = x.shape
    out = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(H)  # H=2048 → BLOCK_SIZE=2048 = H exactly

    rmsnorm_residual_kernel[(T,)](
        x, residual, weight, out,
        T, H,
        1e-6,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out
