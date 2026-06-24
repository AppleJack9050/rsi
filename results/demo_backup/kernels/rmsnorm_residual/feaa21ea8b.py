
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'ROWS_PER_PROG': rpp}, num_warps=nw, num_stages=ns)
        for rpp in [1, 2, 4, 8, 16, 32]
        for nw in [4, 8, 16, 32]
        for ns in [1, 2, 3]
    ],
    key=['T', 'H'],
)
@triton.jit
def rmsnorm_residual_kernel(
    x_ptr, residual_ptr, weight_ptr, out_ptr,
    T, H,
    eps,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    start_row = tl.program_id(0) * ROWS_PER_PROG

    offs = tl.arange(0, BLOCK_SIZE)
    base_mask = offs < H

    # Load weight ONCE for all rows in this program — amortises the weight-load
    # cost across ROWS_PER_PROG rows (saving (RPP-1)/RPP weight loads vs baseline).
    # Issue it first so the hardware memory pipeline can overlap it with later loads.
    w_vals = tl.load(weight_ptr + offs, mask=base_mask, other=0.0).to(tl.float32)

    # Unrolled loop (tl.static_range) so each iteration is fully independent
    # and the compiler can schedule loads/maths optimally.
    for i in tl.static_range(ROWS_PER_PROG):
        row = start_row + i
        # scalar guard: skips OOB rows when T is not a multiple of ROWS_PER_PROG
        valid = row < T
        row_mask = base_mask & valid   # broadcast scalar bool over the vector

        # Issue both row loads upfront so the memory pipeline can overlap them
        x_vals   = tl.load(x_ptr       + row * H + offs, mask=row_mask, other=0.0).to(tl.float32)
        res_vals = tl.load(residual_ptr + row * H + offs, mask=row_mask, other=0.0).to(tl.float32)

        h = x_vals + res_vals

        # fp32 reduction, then scale and cast back to fp16
        sum_sq = tl.sum(h * h, axis=0)
        rrms   = tl.rsqrt(sum_sq / H + eps)

        out = (h * rrms * w_vals).to(tl.float16)
        tl.store(out_ptr + row * H + offs, out, mask=row_mask)


def run(x, residual, weight):
    T, H = x.shape
    out = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(H)  # H=2048 → 2048

    def grid(meta):
        return ((T + meta['ROWS_PER_PROG'] - 1) // meta['ROWS_PER_PROG'],)

    rmsnorm_residual_kernel[grid](
        x, residual, weight, out,
        T, H,
        1e-6,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out
