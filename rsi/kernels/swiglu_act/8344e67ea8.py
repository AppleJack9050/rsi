
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=32, num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=32, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=32, num_stages=3),
    ],
    key=['T', 'I'],
)
@triton.jit
def swiglu_act_kernel(
    gate_ptr, up_ptr, out_ptr,
    T, I,
    BLOCK_SIZE: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    col_start  = pid_col * BLOCK_SIZE
    offs       = col_start + tl.arange(0, BLOCK_SIZE)
    row_offset = pid_row * I

    # Issue both independent loads upfront so the hardware memory pipeline
    # can overlap them. evict_first = streaming data, don't pollute L1.
    gate = tl.load(gate_ptr + row_offset + offs, eviction_policy='evict_first').to(tl.float32)
    up   = tl.load(up_ptr   + row_offset + offs, eviction_policy='evict_first').to(tl.float32)

    # silu(gate) * up, accumulated in fp32
    gate_silu = gate * tl.sigmoid(gate)
    out = gate_silu * up

    tl.store(out_ptr + row_offset + offs, out.to(tl.float16),
             eviction_policy='evict_first')


def run(gate, up):
    T, I = gate.shape           # I = 6144 = 3 * 2048
    out = torch.empty_like(gate)

    BLOCK_SIZE    = 2048
    num_col_blocks = I // BLOCK_SIZE   # = 3, exact — no mask needed

    grid = (T, num_col_blocks)

    swiglu_act_kernel[grid](
        gate, up, out,
        T, I,
    )
    return out
