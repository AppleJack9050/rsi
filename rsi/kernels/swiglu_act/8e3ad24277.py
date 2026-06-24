
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        # BLOCK_SIZE=256: I/256=24 chunks → decode gets 32*24=768 CTAs
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8,  num_stages=3),
        # BLOCK_SIZE=512: I/512=12 chunks → decode gets 32*12=384 CTAs
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16, num_stages=3),
        # BLOCK_SIZE=1024: I/1024=6 chunks → decode gets 32*6=192 CTAs
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=32, num_stages=1),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=32, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=32, num_stages=3),
        # BLOCK_SIZE=2048: I/2048=3 chunks → decode gets 32*3=96 CTAs
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

    # Both loads issued upfront: hardware can pipeline gate and up fetches.
    # evict_first = streaming data, don't pollute L1.
    gate = tl.load(gate_ptr + row_offset + offs, eviction_policy='evict_first').to(tl.float32)
    up   = tl.load(up_ptr   + row_offset + offs, eviction_policy='evict_first').to(tl.float32)

    # silu(gate) * up, computed in fp32 for precision
    gate_silu = gate * tl.sigmoid(gate)
    out = gate_silu * up

    tl.store(out_ptr + row_offset + offs, out.to(tl.float16),
             eviction_policy='evict_first')


def run(gate, up):
    T, I = gate.shape   # I = 6144 = 2^11 * 3; all power-of-2 BLOCK_SIZEs up to 2048 divide it
    out = torch.empty_like(gate)

    # Grid is determined at autotune time via meta['BLOCK_SIZE']
    def grid(meta):
        return (T, triton.cdiv(I, meta['BLOCK_SIZE']))

    swiglu_act_kernel[grid](
        gate, up, out,
        T, I,
    )
    return out
