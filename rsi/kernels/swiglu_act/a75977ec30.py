
import torch
import triton
import triton.language as tl

# I=6144 = 2^11 * 3.  All power-of-2 BLOCK_SIZEs up to 2048 divide 6144 exactly:
#   6144/64=96, /128=48, /256=24, /512=12, /1024=6, /2048=3 → no mask needed.
# For decode (T=32): want many CTAs → small BLOCK_SIZE wins.
# For prefill (T=4096): already huge #CTAs → large BLOCK_SIZE preferred.
# Autotune with key=['T','I'] picks different configs per shape.

@triton.autotune(
    configs=[
        # BLOCK_SIZE=64  → decode: 32*96=3072 CTAs (~18 waves on 170 SMs)
        triton.Config({'BLOCK_SIZE': 64},   num_warps=2,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 64},   num_warps=4,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 64},   num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 64},   num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 64},   num_warps=2,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 64},   num_warps=4,  num_stages=3),
        # BLOCK_SIZE=128 → decode: 32*48=1536 CTAs (~9 waves)
        triton.Config({'BLOCK_SIZE': 128},  num_warps=2,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=4,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=2,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=4,  num_stages=3),
        # BLOCK_SIZE=256 → decode: 32*24=768 CTAs (~4.5 waves)
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=8,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=8,  num_stages=3),
        # BLOCK_SIZE=512 → decode: 32*12=384 CTAs
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=8,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=16, num_stages=1),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=16, num_stages=3),
        # BLOCK_SIZE=1024 → decode: 32*6=192 CTAs
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
        # BLOCK_SIZE=2048 → decode: 32*3=96 CTAs; prefill: 4096*3=12288
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

    # Both loads issued upfront so memory pipeline overlaps them.
    # evict_first = streaming — don't displace other data from L1.
    gate = tl.load(gate_ptr + row_offset + offs, eviction_policy='evict_first').to(tl.float32)
    up   = tl.load(up_ptr   + row_offset + offs, eviction_policy='evict_first').to(tl.float32)

    # silu(gate) * up in fp32
    gate_silu = gate * tl.sigmoid(gate)
    out = gate_silu * up

    tl.store(out_ptr + row_offset + offs, out.to(tl.float16),
             eviction_policy='evict_first')


def run(gate, up):
    T, I = gate.shape   # I=6144 always
    out = torch.empty_like(gate)

    def grid(meta):
        return (T, triton.cdiv(I, meta['BLOCK_SIZE']))

    swiglu_act_kernel[grid](gate, up, out, T, I)
    return out
