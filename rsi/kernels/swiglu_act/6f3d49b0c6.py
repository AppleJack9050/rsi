import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Flat 1D element-parallel grid. Pick (BLOCK, warps) so each thread loads
        # 8 contiguous fp16 = 128-bit vectorized: BLOCK / (warps*32) == 8.
        triton.Config({'BLOCK': 1024},  num_warps=4,  num_stages=1),
        triton.Config({'BLOCK': 2048},  num_warps=8,  num_stages=1),
        triton.Config({'BLOCK': 4096},  num_warps=16, num_stages=1),
        triton.Config({'BLOCK': 8192},  num_warps=32, num_stages=1),
        triton.Config({'BLOCK': 2048},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK': 4096},  num_warps=16, num_stages=2),
        triton.Config({'BLOCK': 8192},  num_warps=32, num_stages=2),
        # a few non-vectorized-ratio fallbacks
        triton.Config({'BLOCK': 2048},  num_warps=4,  num_stages=1),
        triton.Config({'BLOCK': 4096},  num_warps=8,  num_stages=1),
        triton.Config({'BLOCK': 1024},  num_warps=8,  num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def swiglu_flat_kernel(gate_ptr, up_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    g = tl.load(gate_ptr + offs, mask=mask, other=0.0)            # fp16
    u = tl.load(up_ptr + offs, mask=mask, other=0.0)             # fp16
    gf = g.to(tl.float32)
    silu = (gf * tl.sigmoid(gf)).to(tl.float16)                  # silu(gate) in fp16
    tl.store(out_ptr + offs, silu * u, mask=mask)               # fp16 * fp16


def run(gate, up):
    # gate, up: [T, I] contiguous -> flatten to N = T*I element-parallel work
    out = torch.empty_like(gate)
    N = gate.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK']),)
    swiglu_flat_kernel[grid](gate, up, out, N)
    return out
