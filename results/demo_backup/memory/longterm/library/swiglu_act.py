
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        # BLOCK_SIZE=128: decode gets 32*48=1536 CTAs → fills 170 SMs
        triton.Config({'BLOCK_SIZE': 128}, num_warps=1, num_stages=1),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=1, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=1, num_stages=3),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=3),
        # BLOCK_SIZE=256: decode gets 32*24=768 CTAs
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=3),
        # BLOCK_SIZE=512: decode gets 32*12=384 CTAs; warps=2 → 128-bit vectorized loads
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16, num_stages=3),
        # BLOCK_SIZE=1024: warps=4 → 128-bit vectorized loads (8 fp16/thread)
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
        # BLOCK_SIZE=2048: warps=8 → 128-bit vectorized loads (8 fp16/thread)
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

    # Load both inputs upfront so hardware can pipeline the two HBM fetches.
    # Keep 'up' in fp16 — no upcast needed since final multiply is fp16.
    # evict_first = streaming, don't pollute L1.
    gate = tl.load(gate_ptr + row_offset + offs, eviction_policy='evict_first')  # fp16
    up   = tl.load(up_ptr   + row_offset + offs, eviction_policy='evict_first')  # fp16

    # sigmoid requires fp32; upcast gate only (not up — saves 1 upcast per element).
    gate_f32 = gate.to(tl.float32)
    # silu(gate) = gate * sigmoid(gate), computed in fp32 for precision.
    gate_silu_f16 = (gate_f32 * tl.sigmoid(gate_f32)).to(tl.float16)

    # Final multiply in fp16: up stays fp16, saves 1 upcast + 1 downcast vs doing it in fp32.
    # Also reduces peak register usage: 'up' never occupies fp32 registers.
    out = gate_silu_f16 * up  # fp16 * fp16 → fp16

    # Store fp16 directly — no downcast instruction needed.
    tl.store(out_ptr + row_offset + offs, out, eviction_policy='evict_first')


def run(gate, up):
    T, I = gate.shape   # I=6144; all BLOCK_SIZEs divide 6144 exactly, no masking needed
    out = torch.empty_like(gate)

    def grid(meta):
        return (T, triton.cdiv(I, meta['BLOCK_SIZE']))

    swiglu_act_kernel[grid](
        gate, up, out,
        T, I,
    )
    return out
