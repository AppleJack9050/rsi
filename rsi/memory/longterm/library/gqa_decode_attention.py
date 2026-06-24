
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 32},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 32},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 64},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_S': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_S': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_S': 256}, num_warps=8, num_stages=3),
    ],
    key=['S'],
)
@triton.jit
def gqa_decode_fwd_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh,        # q  [B, Hq, D]; stride_qd=1
    stride_kb, stride_kh, stride_ks,  # k  [B, Hkv, S, D]; stride_kd=1
    stride_vb, stride_vh, stride_vs,  # v  [B, Hkv, S, D]; stride_vd=1
    stride_ob, stride_oh,        # out[B, Hq, D]; stride_od=1
    Hq, S,
    scale,
    GROUP: tl.constexpr,
    D:     tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Flash-decode GQA: one CTA per (batch, query-head) pair."""
    pid = tl.program_id(0)
    b  = pid // Hq
    hq = pid % Hq
    kh = hq // GROUP          # grouped KV head index

    d_offs = tl.arange(0, D)

    # Base pointers for this (b, hq) / (b, kh)
    q_base = Q   + b * stride_qb + hq * stride_qh
    k_base = K   + b * stride_kb + kh * stride_kh
    v_base = V   + b * stride_vb + kh * stride_vh
    o_base = Out + b * stride_ob + hq * stride_oh

    # Load query vector [D] -> fp32 (stays in registers)
    q = tl.load(q_base + d_offs).to(tl.float32)  # [D]

    # Online-softmax running state
    m_i = tl.zeros([1], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([1], dtype=tl.float32)
    acc  = tl.zeros([D], dtype=tl.float32)

    # Iterate over the KV sequence in tiles
    for start_s in range(0, S, BLOCK_S):
        s_offs = start_s + tl.arange(0, BLOCK_S)
        mask_s = s_offs < S

        # ---- K tile [BLOCK_S, D] ------------------------------------------------
        k_tile = tl.load(
            k_base + s_offs[:, None] * stride_ks + d_offs[None, :],
            mask=mask_s[:, None], other=0.0
        ).to(tl.float32)

        # Attention logits [BLOCK_S]  =  q · k^T * scale
        scores = tl.sum(q[None, :] * k_tile, axis=1) * scale
        scores = tl.where(mask_s, scores, float('-inf'))

        # ---- Online softmax update ---------------------------------------------
        m_ij  = tl.max(scores, axis=0)          # 0-d (block max)
        m_new = tl.maximum(m_i, m_ij)            # [1]
        alpha = tl.exp(m_i - m_new)              # [1]  rescale factor
        p     = tl.exp(scores - m_new)           # [BLOCK_S]
        p     = tl.where(mask_s, p, 0.0)

        # ---- V tile [BLOCK_S, D] ------------------------------------------------
        v_tile = tl.load(
            v_base + s_offs[:, None] * stride_vs + d_offs[None, :],
            mask=mask_s[:, None], other=0.0
        ).to(tl.float32)

        # Accumulate
        l_i  = alpha * l_i + tl.sum(p, axis=0)                      # [1]
        acc  = alpha * acc + tl.sum(p[:, None] * v_tile, axis=0)     # [D]
        m_i  = m_new

    # Normalise and write output
    out = (acc / l_i).to(tl.float16)   # [D]
    tl.store(o_base + d_offs, out)


def run(q, k, v):
    B, Hq, D = q.shape
    Hkv = k.shape[1]
    S   = k.shape[2]
    GROUP = Hq // Hkv
    scale = float(D ** -0.5)

    out  = torch.empty_like(q)
    grid = (B * Hq,)

    gqa_decode_fwd_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1),
        Hq, S, scale,
        GROUP=GROUP,
        D=128,
    )
    return out
