"""Op registry (KernelBench-style): each OpSpec bundles the reference, the canonical
benchmark shapes for Qwen3-1.7B, correctness tolerance, an input factory, and a
bytes-moved model used to score memory-bound kernels.

A generated kernel module must expose `run(*inputs) -> Tensor` taking inputs in the
order given by `input_names` and returning a tensor matching `reference(*inputs)`.

torch is imported lazily inside `get_ops()` so this module imports without a GPU.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

from .. import config


@dataclass
class OpSpec:
    name: str
    summary: str
    signature_hint: str
    input_names: list[str]
    shapes: dict[str, dict]                       # named shape configs
    make_inputs: Callable[[dict, str, Any], tuple]  # (shape, device, dtype) -> tensors
    reference: Callable[..., Any]
    bytes_moved: Callable[[dict], int]
    atol: float = 2e-2
    rtol: float = 2e-2
    reference_source: str = ""
    # Optional stronger SPEED baseline (e.g. FlashAttention) timed instead of the
    # eager reference. Correctness is always checked against `reference`.
    baseline: Callable[..., Any] | None = None
    baseline_name: str = "torch-eager"

    def shape_list(self) -> list[tuple[str, dict]]:
        return list(self.shapes.items())


def _torch_dtype(dtype_str: str):
    import torch
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_str]


# --------------------------------------------------------------------------- #
# FlashAttention-class baseline for the attention op.
#
# On Blackwell (sm_120) FlashAttention-3 (Hopper-only) is unavailable; the
# strongest runnable FA-class kernel is torch SDPA's fused backend — cuDNN's
# fused attention (fastest here) or FlashAttention-2. We pick the fastest
# available backend once and time the attention kernel against it, so the
# reported speedup is "vs FlashAttention", not vs unfused eager torch.
# --------------------------------------------------------------------------- #
_FLASH_BASELINE_NAME = "FlashAttention (SDPA: cuDNN→FA-2)"
_flash_state: dict = {"backend": None, "label": None, "resolved": False}


def _select_flash_backend():
    """Pick + cache the fastest available SDPA fused attention backend (or None)."""
    if _flash_state["resolved"]:
        return _flash_state["backend"]
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    q = torch.randn(1, 8, 1, 128, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 4, 16, 128, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 4, 16, 128, device="cuda", dtype=torch.float16)
    for be, label in ((SDPBackend.CUDNN_ATTENTION, "cuDNN"),
                      (SDPBackend.FLASH_ATTENTION, "FA-2")):
        try:
            with sdpa_kernel(be):
                F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
            torch.cuda.synchronize()
            _flash_state.update(backend=be, label=label, resolved=True)
            return be
        except Exception:
            continue
    _flash_state.update(backend=None, label="SDPA-default", resolved=True)
    return None


def flash_backend_label() -> str:
    _select_flash_backend()
    return _flash_state["label"] or "SDPA-default"


def _flash_attention_baseline(q, k, v):
    """FlashAttention-class GQA decode via torch SDPA fused backend.

    q=[B, Hq, D], k,v=[B, Hkv, S, D]  ->  out=[B, Hq, D].
    """
    import torch.nn.functional as F
    from torch.nn.attention import sdpa_kernel
    be = _select_flash_backend()
    qq = q.unsqueeze(2)  # [B, Hq, 1, D]
    if be is not None:
        with sdpa_kernel(be):
            o = F.scaled_dot_product_attention(qq, k, v, is_causal=False, enable_gqa=True)
    else:
        o = F.scaled_dot_product_attention(qq, k, v, is_causal=False, enable_gqa=True)
    return o.squeeze(2)


def get_ops() -> dict[str, OpSpec]:
    """Build the op registry. Imports torch + the reference module (GPU runtime)."""
    import torch
    from . import reference as ref

    D = config.DIMS
    H, I, Hq, Hkv, HD = D.hidden, D.intermediate, D.n_q_heads, D.n_kv_heads, D.head_dim
    eps = D.eps
    itemsize = 4 if D.dtype == "float32" else 2

    def randn(shape, device, dtype):
        return torch.randn(shape, device=device, dtype=dtype)

    ops: dict[str, OpSpec] = {}

    # 1. fused residual-add + RMSNorm ---------------------------------------- #
    def mk_rmsnorm(shape, device, dtype):
        T = shape["T"]
        return (randn((T, H), device, dtype),
                randn((T, H), device, dtype),
                randn((H,), device, dtype))

    ops["rmsnorm_residual"] = OpSpec(
        name="rmsnorm_residual",
        summary="Fused residual-add + RMSNorm over the hidden dim, scaled by weight.",
        signature_hint=("run(x, residual, weight) -> out  | shapes: x,residual=[T, H], "
                        f"weight=[H]; H={H}, eps={eps}. out = RMSNorm(x+residual)*weight "
                        "(reduce over last dim; accumulate in fp32)."),
        input_names=["x", "residual", "weight"],
        shapes={"decode": {"T": 32}, "prefill": {"T": 4096}},
        make_inputs=mk_rmsnorm,
        reference=lambda x, r, w: ref.rmsnorm_residual(x, r, w, eps),
        bytes_moved=lambda s: int(3 * s["T"] * H * itemsize + H * itemsize),
        atol=3e-2, rtol=3e-2,
    )

    # 2. SwiGLU activation --------------------------------------------------- #
    def mk_swiglu(shape, device, dtype):
        T = shape["T"]
        return (randn((T, I), device, dtype), randn((T, I), device, dtype))

    ops["swiglu_act"] = OpSpec(
        name="swiglu_act",
        summary="SwiGLU activation: silu(gate) * up, elementwise.",
        signature_hint=(f"run(gate, up) -> out  | gate,up=[T, I]; I={I}. out = silu(gate)*up "
                        "(silu(z)=z*sigmoid(z))."),
        input_names=["gate", "up"],
        shapes={"decode": {"T": 32}, "prefill": {"T": 4096}},
        make_inputs=mk_swiglu,
        reference=lambda g, u: ref.swiglu_act(g, u),
        bytes_moved=lambda s: int(3 * s["T"] * I * itemsize),
        atol=2e-2, rtol=2e-2,
    )

    # 3. RoPE ---------------------------------------------------------------- #
    def mk_rope(shape, device, dtype):
        T = shape["T"]
        return (randn((T, Hq, HD), device, dtype),
                randn((T, HD), device, dtype),
                randn((T, HD), device, dtype))

    ops["rope"] = OpSpec(
        name="rope",
        summary="Rotary positional embedding applied to Q/K heads.",
        signature_hint=(f"run(x, cos, sin) -> out  | x=[T, Hq, D] (Hq={Hq}, D={HD}), "
                        "cos,sin=[T, D] (broadcast over heads). "
                        "out = x*cos + rotate_half(x)*sin; rotate_half([a,b])=[-b,a] on D halves."),
        input_names=["x", "cos", "sin"],
        shapes={"decode": {"T": 32}, "prefill": {"T": 4096}},
        make_inputs=mk_rope,
        reference=lambda x, c, s: ref.rope(x, c, s),
        bytes_moved=lambda s: int(2 * s["T"] * Hq * HD * itemsize + 2 * s["T"] * HD * itemsize),
        atol=2e-2, rtol=2e-2,
    )

    # 4. row-wise softmax ---------------------------------------------------- #
    def mk_softmax(shape, device, dtype):
        return (randn((shape["rows"], shape["n"]), device, dtype),)

    ops["softmax"] = OpSpec(
        name="softmax",
        summary="Numerically-stable row softmax over the last dim (fp32 accumulation).",
        signature_hint="run(x) -> out  | x=[rows, n]; out = softmax(x, dim=-1).",
        input_names=["x"],
        shapes={"narrow": {"rows": 4096, "n": 1024}, "wide": {"rows": 2048, "n": 4096}},
        make_inputs=mk_softmax,
        reference=lambda x: ref.softmax(x),
        bytes_moved=lambda s: int(2 * s["rows"] * s["n"] * itemsize),
        atol=2e-2, rtol=2e-2,
    )

    # 5. GQA decode attention (STRETCH) ------------------------------------- #
    def mk_attn(shape, device, dtype):
        B, S = shape["B"], shape["S"]
        return (randn((B, Hq, HD), device, dtype),
                randn((B, Hkv, S, HD), device, dtype),
                randn((B, Hkv, S, HD), device, dtype))

    ops["gqa_decode_attention"] = OpSpec(
        name="gqa_decode_attention",
        summary="Single-query GQA attention over a KV cache (FlashAttention-decode style).",
        signature_hint=(f"run(q, k, v) -> out  | q=[B, Hq, D], k,v=[B, Hkv, S, D]; "
                        f"Hq={Hq}, Hkv={Hkv} (group={Hq//Hkv}), D={HD}, scale=1/sqrt(D). "
                        "softmax over S in fp32; out=[B, Hq, D]."),
        input_names=["q", "k", "v"],
        shapes={"s1k": {"B": 32, "S": 1024}, "s4k": {"B": 16, "S": 4096}},
        make_inputs=mk_attn,
        reference=lambda q, k, v: ref.gqa_decode_attention(q, k, v),
        bytes_moved=lambda s: int(2 * s["B"] * Hkv * s["S"] * HD * itemsize),
        atol=3e-2, rtol=3e-2,
        baseline=_flash_attention_baseline,
        baseline_name=_FLASH_BASELINE_NAME,
    )

    # attach reference source text (for the LLM to read exact semantics)
    src_by_fn = {
        "rmsnorm_residual": ref.rmsnorm_residual,
        "swiglu_act": ref.swiglu_act,
        "rope": ref.rope,         # also pulls in _rotate_half via the module
        "softmax": ref.softmax,
        "gqa_decode_attention": ref.gqa_decode_attention,
    }
    for name, spec in ops.items():
        try:
            spec.reference_source = inspect.getsource(src_by_fn[name])
        except Exception:
            spec.reference_source = "(source unavailable)"

    return ops


# Names that make up the default priority queue (memory-bound decode wins first;
# attention is the stretch target).
DEFAULT_OP_ORDER = ["rmsnorm_residual", "swiglu_act", "rope", "softmax", "gqa_decode_attention"]
