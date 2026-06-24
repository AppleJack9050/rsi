#!/usr/bin/env python3
"""Deterministic smoke test — proves the toolchain + harness work BEFORE spending LLM budget.

  1. torch sees the RTX 5090 (Blackwell sm_120)
  2. a trivial Triton kernel compiles + runs + matches torch
  3. the rsi bench harness verifies & benchmarks a known-correct rmsnorm_residual kernel
  4. the dual-level memory store round-trips

Run:  python3 scripts/smoke_test.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def step(msg):
    print(f"\n=== {msg} ===")


def main() -> int:
    step("1. torch + CUDA")
    import torch
    print("torch", torch.__version__, "| cuda", torch.version.cuda, "| available", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("FAIL: CUDA not available — Triton kernels need a GPU.")
        return 1
    print("device", torch.cuda.get_device_name(0), "| cap", torch.cuda.get_device_capability(0))

    step("2. trivial Triton kernel (vector add)")
    import triton
    import triton.language as tl

    @triton.jit
    def _add(a, b, c, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(c + offs, tl.load(a + offs, mask=mask) + tl.load(b + offs, mask=mask), mask=mask)

    n = 8192
    a = torch.randn(n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, device="cuda", dtype=torch.float16)
    c = torch.empty_like(a)
    _add[(triton.cdiv(n, 1024),)](a, b, c, n, BLOCK=1024)
    torch.cuda.synchronize()
    assert torch.allclose(c.float(), (a + b).float(), atol=1e-2), "triton add mismatch"
    print("OK: Triton add matches torch")

    step("3. rsi harness on a known-correct rmsnorm_residual kernel")
    from rsi.bench.ops import get_ops
    from rsi.bench import runner

    spec = get_ops()["rmsnorm_residual"]
    KERNEL = '''
import torch, triton, triton.language as tl

@triton.jit
def _rms(x_ptr, r_ptr, w_ptr, o_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    base = row * n_cols + offs
    x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + base, mask=mask, other=0.0).to(tl.float32)
    h = x + r
    var = tl.sum(h * h, axis=0) / n_cols
    inv = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    o = (h * inv) * w
    tl.store(o_ptr + base, o.to(o_ptr.dtype.element_ty), mask=mask)

def run(x, residual, weight):
    T, H = x.shape
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(H)
    _rms[(T,)](x, residual, weight, out, H, 1e-6, BLOCK=BLOCK, num_warps=8)
    return out
'''
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "k.py"
        path.write_text(KERNEL)
        comp = runner.compile_check(spec, path)
        print("compile:", comp)
        assert comp["compiles"], "kernel did not compile"
        ver = runner.verify(spec, path)
        print("verify:", {k: ver[k] for k in ("correct", "max_abs_err", "max_rel_err")})
        assert ver["correct"], f"kernel incorrect: {ver}"
        bench = runner.benchmark(spec, path)
        print("benchmark geomean speedup:", bench["geomean_speedup"])
        for s, m in bench["per_shape"].items():
            print(f"  {s}: kernel {m.get('kernel_ms')}ms vs ref {m.get('ref_ms')}ms "
                  f"-> {m.get('speedup')}x  ({m.get('pct_peak_bw')}% peak BW)")

    step("4. dual-level memory round-trip")
    from rsi.memory import MemoryStore
    mem = MemoryStore()
    mem.write_strategy("rmsnorm_residual", "single-pass row fusion beats torch's add+norm launches",
                       tags=["fusion", "memory-bound"], speedup=bench["geomean_speedup"])
    ctx = mem.context_for("rmsnorm_residual")
    print("strategies stored:", len(ctx["strategies"]))
    assert ctx["strategies"], "memory write/read failed"

    print("\nALL SMOKE CHECKS PASSED ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
