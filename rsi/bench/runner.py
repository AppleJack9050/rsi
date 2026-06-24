"""Low-level verify + benchmark primitives used by the MCP tools.

All torch/triton imports are lazy (function-local) so the package imports without a GPU.
"""
from __future__ import annotations

import importlib.util
import math
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable

from .. import config
from .ops import OpSpec, _torch_dtype


def load_kernel_run(path: str | Path) -> tuple[Callable | None, str]:
    """Import a generated kernel module from `path` and return its `run` callable.

    Returns (run, "") on success or (None, error_text) on failure.
    """
    path = Path(path)
    mod_name = f"rsi_kernel_{uuid.uuid4().hex[:12]}"
    try:
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            return None, f"could not create import spec for {path}"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # may raise (syntax / import errors)
    except Exception:
        return None, f"import error:\n{traceback.format_exc()}"
    run = getattr(module, "run", None)
    if run is None or not callable(run):
        return None, "module does not define a callable `run(*inputs)`"
    return run, ""


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _do_bench(fn: Callable[[], Any]) -> float:
    """Return median wall time in ms. Prefers triton.testing.do_bench."""
    import torch
    try:
        import triton.testing as tt
        return float(tt.do_bench(fn, warmup=25, rep=100))
    except Exception:
        # Manual CUDA-event timing fallback.
        torch.cuda.synchronize()
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        n = 50
        start.record()
        for _ in range(n):
            fn()
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end) / n)


def _geomean(xs: list[float]) -> float:
    xs = [x for x in xs if x and x > 0]
    if not xs:
        return 0.0
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def compile_check(spec: OpSpec, path: str | Path) -> dict:
    """Import the kernel and force a single launch on the smallest shape (JIT compile)."""
    import torch
    run, err = load_kernel_run(path)
    if run is None:
        return {"compiles": False, "log": err}
    if not torch.cuda.is_available():
        return {"compiles": False, "log": "CUDA not available — Triton kernels require a GPU"}
    dtype = _torch_dtype(config.DIMS.dtype)
    # smallest shape by declared bytes_moved
    shape_name, shape = min(spec.shape_list(), key=lambda kv: spec.bytes_moved(kv[1]))
    try:
        inputs = spec.make_inputs(shape, _device(), dtype)
        out = run(*inputs)
        torch.cuda.synchronize()
        if not torch.is_tensor(out):
            return {"compiles": False, "log": f"run() returned {type(out)}, expected a tensor"}
        return {"compiles": True, "log": f"launched on shape '{shape_name}'; output {tuple(out.shape)} {out.dtype}"}
    except Exception:
        return {"compiles": False, "log": f"launch error:\n{traceback.format_exc()}"}


def verify(spec: OpSpec, path: str | Path) -> dict:
    """Run the kernel against the reference across all shapes; check torch.allclose."""
    import torch
    run, err = load_kernel_run(path)
    if run is None:
        return {"correct": False, "error": err, "per_shape": {}}
    if not torch.cuda.is_available():
        return {"correct": False, "error": "CUDA not available", "per_shape": {}}
    dtype = _torch_dtype(config.DIMS.dtype)
    per_shape: dict[str, Any] = {}
    all_ok = True
    max_abs = max_rel = 0.0
    for shape_name, shape in spec.shape_list():
        try:
            inputs = spec.make_inputs(shape, _device(), dtype)
            out = run(*inputs)
            ref = spec.reference(*inputs)
            torch.cuda.synchronize()
            if out.shape != ref.shape:
                per_shape[shape_name] = {"ok": False, "reason": f"shape {tuple(out.shape)} != ref {tuple(ref.shape)}"}
                all_ok = False
                continue
            of, rf = out.float(), ref.float()
            abs_err = (of - rf).abs()
            rel_err = abs_err / (rf.abs() + 1e-6)
            a, r = float(abs_err.max()), float(rel_err.max())
            max_abs, max_rel = max(max_abs, a), max(max_rel, r)
            ok = bool(torch.allclose(of, rf, atol=spec.atol, rtol=spec.rtol))
            per_shape[shape_name] = {"ok": ok, "max_abs_err": a, "max_rel_err": r}
            all_ok = all_ok and ok
        except Exception:
            per_shape[shape_name] = {"ok": False, "reason": traceback.format_exc()}
            all_ok = False
    return {"correct": all_ok, "max_abs_err": max_abs, "max_rel_err": max_rel,
            "atol": spec.atol, "rtol": spec.rtol, "per_shape": per_shape, "error": ""}


def benchmark(spec: OpSpec, path: str | Path) -> dict:
    """Benchmark kernel vs reference across shapes; return per-shape + geomean speedup."""
    import torch
    run, err = load_kernel_run(path)
    if run is None:
        return {"ok": False, "error": err}
    if not torch.cuda.is_available():
        return {"ok": False, "error": "CUDA not available"}
    dtype = _torch_dtype(config.DIMS.dtype)
    per_shape: dict[str, Any] = {}
    speedups: list[float] = []
    for shape_name, shape in spec.shape_list():
        try:
            inputs = spec.make_inputs(shape, _device(), dtype)
            # correctness gate per shape before timing
            out = run(*inputs)
            ref = spec.reference(*inputs)
            torch.cuda.synchronize()
            ok = out.shape == ref.shape and bool(
                torch.allclose(out.float(), ref.float(), atol=spec.atol, rtol=spec.rtol))
            k_ms = _do_bench(lambda: run(*inputs))
            r_ms = _do_bench(lambda: spec.reference(*inputs))
            # Speed baseline: a stronger one (e.g. FlashAttention) if the op
            # defines it, else the eager reference. speedup is measured vs THIS.
            b_ms = r_ms if spec.baseline is None else _do_bench(lambda: spec.baseline(*inputs))
            speedup = (b_ms / k_ms) if k_ms > 0 else 0.0
            gbps = spec.bytes_moved(shape) / (k_ms * 1e-3) / 1e9 if k_ms > 0 else 0.0
            per_shape[shape_name] = {
                "correct": ok,
                "kernel_ms": round(k_ms, 5),
                "ref_ms": round(r_ms, 5),
                "baseline_ms": round(b_ms, 5),
                "baseline": spec.baseline_name,
                "speedup": round(speedup, 3),
                "achieved_gbps": round(gbps, 1),
                "pct_peak_bw": round(100 * gbps / config.HW.peak_bw_gbps, 1),
            }
            if ok:
                speedups.append(speedup)
        except Exception:
            per_shape[shape_name] = {"correct": False, "reason": traceback.format_exc()}
    geo = _geomean(speedups)
    return {"ok": True, "geomean_speedup": round(geo, 3),
            "n_correct_shapes": len(speedups), "n_shapes": len(spec.shapes),
            "per_shape": per_shape, "error": ""}
