"""In-process MCP tool server exposed to the kernel agents.

These tools are the deterministic backbone: the LLM writes Triton source, and these
tools compile / verify (torch.allclose) / benchmark (triton.do_bench) / profile it,
persist every attempt to disk, and read/write the dual-level memory. Because they run
in the SAME process as the orchestrator, they call torch/triton directly — no shell
access is needed for the core loop, which keeps the agents tightly sandboxed.

Tool names exposed to agents are `mcp__rsi__<name>`.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool

from . import config
from .memory import MemoryStore

# Shared singletons (tools run in-process).
MEM = MemoryStore()
_OPS: dict[str, Any] | None = None


def ops() -> dict[str, Any]:
    """Lazily build the op registry (imports torch on first call)."""
    global _OPS
    if _OPS is None:
        from .bench.ops import get_ops
        _OPS = get_ops()
    return _OPS


# --------------------------------------------------------------------------- #
# Kernel store (per-op dir of <kernel_id>.py + <kernel_id>.json sidecars)
# --------------------------------------------------------------------------- #
def _op_dir(op: str) -> Path:
    d = config.KERNELS_DIR / op
    d.mkdir(parents=True, exist_ok=True)
    return d


def _kernel_id(code: str) -> str:
    return hashlib.sha1(code.encode()).hexdigest()[:10]


def _sidecar(op: str, kid: str) -> Path:
    return _op_dir(op) / f"{kid}.json"


def _load_sidecar(op: str, kid: str) -> dict:
    p = _sidecar(op, kid)
    return json.loads(p.read_text()) if p.exists() else {"op": op, "kernel_id": kid}


def _save_sidecar(op: str, kid: str, **fields) -> dict:
    d = _load_sidecar(op, kid)
    d.update(fields)
    d["ts"] = time.time()
    _sidecar(op, kid).write_text(json.dumps(d, indent=2))
    return d


def best_on_disk(op: str) -> dict | None:
    """Best correct kernel for an op by geomean speedup, scanning sidecars."""
    best = None
    for p in _op_dir(op).glob("*.json"):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if d.get("correct") and (d.get("speedup") or 0) > (best.get("speedup", 0) if best else 0):
            best = d
    return best


def kernel_code(op: str, kid: str) -> str:
    p = _op_dir(op) / f"{kid}.py"
    return p.read_text() if p.exists() else ""


# --------------------------------------------------------------------------- #
# Leaderboard
# --------------------------------------------------------------------------- #
def _read_leaderboard() -> dict:
    if config.LEADERBOARD.exists():
        return json.loads(config.LEADERBOARD.read_text())
    return {}


def _update_leaderboard(op: str, kid: str, speedup: float, per_shape: dict) -> bool:
    lb = _read_leaderboard()
    cur = lb.get(op)
    if cur and cur.get("speedup", 0) >= speedup:
        return False
    lb[op] = {"kernel_id": kid, "speedup": speedup, "per_shape": per_shape, "ts": time.time()}
    config.LEADERBOARD.write_text(json.dumps(lb, indent=2))
    return True


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _text(payload: Any) -> dict:
    """Wrap a structured payload as an MCP text result (JSON for easy LLM parsing)."""
    return {"content": [{"type": "text", "text": json.dumps(payload, indent=2, default=str)}]}


def _require_op(op: str):
    o = ops()
    if op not in o:
        return None, _text({"error": f"unknown op '{op}'. available: {list(o)}"})
    return o[op], None


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
@tool("list_ops", "List target ops, their summaries, and the current best speedup for each.", {})
async def list_ops(args: dict) -> dict:
    lb = _read_leaderboard()
    o = ops()
    rows = [{
        "op": name, "summary": spec.summary,
        "best_speedup": lb.get(name, {}).get("speedup"),
        "target": config.TARGET_SPEEDUP,
    } for name, spec in o.items()]
    return _text({"model": config.DIMS.name, "gpu": config.HW.name, "ops": rows})


@tool("get_op_spec",
      "Get reference semantics, benchmark shapes, tolerance and the required run() signature for an op.",
      {"op": str})
async def get_op_spec(args: dict) -> dict:
    spec, err = _require_op(args["op"])
    if err:
        return err
    return _text({
        "op": spec.name,
        "summary": spec.summary,
        "signature": spec.signature_hint,
        "input_names": spec.input_names,
        "shapes": spec.shapes,
        "tolerance": {"atol": spec.atol, "rtol": spec.rtol},
        "dtype": config.DIMS.dtype,
        "gpu": f"{config.HW.name} ({config.HW.arch}), peak BW ~{config.HW.peak_bw_gbps} GB/s",
        "reference_source": spec.reference_source,
        "contract": ("Write a Python module that imports triton+torch, defines a @triton.jit "
                     "kernel, and exposes `run(*inputs)` (inputs in input_names order) returning "
                     "ONE tensor matching the reference. Accumulate reductions in fp32."),
    })


@tool("compile_triton_kernel",
      "Save a Triton kernel module (must define run(*inputs)) and check it imports and launches.",
      {"op": str, "code": str})
async def compile_triton_kernel(args: dict) -> dict:
    spec, err = _require_op(args["op"])
    if err:
        return err
    op, code = args["op"], args["code"]
    kid = _kernel_id(code)
    (_op_dir(op) / f"{kid}.py").write_text(code)
    from .bench import runner
    res = runner.compile_check(spec, _op_dir(op) / f"{kid}.py")
    _save_sidecar(op, kid, compiles=res["compiles"], compile_log=res["log"])
    return _text({"kernel_id": kid, **res})


@tool("verify_kernel",
      "Check a saved kernel matches the PyTorch reference within tolerance across all shapes.",
      {"op": str, "kernel_id": str})
async def verify_kernel(args: dict) -> dict:
    spec, err = _require_op(args["op"])
    if err:
        return err
    op, kid = args["op"], args["kernel_id"]
    if not (_op_dir(op) / f"{kid}.py").exists():
        return _text({"error": f"kernel {kid} not found for op {op}"})
    from .bench import runner
    res = runner.verify(spec, _op_dir(op) / f"{kid}.py")
    _save_sidecar(op, kid, correct=res["correct"],
                  max_abs_err=res.get("max_abs_err"), max_rel_err=res.get("max_rel_err"),
                  verify=res["per_shape"])
    return _text({"kernel_id": kid, **res})


@tool("benchmark_kernel",
      "Benchmark a kernel vs the PyTorch reference; returns per-shape + geomean speedup and bandwidth.",
      {"op": str, "kernel_id": str})
async def benchmark_kernel(args: dict) -> dict:
    spec, err = _require_op(args["op"])
    if err:
        return err
    op, kid = args["op"], args["kernel_id"]
    if not (_op_dir(op) / f"{kid}.py").exists():
        return _text({"error": f"kernel {kid} not found for op {op}"})
    from .bench import runner
    res = runner.benchmark(spec, _op_dir(op) / f"{kid}.py")
    if res.get("ok"):
        correct = res["n_correct_shapes"] == res["n_shapes"]
        speedup = res["geomean_speedup"] if correct else 0.0
        _save_sidecar(op, kid, correct=correct, speedup=speedup, bench=res["per_shape"])
        new_record = False
        if correct and speedup > 0:
            new_record = _update_leaderboard(op, kid, speedup, res["per_shape"])
            if MEM.update_library(op, kernel_code(op, kid), speedup, kid, correct):
                new_record = True
        res["new_record"] = new_record
        res["target_speedup"] = config.TARGET_SPEEDUP
        res["beats_target"] = correct and speedup >= config.TARGET_SPEEDUP
    return _text({"kernel_id": kid, **res})


@tool("profile_kernel",
      "Profile a kernel to find the bottleneck (Nsight Compute if available, else a bandwidth/occupancy estimate).",
      {"op": str, "kernel_id": str})
async def profile_kernel(args: dict) -> dict:
    spec, err = _require_op(args["op"])
    if err:
        return err
    op, kid = args["op"], args["kernel_id"]
    side = _load_sidecar(op, kid)
    bench = side.get("bench")
    if not bench:
        # ensure we have timing first
        from .bench import runner
        b = runner.benchmark(spec, _op_dir(op) / f"{kid}.py")
        bench = b.get("per_shape", {})
        if b.get("ok"):
            _save_sidecar(op, kid, bench=bench)
    # Bandwidth-based bottleneck heuristic (robust; works without ncu perms).
    analysis = {}
    for shape_name, m in (bench or {}).items():
        pct = m.get("pct_peak_bw", 0)
        if pct >= 70:
            verdict = "memory-bandwidth bound (near roofline) — focus on reducing bytes moved / fusing"
        elif pct >= 30:
            verdict = "partially memory bound — try larger BLOCK_SIZE, vectorized loads, fewer passes"
        else:
            verdict = ("launch/occupancy bound — kernel-launch or low occupancy dominates; "
                       "raise occupancy (block/warps), reduce launches, increase work per program")
        analysis[shape_name] = {
            "kernel_ms": m.get("kernel_ms"), "achieved_gbps": m.get("achieved_gbps"),
            "pct_peak_bw": pct, "verdict": verdict,
        }
    out = {"kernel_id": kid, "ncu_available": bool(shutil.which("ncu")), "analysis": analysis,
           "note": "ncu profiling on consumer GPUs often needs elevated perms; "
                   "the bandwidth-roofline estimate above is the graceful fallback."}
    _save_sidecar(op, kid, profiled=True, profile=analysis)
    return _text(out)


@tool("read_memory", "Retrieve long-term strategy cards and the best-known kernel for an op.", {"op": str})
async def read_memory(args: dict) -> dict:
    op = args["op"]
    best = MEM.best_kernel(op)
    return _text({
        "op": op,
        "strategies": MEM.strategies_for(op),
        "best_kernel": ({"speedup": best["speedup"], "code": best["code"]} if best else None),
    })


def _parse_tags(raw) -> list:
    """Normalize a tags argument into a clean list of strings.

    Agents sometimes pass a real list, a comma-separated string, or a JSON-array
    *string* like '["fusion", "autotune"]'. Accept all three so a stray encoding
    never lands double-quoted/bracketed in long-term memory.
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        items = raw
    else:
        s = str(raw).strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                items = parsed if isinstance(parsed, list) else [parsed]
            except Exception:
                items = s.strip("[]").split(",")
        else:
            items = s.split(",")
    out = []
    for t in items:
        t = str(t).strip().strip("[]").strip().strip('"').strip("'").strip()
        if t:
            out.append(t)
    return out


@tool("record_strategy",
      "Append a reusable optimization lesson (what worked / why) to long-term memory for future runs.",
      {"op": str, "lesson": str, "tags": str, "evidence": str})
async def record_strategy(args: dict) -> dict:
    tags = _parse_tags(args.get("tags"))
    card = MEM.write_strategy(args["op"], args["lesson"], tags=tags, evidence=args.get("evidence", ""))
    return _text({"recorded": card})


@tool("get_leaderboard", "Show the current best kernel and speedup for every op.", {})
async def get_leaderboard(args: dict) -> dict:
    return _text(_read_leaderboard())


def build_server():
    """Create the in-process MCP server exposing the tools above."""
    return create_sdk_mcp_server(
        name="rsi",
        version="0.1.0",
        tools=[list_ops, get_op_spec, compile_triton_kernel, verify_kernel,
               benchmark_kernel, profile_kernel, read_memory, record_strategy, get_leaderboard],
    )


# Fully-qualified tool names (for allowed_tools).
TOOL_NAMES = [
    "mcp__rsi__list_ops", "mcp__rsi__get_op_spec", "mcp__rsi__compile_triton_kernel",
    "mcp__rsi__verify_kernel", "mcp__rsi__benchmark_kernel", "mcp__rsi__profile_kernel",
    "mcp__rsi__read_memory", "mcp__rsi__record_strategy", "mcp__rsi__get_leaderboard",
]
