"""Agent definitions for the kernel-optimization subagents.

These `AgentDefinition`s are registered in `ClaudeAgentOptions.agents` so the orchestrator
(or any agent) can delegate to them via the Task tool (sub-subagents). Their system prompts
are also reused by the Python-driven phase loop in `orchestrator.py`.
"""
from __future__ import annotations

from claude_agent_sdk import AgentDefinition

from . import config
from .tools import TOOL_NAMES

T = {n.split("__")[-1]: n for n in TOOL_NAMES}  # short -> fully-qualified tool name

# --------------------------------------------------------------------------- #
# Shared Triton guidance injected into every kernel agent.
# --------------------------------------------------------------------------- #
TRITON_GUIDE = f"""\
You write and optimize **Triton** GPU kernels for {config.HW.name} ({config.HW.arch}).
Target model: {config.DIMS.name} (dtype {config.DIMS.dtype}).

KERNEL CONTRACT (strict):
- Output ONE Python module as the `code` argument to compile_triton_kernel.
- It must `import torch, triton, triton.language as tl`, define one or more `@triton.jit`
  kernels, and expose `run(*inputs)` taking inputs in the documented order and returning a
  SINGLE tensor that matches the reference within the op's tolerance.
- Allocate the output inside `run` with torch.empty/empty_like; compute the launch grid;
  launch the kernel; return the tensor. Do NOT call .item()/print or read env.

TRITON CRAFT:
- Use `tl.program_id`, `BLOCK_SIZE: tl.constexpr`, and masks (`mask = offs < N`) for bounds.
- Accumulate reductions (mean, sum, softmax) in fp32 (`.to(tl.float32)`), then cast back.
- For row-wise ops, map one program to one row (or a tile of rows) and loop over columns
  in BLOCK_SIZE chunks; keep data in registers/SRAM to avoid extra DRAM passes.
- Tune `BLOCK_SIZE` (powers of two), `num_warps`, `num_stages`. Fuse elementwise chains.
- These decode/MLP/norm ops are MEMORY-BANDWIDTH bound: the win comes from fusing multiple
  PyTorch kernel launches into one and minimizing bytes moved — not from clever math.

WORKFLOW (use the tools, do not guess results):
1. get_op_spec(op) to read exact semantics, shapes, tolerance, signature.
2. read_memory(op) to reuse prior winning strategies and the best-known kernel.
3. compile_triton_kernel(op, code) -> fix any compile/launch error and retry.
4. verify_kernel(op, kernel_id) -> MUST be correct before optimizing.
5. benchmark_kernel(op, kernel_id) -> read the geomean speedup.
Correctness first, then speed. Iterate within your turn budget; report the final
kernel_id and its geomean speedup.
"""


def agent_defs() -> dict[str, AgentDefinition]:
    return {
        "kernel-generator": AgentDefinition(
            description="Writes a correct, reasonably fast seed Triton kernel for a target op.",
            prompt=(TRITON_GUIDE + "\nROLE: GENERATOR. Produce a CORRECT seed kernel first; "
                    "prefer a simple fused implementation that passes verification, then take one "
                    "or two easy speed wins. Do not over-engineer the seed."),
            tools=[T["get_op_spec"], T["read_memory"], T["compile_triton_kernel"],
                   T["verify_kernel"], T["benchmark_kernel"]],
            model=config.MODELS.generator,
            maxTurns=config.BUDGET.max_turns_per_phase,
        ),
        "kernel-optimizer": AgentDefinition(
            description="Improves a working Triton kernel using profiling feedback and memory.",
            prompt=(TRITON_GUIDE + "\nROLE: OPTIMIZER. Start from the best-known kernel. "
                    "profile_kernel to find the bottleneck, retrieve strategies from memory, then "
                    "make ONE targeted change per attempt (block size, vectorization, fusion, warps). "
                    "Keep every change correct. When you beat the previous best, call record_strategy "
                    "with the lesson (what change, why it helped, the speedup) so future runs reuse it."
                    + (" You may delegate a profiling read to the profiler-analyst subagent via Task."
                       if config.BUDGET.delegate else "")),
            tools=[T["get_op_spec"], T["read_memory"], T["compile_triton_kernel"],
                   T["verify_kernel"], T["benchmark_kernel"], T["profile_kernel"],
                   T["record_strategy"]] + (["Task"] if config.BUDGET.delegate else []),
            model=config.MODELS.optimizer,
            maxTurns=config.BUDGET.max_turns_per_phase,
            effort=config.BUDGET.effort,
        ),
        "kernel-repairer": AgentDefinition(
            description="Fixes a Triton kernel that fails to compile, launch, or verify.",
            prompt=(TRITON_GUIDE + "\nROLE: REPAIRER. You are given the failing code and its error "
                    "log. Diagnose the root cause (shape/mask/dtype/grid/indexing) and produce a "
                    "minimal corrected kernel that compiles AND verifies. Do not chase speed yet."),
            tools=[T["get_op_spec"], T["read_memory"], T["compile_triton_kernel"], T["verify_kernel"]],
            model=config.MODELS.repairer,
            maxTurns=config.BUDGET.max_turns_per_phase,
        ),
        "profiler-analyst": AgentDefinition(
            description="Reads a kernel's profile and names the dominant bottleneck + next fix.",
            prompt=("ROLE: PROFILER-ANALYST. Call profile_kernel for the given op+kernel_id, then "
                    "return a 2-3 sentence diagnosis: the dominant bottleneck and the single most "
                    "promising next optimization. Be concrete and brief."),
            tools=[T["profile_kernel"], T["read_memory"]],
            model=config.MODELS.analyst,
            maxTurns=6,
        ),
    }
