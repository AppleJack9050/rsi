"""Recursive control loop.

Python drives the OUTER loop (op queue, phases, budget, memory, leaderboard) for
reliability; each phase is one `query()` call to a kernel agent that autonomously
iterates compile->verify->benchmark via the MCP tools. A persistent dual-level memory
makes later runs start from prior winners — the non-trainable self-improvement.

Also provides `run_autonomous`: a single top-level Opus agent that delegates to the
kernel subagents via the Task tool (the "pure" hierarchical mode).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, ResultMessage,
                              TextBlock, query)

from . import config
from .agents import TRITON_GUIDE, agent_defs
from .bench.ops import DEFAULT_OP_ORDER
from .hooks import make_hooks
from .memory import MemoryStore
from .tools import TOOL_NAMES, best_on_disk, build_server


class Orchestrator:
    def __init__(self, run_id: str | None = None, *, per_run_usd: float | None = None,
                 permission_mode: str = "bypassPermissions"):
        # Guarantee subscription-only billing before any agent is spawned.
        self.auth = config.enforce_subscription_auth()
        self.run_id = run_id or f"run_{int(time.time())}"
        self.run_dir = config.RESULTS_DIR / "runs" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_log = self.run_dir / "events.jsonl"
        self.mem = MemoryStore()
        self.server = build_server()
        self.agents = agent_defs()
        self.hooks = make_hooks(self.events_log)
        self.permission_mode = permission_mode
        self.budget_cap = per_run_usd if per_run_usd is not None else config.BUDGET.per_run_usd
        self.spent = 0.0

    # ------------------------------------------------------------------ #
    # Options + phase runner
    # ------------------------------------------------------------------ #
    def _options(self, system_prompt: str, model: str, allowed: list[str], max_turns: int):
        return ClaudeAgentOptions(
            system_prompt=system_prompt,
            model=model,
            fallback_model=config.MODELS.fallback,
            mcp_servers={"rsi": self.server},
            agents=self.agents,
            allowed_tools=allowed,
            hooks=self.hooks,
            permission_mode=self.permission_mode,
            max_turns=max_turns,
            max_budget_usd=config.BUDGET.per_call_usd,
            effort=config.BUDGET.effort,
            cwd=str(config.PROJECT_ROOT),
            setting_sources=[],          # hermetic: ignore ambient CLAUDE.md / settings
        )

    async def _run_phase(self, *, label: str, system_prompt: str, model: str,
                         allowed: list[str], prompt: str, max_turns: int) -> dict:
        opts = self._options(system_prompt, model, allowed, max_turns)
        texts: list[str] = []
        cost = 0.0
        err = None
        try:
            async for msg in query(prompt=prompt, options=opts):
                if isinstance(msg, AssistantMessage):
                    for b in msg.content:
                        if isinstance(b, TextBlock):
                            texts.append(b.text)
                elif isinstance(msg, ResultMessage):
                    cost = msg.total_cost_usd or 0.0
                    if msg.is_error:
                        err = (msg.result or "result error")
        except Exception as e:  # CLI/auth/transport failure — keep the loop alive
            err = f"{type(e).__name__}: {e}"
        self.spent += cost
        self._log({"event": "phase", "label": label, "model": model,
                   "cost_usd": cost, "spent_usd": round(self.spent, 4), "error": err})
        return {"text": "\n".join(texts)[-4000:], "cost": cost, "error": err}

    def _log(self, rec: dict) -> None:
        rec = {"ts": time.time(), **rec}
        with self.events_log.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    # ------------------------------------------------------------------ #
    # Memory-injected prompts
    # ------------------------------------------------------------------ #
    def _memory_blurb(self, op: str, task_id: str) -> str:
        ctx = self.mem.context_for(op, task_id)
        lines = []
        if ctx["best_kernel"]:
            lines.append(f"A best-known kernel exists (geomean speedup "
                         f"{ctx['best_kernel']['speedup']:.3f}). Call read_memory('{op}') to get its code.")
        if ctx["strategies"]:
            lines.append("Prior winning strategies (reuse what applies):")
            for s in ctx["strategies"][:6]:
                tag = f" [{','.join(s.get('tags', []))}]" if s.get("tags") else ""
                lines.append(f"  - {s['lesson']}{tag}")
        if ctx["trajectory"]:
            lines.append("Recent attempts this run (avoid repeating dead ends):")
            for t in ctx["trajectory"][-4:]:
                lines.append(f"  - phase {t.get('phase')}: best_speedup={t.get('best_speedup')}")
        return "\n".join(lines) if lines else "(memory is empty for this op — you are the first attempt.)"

    # ------------------------------------------------------------------ #
    # Per-op optimization (the recursive refinement)
    # ------------------------------------------------------------------ #
    async def optimize_op(self, op: str, rounds: int | None = None) -> dict:
        rounds = rounds or config.BUDGET.rounds
        task_id = f"{self.run_id}_{op}"
        defs = self.agents
        self._log({"event": "op_start", "op": op, "rounds": rounds})

        # ---- Phase 0: seed (generator) ----
        seed_prompt = (
            f"Create a correct, fast Triton kernel for op '{op}'.\n"
            f"Start by calling get_op_spec('{op}') and read_memory('{op}').\n"
            f"MEMORY:\n{self._memory_blurb(op, task_id)}\n\n"
            "Then implement, compile_triton_kernel, verify_kernel (must pass), and "
            "benchmark_kernel. Report the final kernel_id and geomean speedup."
        )
        await self._run_phase(label=f"{op}:seed", system_prompt=defs["kernel-generator"].prompt,
                              model=config.MODELS.generator,
                              allowed=defs["kernel-generator"].tools,
                              prompt=seed_prompt, max_turns=config.BUDGET.max_turns_per_phase)
        best = best_on_disk(op)
        self.mem.append_trajectory(task_id, {"phase": "seed",
                                             "best_speedup": (best or {}).get("speedup"),
                                             "kernel_id": (best or {}).get("kernel_id")})

        # ---- Repair if no correct kernel yet ----
        if not best:
            repair_prompt = (
                f"No correct kernel exists yet for '{op}'. Inspect recent compile/verify failures, "
                f"call get_op_spec('{op}'), and produce a minimal CORRECT Triton kernel that "
                "compiles and verifies. Then benchmark it."
            )
            await self._run_phase(label=f"{op}:repair", system_prompt=defs["kernel-repairer"].prompt,
                                  model=config.MODELS.repairer,
                                  allowed=defs["kernel-repairer"].tools,
                                  prompt=repair_prompt, max_turns=config.BUDGET.max_turns_per_phase)
            best = best_on_disk(op)
            self.mem.append_trajectory(task_id, {"phase": "repair",
                                                 "best_speedup": (best or {}).get("speedup"),
                                                 "kernel_id": (best or {}).get("kernel_id")})

        # ---- Optimize phases ----
        for i in range(1, rounds):
            if self.spent >= self.budget_cap:
                self._log({"event": "budget_stop", "op": op, "spent": self.spent})
                break
            if not best:
                break  # still nothing correct; stop wasting budget
            opt_prompt = (
                f"Beat the current best kernel for '{op}' (geomean speedup {best['speedup']:.3f}, "
                f"kernel_id {best['kernel_id']}).\n"
                f"Call read_memory('{op}') for its code, then profile_kernel('{op}','{best['kernel_id']}').\n"
                f"MEMORY:\n{self._memory_blurb(op, task_id)}\n\n"
                "Make ONE targeted optimization, keep it correct (verify_kernel), benchmark_kernel. "
                "If you beat the best, call record_strategy with the lesson."
            )
            await self._run_phase(label=f"{op}:opt{i}", system_prompt=defs["kernel-optimizer"].prompt,
                                  model=config.MODELS.optimizer,
                                  allowed=defs["kernel-optimizer"].tools,
                                  prompt=opt_prompt, max_turns=config.BUDGET.max_turns_per_phase)
            best = best_on_disk(op)
            self.mem.append_trajectory(task_id, {"phase": f"opt{i}",
                                                 "best_speedup": (best or {}).get("speedup"),
                                                 "kernel_id": (best or {}).get("kernel_id")})

        summary = {"op": op, "best": best, "spent_usd": round(self.spent, 4)}
        self._log({"event": "op_done", **summary})
        return summary

    # ------------------------------------------------------------------ #
    # Full sweep with an outer recursive re-attack pass
    # ------------------------------------------------------------------ #
    async def optimize_all(self, ops: list[str] | None = None, rounds: int | None = None,
                           passes: int = 1) -> dict:
        ops = ops or DEFAULT_OP_ORDER
        results: dict[str, Any] = {}
        for p in range(passes):
            for op in ops:
                if self.spent >= self.budget_cap:
                    self._log({"event": "budget_stop_global", "spent": self.spent})
                    return {"results": results, "spent_usd": self.spent, "stopped": "budget"}
                best = best_on_disk(op)
                # outer recursion: on later passes, skip ops already past target
                if p > 0 and best and (best.get("speedup") or 0) >= config.TARGET_SPEEDUP:
                    continue
                results[op] = await self.optimize_op(op, rounds=rounds)
        return {"results": results, "spent_usd": round(self.spent, 4)}

    # ------------------------------------------------------------------ #
    # Pure hierarchical mode: one Opus orchestrator delegating via Task
    # ------------------------------------------------------------------ #
    async def run_autonomous(self, ops: list[str] | None = None) -> dict:
        ops = ops or DEFAULT_OP_ORDER
        system = (
            "You are the ORCHESTRATOR of a recursive kernel-optimization system. You do not write "
            "kernels yourself; you delegate to subagents via the Task tool and track progress.\n\n"
            + TRITON_GUIDE
            + "\nFor each target op: delegate to the 'kernel-generator' subagent to get a correct "
            "seed kernel, then repeatedly delegate to 'kernel-optimizer' to beat it, consulting "
            "memory. Use get_leaderboard to track best speedups and stop an op once it beats the "
            f"target ({config.TARGET_SPEEDUP}x) or yields no further gains."
        )
        prompt = (f"Optimize these ops in order: {ops}. "
                  f"Maximize geomean per-kernel speedup vs the PyTorch reference. "
                  f"Report the final leaderboard.")
        allowed = TOOL_NAMES + ["Task"]
        return await self._run_phase(label="autonomous", system_prompt=system,
                                     model=config.MODELS.orchestrator, allowed=allowed,
                                     prompt=prompt, max_turns=200)
