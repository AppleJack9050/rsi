"""Dual-level memory store (the KernelMem-style core of the framework's RSI).

Long-term  (persists across runs -> the cross-run self-improvement signal):
    longterm/strategies.jsonl     append-only optimization "strategy cards"
    longterm/library/<op>.py      best-known kernel source for each op
    longterm/library/<op>.json    its score + provenance

Short-term (per task, prevents oscillation within a run):
    shortterm/<task_id>.jsonl     refinement trajectory events

No torch needed — pure stdlib.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .. import config


class MemoryStore:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else config.MEMORY_DIR
        self.longterm = self.root / "longterm"
        self.library = self.longterm / "library"
        self.shortterm = self.root / "shortterm"
        for p in (self.longterm, self.library, self.shortterm):
            p.mkdir(parents=True, exist_ok=True)
        self.strategies_path = self.longterm / "strategies.jsonl"

    # ------------------------------------------------------------------ #
    # Long-term: strategy cards
    # ------------------------------------------------------------------ #
    def write_strategy(self, op: str, lesson: str, *, tags: list[str] | None = None,
                       evidence: str = "", speedup: float | None = None) -> dict:
        card = {
            "op": op,
            "lesson": lesson.strip(),
            "tags": tags or [],
            "evidence": evidence.strip(),
            "speedup": speedup,
            "ts": time.time(),
        }
        with self.strategies_path.open("a") as f:
            f.write(json.dumps(card) + "\n")
        return card

    def _all_strategies(self) -> list[dict]:
        if not self.strategies_path.exists():
            return []
        out = []
        for line in self.strategies_path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def strategies_for(self, op: str, limit: int = 12) -> list[dict]:
        cards = self._all_strategies()
        same = [c for c in cards if c.get("op") == op]
        other = [c for c in cards if c.get("op") != op]
        # prefer op-specific, then highest-speedup cross-op transfers
        other.sort(key=lambda c: (c.get("speedup") or 0), reverse=True)
        ranked = same[::-1] + other  # newest op-specific first
        return ranked[:limit]

    # ------------------------------------------------------------------ #
    # Long-term: best-kernel library
    # ------------------------------------------------------------------ #
    def best_kernel(self, op: str) -> dict | None:
        meta = self.library / f"{op}.json"
        code = self.library / f"{op}.py"
        if meta.exists() and code.exists():
            d = json.loads(meta.read_text())
            d["code"] = code.read_text()
            return d
        return None

    def update_library(self, op: str, code: str, speedup: float,
                       kernel_id: str = "", correct: bool = True) -> bool:
        """Store `code` as the op's best kernel iff it beats the current record. Returns True if updated."""
        if not correct:
            return False
        cur = self.best_kernel(op)
        if cur and (cur.get("speedup") or 0) >= speedup:
            return False
        (self.library / f"{op}.py").write_text(code)
        (self.library / f"{op}.json").write_text(json.dumps({
            "op": op, "speedup": speedup, "kernel_id": kernel_id, "ts": time.time(),
        }, indent=2))
        return True

    # ------------------------------------------------------------------ #
    # Short-term: per-task trajectory
    # ------------------------------------------------------------------ #
    def append_trajectory(self, task_id: str, event: dict) -> None:
        event = {"ts": time.time(), **event}
        with (self.shortterm / f"{task_id}.jsonl").open("a") as f:
            f.write(json.dumps(event) + "\n")

    def read_trajectory(self, task_id: str, last_n: int = 8) -> list[dict]:
        p = self.shortterm / f"{task_id}.jsonl"
        if not p.exists():
            return []
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        return rows[-last_n:]

    # ------------------------------------------------------------------ #
    # Retrieval bundle for prompting
    # ------------------------------------------------------------------ #
    def context_for(self, op: str, task_id: str | None = None) -> dict:
        """Everything the agent should see before a refinement step."""
        best = self.best_kernel(op)
        return {
            "strategies": self.strategies_for(op),
            "best_kernel": ({"speedup": best["speedup"], "code": best["code"]} if best else None),
            "trajectory": self.read_trajectory(task_id) if task_id else [],
        }
