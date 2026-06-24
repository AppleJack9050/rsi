"""Hooks: lightweight telemetry + defense-in-depth tool guardrails.

`allowed_tools` already restricts the agents to the rsi MCP tools (+ Task), but a
PreToolUse hook gives a second, explicit gate and an audit trail of every tool call.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from claude_agent_sdk import HookMatcher

from .tools import TOOL_NAMES

ALLOWED = set(TOOL_NAMES) | {"Task"}


def make_hooks(log_path: Path):
    """Build a hooks config dict for ClaudeAgentOptions, logging events to `log_path`."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(rec: dict) -> None:
        rec = {"ts": time.time(), **rec}
        with log_path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    async def pre_tool(input_data, tool_use_id, context):
        name = input_data.get("tool_name", "")
        _log({"event": "PreToolUse", "tool": name})
        # Block anything outside the rsi tool surface (e.g. stray Bash/Write/Edit).
        if name and name not in ALLOWED and not name.startswith("mcp__rsi__"):
            return {"hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": f"tool '{name}' is not permitted in the kernel sandbox",
            }}
        return {}

    async def post_tool(input_data, tool_use_id, context):
        _log({"event": "PostToolUse", "tool": input_data.get("tool_name", "")})
        return {}

    async def subagent_stop(input_data, tool_use_id, context):
        _log({"event": "SubagentStop"})
        return {}

    return {
        "PreToolUse": [HookMatcher(hooks=[pre_tool])],
        "PostToolUse": [HookMatcher(hooks=[post_tool])],
        "SubagentStop": [HookMatcher(hooks=[subagent_stop])],
    }
