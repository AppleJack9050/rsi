"""Central configuration: model dims, agent models, budgets, paths.

Importable WITHOUT torch/triton installed (no heavy imports here) so that
`rsi --help` and config inspection work on any machine.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KERNELS_DIR = PROJECT_ROOT / "rsi" / "kernels"        # generated kernels (per op)
MEMORY_DIR = PROJECT_ROOT / "rsi" / "memory"          # dual-level memory store
RESULTS_DIR = PROJECT_ROOT / "results"                # leaderboard + run logs
LEADERBOARD = RESULTS_DIR / "leaderboard.json"

for _p in (KERNELS_DIR, MEMORY_DIR, RESULTS_DIR):
    _p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Target model dims.
#
# Qwen3.5-2B is not publicly documented; we use Qwen3-1.7B as the concrete proxy.
# Everything is overridable via env so a real Qwen3.5-2B config drops in later.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelDims:
    name: str = os.environ.get("RSI_MODEL_NAME", "Qwen3-1.7B (proxy for Qwen3.5-2B)")
    hidden: int = int(os.environ.get("RSI_HIDDEN", 2048))
    n_layers: int = int(os.environ.get("RSI_N_LAYERS", 28))
    n_q_heads: int = int(os.environ.get("RSI_N_Q_HEADS", 16))
    n_kv_heads: int = int(os.environ.get("RSI_N_KV_HEADS", 8))      # GQA
    head_dim: int = int(os.environ.get("RSI_HEAD_DIM", 128))
    intermediate: int = int(os.environ.get("RSI_INTERMEDIATE", 6144))  # SwiGLU
    eps: float = float(os.environ.get("RSI_EPS", 1e-6))
    dtype: str = os.environ.get("RSI_DTYPE", "float16")            # float16 | bfloat16


DIMS = ModelDims()


# --------------------------------------------------------------------------- #
# Hardware
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Hardware:
    name: str = os.environ.get("RSI_GPU_NAME", "NVIDIA GeForce RTX 5090")
    arch: str = os.environ.get("RSI_GPU_ARCH", "Blackwell sm_120")
    # Theoretical peak HBM bandwidth (GB/s) used to score memory-bound kernels.
    peak_bw_gbps: float = float(os.environ.get("RSI_PEAK_BW_GBPS", 1792.0))  # RTX 5090 ~1.79 TB/s


HW = Hardware()


# --------------------------------------------------------------------------- #
# Agent models (Claude aliases — resolve to latest of each tier)
# orchestrator -> opus, kernel coders -> sonnet, cheap analysis -> haiku
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Models:
    orchestrator: str = os.environ.get("RSI_MODEL_ORCH", "opus")
    generator: str = os.environ.get("RSI_MODEL_GEN", "sonnet")
    optimizer: str = os.environ.get("RSI_MODEL_OPT", "sonnet")
    repairer: str = os.environ.get("RSI_MODEL_REP", "sonnet")
    analyst: str = os.environ.get("RSI_MODEL_ANALYST", "haiku")
    fallback: str = os.environ.get("RSI_MODEL_FALLBACK", "sonnet")


MODELS = Models()


# --------------------------------------------------------------------------- #
# Run budgets / loop control
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Budget:
    # Hard USD cap per query() call AND tracked cumulatively by the orchestrator.
    per_call_usd: float = float(os.environ.get("RSI_PER_CALL_USD", 2.0))
    per_run_usd: float = float(os.environ.get("RSI_PER_RUN_USD", 25.0))
    # Agentic turn cap inside one query() (lets the agent iterate compile->verify->bench).
    max_turns_per_phase: int = int(os.environ.get("RSI_MAX_TURNS", 40))
    # Refinement phases per op (1 seed + N optimize/repair).
    rounds: int = int(os.environ.get("RSI_ROUNDS", 4))
    effort: str = os.environ.get("RSI_EFFORT", "high")  # low|medium|high|xhigh|max
    # Let the optimizer spawn a profiler-analyst sub-agent via Task. OFF by default:
    # nested LLM loops roughly double rate-limit/token use per phase. The Python outer
    # loop + cross-run memory already provide recursion; `rsi optimize --autonomous`
    # is the dedicated hierarchical-delegation showcase.
    delegate: bool = os.environ.get("RSI_DELEGATE", "0") == "1"


BUDGET = Budget()

# A kernel counts as "beating the reference" at/above this geomean speedup.
TARGET_SPEEDUP = float(os.environ.get("RSI_TARGET_SPEEDUP", 1.15))


# --------------------------------------------------------------------------- #
# Auth guard — guarantee SDK-spawned agents bill the Claude SUBSCRIPTION, never
# a metered API key. The claude-agent-sdk launches the `claude` CLI as a child
# process which inherits this process's environment; if an ANTHROPIC_API_KEY (or
# an alternate-provider flag) is present it would bill that instead of the
# ~/.claude OAuth subscription token. We strip those vars from THIS process
# (does not touch the user's shell) so the child can only fall back to the
# subscription, and we refuse to run if a key was found.
# --------------------------------------------------------------------------- #
# Env vars that would divert billing away from the subscription OAuth token.
_API_BILLING_VARS = (
    "ANTHROPIC_API_KEY",       # metered first-party API billing
    "ANTHROPIC_AUTH_TOKEN",    # bearer-token override
    "ANTHROPIC_BASE_URL",      # custom/proxy endpoint
    "CLAUDE_CODE_USE_BEDROCK",  # routes to AWS Bedrock billing
    "CLAUDE_CODE_USE_VERTEX",   # routes to GCP Vertex billing
)


def enforce_subscription_auth(*, strict: bool = True) -> dict:
    """Force subscription-only auth for any agents this process spawns.

    Removes every API-billing / alternate-provider env var from `os.environ` so
    the spawned `claude` CLI can only authenticate with the ~/.claude OAuth
    (subscription) token. Returns a status dict. With ``strict=True`` (default)
    raises RuntimeError if an API key/token was set or if no subscription token
    is present — so a misconfigured environment fails loudly instead of silently
    billing an API key.
    """
    import json

    removed = []
    for var in _API_BILLING_VARS:
        if os.environ.pop(var, None) is not None:
            removed.append(var)

    cred = Path.home() / ".claude" / ".credentials.json"
    sub_type = None
    has_sub = False
    if cred.exists():
        try:
            oauth = json.loads(cred.read_text()).get("claudeAiOauth", {})
            has_sub = bool(oauth.get("accessToken"))
            sub_type = oauth.get("subscriptionType")
        except Exception:
            pass

    status = {"removed": removed, "subscription_present": has_sub,
              "subscription_type": sub_type}

    if strict:
        diverting = [v for v in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN") if v in removed]
        if diverting:
            raise RuntimeError(
                f"Refusing to run: {', '.join(diverting)} was set and would bill the metered "
                f"API instead of your subscription. It has been removed from this process; "
                f"unset it in your shell to run on the subscription."
            )
        if not has_sub:
            raise RuntimeError(
                "No Claude subscription OAuth token found at ~/.claude/.credentials.json. "
                "Log in with your subscription (run `claude` and use /login) before running rsi."
            )
    return status
