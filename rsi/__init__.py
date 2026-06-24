"""RSI — Recursive kernel-optimization agent framework on the Claude Agent SDK.

A non-trainable, self-improving system: it gets better across runs by accumulating
*memory* (a long-term strategy/kernel library + short-term per-task trajectories)
and by recursive agent search — not by updating any model weights.

The flagship "skill" is a kernel-writing subagent that authors and optimizes Triton
kernels for small-LLM inference ops (Qwen3-1.7B shapes), verified for correctness
and benchmarked for per-kernel speedup against a PyTorch reference.
"""

__version__ = "0.1.0"
