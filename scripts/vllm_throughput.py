"""End-to-end throughput A/B: stock vLLM CUDA kernels vs vLLM + our Triton kernels.

Both arms use IDENTICAL settings (enforce_eager + vLLM custom ops enabled), so the
only difference is which kernels run RMSNorm / SiluAndMul. Set RSI_USE_TRITON=1 to use
our kernels (the source hooks in vllm layernorm.py / activation.py read that env var).

Must be run as a FILE (vLLM v1 spawns an EngineCore subprocess).

Usage: vllm_throughput.py [batch] [max_tokens] [prompt_len]
"""
import os
import sys
import time

import torch
from vllm import LLM, SamplingParams


def main():
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    prompt_len = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    mode = "RSI-TRITON" if os.environ.get("RSI_USE_TRITON") == "1" else "STOCK-vLLM"

    llm = LLM(
        model="Qwen/Qwen3-1.7B",
        load_format="dummy",
        max_model_len=max(2048, prompt_len + max_tokens + 8),
        gpu_memory_utilization=0.6,
        enforce_eager=True,                                   # no inductor (our Triton calls aren't traceable)
        compilation_config={"custom_ops": ["+rms_norm", "+silu_and_mul"]},  # dispatch forward_cuda
        disable_log_stats=True,
        seed=0,
    )
    # fixed-length prompts (token ids) so both arms do identical work
    prompt_ids = [{"prompt_token_ids": list(range(1, prompt_len + 1))} for _ in range(batch)]
    sp = SamplingParams(max_tokens=max_tokens, min_tokens=max_tokens,
                        ignore_eos=True, temperature=0.0)

    # warmup (also warms GPU clocks so steady-state measurements are stable)
    for _ in range(3):
        llm.generate(prompt_ids, sp, use_tqdm=False)
    torch.cuda.synchronize()

    # steady-state: many repeats in ONE process -> kills cross-process/cold-cache noise
    tputs = []
    reps = 12
    for _ in range(reps):
        t0 = time.time()
        outs = llm.generate(prompt_ids, sp, use_tqdm=False)
        torch.cuda.synchronize()
        dt = time.time() - t0
        gen = sum(len(o.outputs[0].token_ids) for o in outs)
        tputs.append(gen / dt)
    tputs.sort()
    med = tputs[len(tputs) // 2]
    print(f"RESULT mode={mode} batch={batch} max_tokens={max_tokens} reps={reps} "
          f"median={med:.0f} min={tputs[0]:.0f} max={tputs[-1]:.0f} tok/s")
    print(f"CHECKSUM first_seq_tokens={list(outs[0].outputs[0].token_ids)[:12]}")


if __name__ == "__main__":
    main()
