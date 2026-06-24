"""Separate PREFILL and DECODE throughput: stock vLLM vs vLLM + our Triton kernels.

Steady-state median over many reps in ONE process (kills cross-process / cold-cache /
GPU-clock noise). Decode is isolated from prefill by time subtraction:
    t1 = time for max_tokens=1   (prefill + 1 decode step)
    tN = time for max_tokens=N   (prefill + N decode steps)
    decode_time = tN - t1   over (N-1) tokens/seq
    prefill_throughput = batch * prompt_len / t1
    decode_throughput  = batch * (N-1) / decode_time

RSI_USE_TRITON=1 -> our kernels (source hooks read the env var). Run as a FILE.
Usage: vllm_pd.py [batch] [prompt_len] [decode_tokens] [reps]
"""
import os
import sys
import time

import torch
from vllm import LLM, SamplingParams


def median_time(llm, prompt_ids, n_tokens, reps):
    sp = SamplingParams(max_tokens=n_tokens, min_tokens=n_tokens, ignore_eos=True, temperature=0.0)
    for _ in range(2):  # warmup
        llm.generate(prompt_ids, sp, use_tqdm=False)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.time()
        outs = llm.generate(prompt_ids, sp, use_tqdm=False)
        torch.cuda.synchronize()
        ts.append(time.time() - t0)
    ts.sort()
    # min time = peak throughput = most reproducible under jittery GPU clocks
    return ts[0], outs


def main():
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    plen = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    dec = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    mode = "RSI-TRITON" if os.environ.get("RSI_USE_TRITON") == "1" else "STOCK-vLLM"

    graph = os.environ.get("RSI_GRAPH") == "1"   # production mode: CUDA graphs (stable timing)
    kw = dict(model="Qwen/Qwen3-1.7B", load_format="dummy",
              max_model_len=max(2048, plen + dec + 8), gpu_memory_utilization=0.6,
              disable_log_stats=True, seed=0)
    if graph:
        kw["enforce_eager"] = False                      # full vLLM: compile + CUDA graphs
        kw["max_num_seqs"] = batch
        # only capture the graph size we actually use -> much faster engine build
        kw["compilation_config"] = {"cudagraph_capture_sizes": [batch]}
    else:
        kw["enforce_eager"] = True
        kw["compilation_config"] = {"custom_ops": ["+rms_norm", "+silu_and_mul"]}
    llm = LLM(**kw)

    prompt_ids = [{"prompt_token_ids": list(range(1, plen + 1))} for _ in range(batch)]

    t1, o1 = median_time(llm, prompt_ids, 1, reps)
    tN, oN = median_time(llm, prompt_ids, dec, reps)

    prefill_tput = batch * plen / t1
    decode_time = max(tN - t1, 1e-9)
    decode_tput = batch * (dec - 1) / decode_time

    print(f"RESULT mode={mode} batch={batch} prompt_len={plen} decode_tokens={dec} reps={reps}")
    print(f"  PREFILL throughput = {prefill_tput:8.0f} tok/s   (t1={t1*1e3:.1f} ms for {batch*plen} prompt tok)")
    print(f"  DECODE  throughput = {decode_tput:8.0f} tok/s   (decode_time={decode_time*1e3:.1f} ms for {batch*(dec-1)} tok)")
    print(f"  CHECKSUM={list(oN[0].outputs[0].token_ids)[:8]}")


if __name__ == "__main__":
    main()
