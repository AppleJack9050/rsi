"""Minimal vLLM smoke test — must be run as a FILE (vLLM v1 spawns an EngineCore
subprocess that references __main__; stdin/heredoc breaks it)."""
import time
import torch
from vllm import LLM, SamplingParams


def main():
    t0 = time.time()
    llm = LLM(model="Qwen/Qwen3-1.7B", load_format="dummy",
              max_model_len=1024, gpu_memory_utilization=0.5,
              enforce_eager=False, disable_log_stats=True)
    print(f"[ENGINE READY in {time.time()-t0:.0f}s, CUDA graphs ON]")
    sp = SamplingParams(max_tokens=64, ignore_eos=True, temperature=0.0)
    prompts = ["The capital of France is"] * 128
    torch.cuda.synchronize(); t1 = time.time()
    outs = llm.generate(prompts, sp)
    torch.cuda.synchronize(); dt = time.time() - t1
    toks = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"[GEN {toks} tokens in {dt:.2f}s -> {toks/dt:.0f} tok/s aggregate @ batch={len(prompts)}]")


if __name__ == "__main__":
    main()
