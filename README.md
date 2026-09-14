# mini-vllm

From-scratch LLM inference engine built one technique at a time.

Every technique is a flag on a shared `EngineConfig` dataclass so a baseline → +feature ablation table is possible at the end.

**Model:** Qwen3-0.6B  
**Hardware:** M4 Mac Air (MPS)

## Run the benchmark harness

```bash
python -m mini_vllm.bench.harness
```

## Structure

```
mini_vllm/
  engine/       model runner, KV cache, scheduler, sampler
  kernels/      custom ops
  serving/      HTTP server and router
  bench/        benchmark harness + workloads
  instrumentation/ metrics logging
  configs/      EngineConfig flags
```
